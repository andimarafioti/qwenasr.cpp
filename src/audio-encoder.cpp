#include "audio-encoder.h"

#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

static void set_error(std::string * error, const std::string & message) {
    if (error) {
        *error = message;
    }
}

static bool require_f32_tensor(
    const QwenAsrGgufModel & model,
    const std::string & name,
    QwenAsrGgufTensorView * out,
    std::string * error) {
    if (!qwenasr_gguf_model_tensor_by_name(model, name.c_str(), out, error)) {
        return false;
    }
    if (out->type != GGML_TYPE_F32) {
        set_error(error, "tensor is not f32: " + name);
        return false;
    }
    return true;
}

static const float * tensor_f32(const QwenAsrGgufTensorView & tensor) {
    return static_cast<const float *>(tensor.data);
}

static bool require_shape(
    const QwenAsrGgufTensorView & tensor,
    int64_t ne0,
    int64_t ne1,
    std::string * error) {
    if (tensor.ne.size() != 2 || tensor.ne[0] != ne0 || tensor.ne[1] != ne1) {
        set_error(error, "tensor shape mismatch: " + tensor.name);
        return false;
    }
    return true;
}

static bool require_shape(
    const QwenAsrGgufTensorView & tensor,
    int64_t ne0,
    std::string * error) {
    if (tensor.ne.size() != 1 || tensor.ne[0] != ne0) {
        set_error(error, "tensor shape mismatch: " + tensor.name);
        return false;
    }
    return true;
}

static void layer_norm(
    const std::vector<float> & input,
    int rows,
    int hidden,
    const float * weight,
    const float * bias,
    std::vector<float> * output) {
    output->assign(static_cast<size_t>(rows) * hidden, 0.0f);
    constexpr float eps = 1.0e-5f;
    for (int row = 0; row < rows; ++row) {
        const float * src = input.data() + static_cast<size_t>(row) * hidden;
        float * dst = output->data() + static_cast<size_t>(row) * hidden;
        double mean = 0.0;
        for (int i = 0; i < hidden; ++i) {
            mean += src[i];
        }
        mean /= static_cast<double>(hidden);
        double var = 0.0;
        for (int i = 0; i < hidden; ++i) {
            const double delta = static_cast<double>(src[i]) - mean;
            var += delta * delta;
        }
        var /= static_cast<double>(hidden);
        const float inv_std = 1.0f / std::sqrt(static_cast<float>(var) + eps);
        for (int i = 0; i < hidden; ++i) {
            dst[i] = (src[i] - static_cast<float>(mean)) * inv_std * weight[i] + bias[i];
        }
    }
}

static void linear(
    const std::vector<float> & input,
    int rows,
    int in_dim,
    int out_dim,
    const QwenAsrGgufTensorView & weight,
    const QwenAsrGgufTensorView & bias,
    std::vector<float> * output) {
    const float * w = tensor_f32(weight);
    const float * b = tensor_f32(bias);
    output->assign(static_cast<size_t>(rows) * out_dim, 0.0f);
    for (int row = 0; row < rows; ++row) {
        const float * src = input.data() + static_cast<size_t>(row) * in_dim;
        float * dst = output->data() + static_cast<size_t>(row) * out_dim;
        for (int out = 0; out < out_dim; ++out) {
            float sum = b[out];
            const float * ww = w + static_cast<size_t>(out) * in_dim;
            for (int in = 0; in < in_dim; ++in) {
                sum += src[in] * ww[in];
            }
            dst[out] = sum;
        }
    }
}

static float gelu(float x) {
    return 0.5f * x * (1.0f + std::erf(x * 0.70710678118654752440f));
}

static void audio_attention_full(
    const std::vector<float> & q,
    const std::vector<float> & k,
    const std::vector<float> & v,
    int tokens,
    int hidden,
    int n_heads,
    std::vector<float> * output) {
    output->assign(static_cast<size_t>(tokens) * hidden, 0.0f);
    const int head_dim = hidden / n_heads;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    std::vector<float> scores(static_cast<size_t>(tokens), 0.0f);

    for (int head = 0; head < n_heads; ++head) {
        const int base = head * head_dim;
        for (int qi = 0; qi < tokens; ++qi) {
            float max_score = -std::numeric_limits<float>::infinity();
            const float * qrow = q.data() + static_cast<size_t>(qi) * hidden + base;
            for (int kj = 0; kj < tokens; ++kj) {
                const float * krow = k.data() + static_cast<size_t>(kj) * hidden + base;
                float score = 0.0f;
                for (int d = 0; d < head_dim; ++d) {
                    score += qrow[d] * krow[d];
                }
                score *= scale;
                scores[static_cast<size_t>(kj)] = score;
                max_score = std::max(max_score, score);
            }
            float denom = 0.0f;
            for (int kj = 0; kj < tokens; ++kj) {
                const float value = std::exp(scores[static_cast<size_t>(kj)] - max_score);
                scores[static_cast<size_t>(kj)] = value;
                denom += value;
            }
            for (int d = 0; d < head_dim; ++d) {
                float sum = 0.0f;
                for (int kj = 0; kj < tokens; ++kj) {
                    const float prob = scores[static_cast<size_t>(kj)] / denom;
                    const float * vrow = v.data() + static_cast<size_t>(kj) * hidden + base;
                    sum += prob * vrow[d];
                }
                (*output)[static_cast<size_t>(qi) * hidden + base + d] = sum;
            }
        }
    }
}

static void add_inplace(std::vector<float> * a, const std::vector<float> & b) {
    for (size_t i = 0; i < a->size(); ++i) {
        (*a)[i] += b[i];
    }
}

bool qwenasr_audio_layer0_forward_cpu(
    const QwenAsrGgufModel & model,
    const QwenAsrFeatures & features,
    int n_threads,
    int n_heads,
    QwenAsrAudioLayerOutput * out,
    std::string * error) {
    if (!out) {
        set_error(error, "qwenasr_audio_layer0_forward_cpu: out is null");
        return false;
    }

    QwenAsrAudioPrepOutput prep;
    if (!qwenasr_audio_prep_forward_ggml(model, features, n_threads, &prep, error)) {
        return false;
    }
    const int tokens = prep.tokens;
    const int hidden = prep.hidden;
    if (tokens <= 0 || hidden <= 0 || n_heads <= 0 || hidden % n_heads != 0) {
        set_error(error, "invalid audio layer geometry");
        return false;
    }

    const std::string prefix = "audio.blk.0.";
    QwenAsrGgufTensorView attn_norm_w;
    QwenAsrGgufTensorView attn_norm_b;
    QwenAsrGgufTensorView ffn_norm_w;
    QwenAsrGgufTensorView ffn_norm_b;
    QwenAsrGgufTensorView q_w;
    QwenAsrGgufTensorView q_b;
    QwenAsrGgufTensorView k_w;
    QwenAsrGgufTensorView k_b;
    QwenAsrGgufTensorView v_w;
    QwenAsrGgufTensorView v_b;
    QwenAsrGgufTensorView out_w;
    QwenAsrGgufTensorView out_b;
    QwenAsrGgufTensorView up_w;
    QwenAsrGgufTensorView up_b;
    QwenAsrGgufTensorView down_w;
    QwenAsrGgufTensorView down_b;
    if (!require_f32_tensor(model, prefix + "attn_norm.weight", &attn_norm_w, error) ||
        !require_f32_tensor(model, prefix + "attn_norm.bias", &attn_norm_b, error) ||
        !require_f32_tensor(model, prefix + "ffn_norm.weight", &ffn_norm_w, error) ||
        !require_f32_tensor(model, prefix + "ffn_norm.bias", &ffn_norm_b, error) ||
        !require_f32_tensor(model, prefix + "attn_q.weight", &q_w, error) ||
        !require_f32_tensor(model, prefix + "attn_q.bias", &q_b, error) ||
        !require_f32_tensor(model, prefix + "attn_k.weight", &k_w, error) ||
        !require_f32_tensor(model, prefix + "attn_k.bias", &k_b, error) ||
        !require_f32_tensor(model, prefix + "attn_v.weight", &v_w, error) ||
        !require_f32_tensor(model, prefix + "attn_v.bias", &v_b, error) ||
        !require_f32_tensor(model, prefix + "attn_output.weight", &out_w, error) ||
        !require_f32_tensor(model, prefix + "attn_output.bias", &out_b, error) ||
        !require_f32_tensor(model, prefix + "ffn_up.weight", &up_w, error) ||
        !require_f32_tensor(model, prefix + "ffn_up.bias", &up_b, error) ||
        !require_f32_tensor(model, prefix + "ffn_down.weight", &down_w, error) ||
        !require_f32_tensor(model, prefix + "ffn_down.bias", &down_b, error)) {
        return false;
    }

    const int ffn = static_cast<int>(up_w.ne.size() == 2 ? up_w.ne[1] : 0);
    if (!require_shape(attn_norm_w, hidden, error) ||
        !require_shape(attn_norm_b, hidden, error) ||
        !require_shape(ffn_norm_w, hidden, error) ||
        !require_shape(ffn_norm_b, hidden, error) ||
        !require_shape(q_w, hidden, hidden, error) ||
        !require_shape(q_b, hidden, error) ||
        !require_shape(k_w, hidden, hidden, error) ||
        !require_shape(k_b, hidden, error) ||
        !require_shape(v_w, hidden, hidden, error) ||
        !require_shape(v_b, hidden, error) ||
        !require_shape(out_w, hidden, hidden, error) ||
        !require_shape(out_b, hidden, error) ||
        !require_shape(up_w, hidden, ffn, error) ||
        !require_shape(up_b, ffn, error) ||
        !require_shape(down_w, ffn, hidden, error) ||
        !require_shape(down_b, hidden, error)) {
        return false;
    }

    std::vector<float> normed;
    std::vector<float> q;
    std::vector<float> k;
    std::vector<float> v;
    std::vector<float> attn;
    std::vector<float> projected;
    layer_norm(prep.values, tokens, hidden, tensor_f32(attn_norm_w), tensor_f32(attn_norm_b), &normed);
    linear(normed, tokens, hidden, hidden, q_w, q_b, &q);
    linear(normed, tokens, hidden, hidden, k_w, k_b, &k);
    linear(normed, tokens, hidden, hidden, v_w, v_b, &v);
    audio_attention_full(q, k, v, tokens, hidden, n_heads, &attn);
    linear(attn, tokens, hidden, hidden, out_w, out_b, &projected);

    std::vector<float> hidden_states = prep.values;
    add_inplace(&hidden_states, projected);

    std::vector<float> ffn_normed;
    std::vector<float> ffn_up;
    std::vector<float> ffn_down;
    layer_norm(hidden_states, tokens, hidden, tensor_f32(ffn_norm_w), tensor_f32(ffn_norm_b), &ffn_normed);
    linear(ffn_normed, tokens, hidden, ffn, up_w, up_b, &ffn_up);
    for (float & value : ffn_up) {
        value = gelu(value);
    }
    linear(ffn_up, tokens, ffn, hidden, down_w, down_b, &ffn_down);
    add_inplace(&hidden_states, ffn_down);

    out->tokens = tokens;
    out->hidden = hidden;
    out->attention_segments = prep.attention_segments;
    out->values = std::move(hidden_states);
    return true;
}
