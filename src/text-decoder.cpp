#include "text-decoder.h"

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

static const float * tensor_f32(const QwenAsrGgufTensorView & tensor) {
    return static_cast<const float *>(tensor.data);
}

struct TextLayerTensors {
    QwenAsrGgufTensorView attn_norm_w;
    QwenAsrGgufTensorView ffn_norm_w;
    QwenAsrGgufTensorView q_w;
    QwenAsrGgufTensorView k_w;
    QwenAsrGgufTensorView v_w;
    QwenAsrGgufTensorView out_w;
    QwenAsrGgufTensorView q_norm_w;
    QwenAsrGgufTensorView k_norm_w;
    QwenAsrGgufTensorView gate_w;
    QwenAsrGgufTensorView up_w;
    QwenAsrGgufTensorView down_w;
};

static bool load_text_layer_tensors(
    const QwenAsrGgufModel & model,
    int layer,
    int hidden,
    int q_dim,
    int kv_dim,
    int head_dim,
    int intermediate,
    TextLayerTensors * tensors,
    std::string * error) {
    const std::string prefix = "text.blk." + std::to_string(layer) + ".";
    if (!require_f32_tensor(model, prefix + "attn_norm.weight", &tensors->attn_norm_w, error) ||
        !require_f32_tensor(model, prefix + "ffn_norm.weight", &tensors->ffn_norm_w, error) ||
        !require_f32_tensor(model, prefix + "attn_q.weight", &tensors->q_w, error) ||
        !require_f32_tensor(model, prefix + "attn_k.weight", &tensors->k_w, error) ||
        !require_f32_tensor(model, prefix + "attn_v.weight", &tensors->v_w, error) ||
        !require_f32_tensor(model, prefix + "attn_output.weight", &tensors->out_w, error) ||
        !require_f32_tensor(model, prefix + "attn_q_norm.weight", &tensors->q_norm_w, error) ||
        !require_f32_tensor(model, prefix + "attn_k_norm.weight", &tensors->k_norm_w, error) ||
        !require_f32_tensor(model, prefix + "ffn_gate.weight", &tensors->gate_w, error) ||
        !require_f32_tensor(model, prefix + "ffn_up.weight", &tensors->up_w, error) ||
        !require_f32_tensor(model, prefix + "ffn_down.weight", &tensors->down_w, error)) {
        return false;
    }

    return
        require_shape(tensors->attn_norm_w, hidden, error) &&
        require_shape(tensors->ffn_norm_w, hidden, error) &&
        require_shape(tensors->q_w, hidden, q_dim, error) &&
        require_shape(tensors->k_w, hidden, kv_dim, error) &&
        require_shape(tensors->v_w, hidden, kv_dim, error) &&
        require_shape(tensors->out_w, q_dim, hidden, error) &&
        require_shape(tensors->q_norm_w, head_dim, error) &&
        require_shape(tensors->k_norm_w, head_dim, error) &&
        require_shape(tensors->gate_w, hidden, intermediate, error) &&
        require_shape(tensors->up_w, hidden, intermediate, error) &&
        require_shape(tensors->down_w, intermediate, hidden, error);
}

static void rms_norm_row(
    const float * input,
    int size,
    const float * weight,
    float eps,
    float * output) {
    double mean_square = 0.0;
    for (int i = 0; i < size; ++i) {
        mean_square += static_cast<double>(input[i]) * input[i];
    }
    mean_square /= static_cast<double>(size);
    const float scale = 1.0f / std::sqrt(static_cast<float>(mean_square) + eps);
    for (int i = 0; i < size; ++i) {
        output[i] = input[i] * scale * weight[i];
    }
}

static void linear(
    const std::vector<float> & input,
    int rows,
    int in_dim,
    int out_dim,
    const QwenAsrGgufTensorView & weight,
    std::vector<float> * output) {
    const float * w = tensor_f32(weight);
    output->assign(static_cast<size_t>(rows) * out_dim, 0.0f);
    for (int row = 0; row < rows; ++row) {
        const float * src = input.data() + static_cast<size_t>(row) * in_dim;
        float * dst = output->data() + static_cast<size_t>(row) * out_dim;
        for (int out = 0; out < out_dim; ++out) {
            float sum = 0.0f;
            const float * ww = w + static_cast<size_t>(out) * in_dim;
            for (int in = 0; in < in_dim; ++in) {
                sum += src[in] * ww[in];
            }
            dst[out] = sum;
        }
    }
}

static float silu(float x) {
    return x / (1.0f + std::exp(-x));
}

static void apply_rope(
    std::vector<float> * states,
    int tokens,
    int n_heads,
    int head_dim,
    float rope_theta) {
    const int half = head_dim / 2;
    const double log_theta = std::log(static_cast<double>(rope_theta));
    for (int token = 0; token < tokens; ++token) {
        for (int i = 0; i < half; ++i) {
            const double inv_freq = std::exp(-log_theta * static_cast<double>(2 * i) / static_cast<double>(head_dim));
            const double angle = static_cast<double>(token) * inv_freq;
            const float c = static_cast<float>(std::cos(angle));
            const float s = static_cast<float>(std::sin(angle));
            for (int head = 0; head < n_heads; ++head) {
                float * row = states->data() +
                    static_cast<size_t>(token) * n_heads * head_dim +
                    static_cast<size_t>(head) * head_dim;
                const float a = row[i];
                const float b = row[half + i];
                row[i] = a * c - b * s;
                row[half + i] = b * c + a * s;
            }
        }
    }
}

static bool text_layer_forward_values_cpu(
    const QwenAsrGgufModel & model,
    const std::vector<float> & input_values,
    int tokens,
    int hidden,
    int layer,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int intermediate,
    float rope_theta,
    float rms_norm_eps,
    std::vector<float> * out_values,
    std::string * error) {
    if (!out_values) {
        set_error(error, "text_layer_forward_values_cpu: out_values is null");
        return false;
    }
    const int q_dim = n_heads * head_dim;
    const int kv_dim = n_kv_heads * head_dim;
    if (tokens <= 0 || hidden <= 0 || n_heads <= 0 || n_kv_heads <= 0 ||
        head_dim <= 0 || intermediate <= 0 || n_heads % n_kv_heads != 0 ||
        q_dim <= 0 || kv_dim <= 0 || input_values.size() != static_cast<size_t>(tokens) * hidden) {
        set_error(error, "invalid text layer configuration");
        return false;
    }

    TextLayerTensors tensors;
    if (!load_text_layer_tensors(model, layer, hidden, q_dim, kv_dim, head_dim, intermediate, &tensors, error)) {
        return false;
    }

    std::vector<float> normed(static_cast<size_t>(tokens) * hidden, 0.0f);
    const float * attn_norm_w = tensor_f32(tensors.attn_norm_w);
    for (int token = 0; token < tokens; ++token) {
        rms_norm_row(
            input_values.data() + static_cast<size_t>(token) * hidden,
            hidden,
            attn_norm_w,
            rms_norm_eps,
            normed.data() + static_cast<size_t>(token) * hidden);
    }

    std::vector<float> q;
    std::vector<float> k;
    std::vector<float> v;
    linear(normed, tokens, hidden, q_dim, tensors.q_w, &q);
    linear(normed, tokens, hidden, kv_dim, tensors.k_w, &k);
    linear(normed, tokens, hidden, kv_dim, tensors.v_w, &v);

    const float * q_norm_w = tensor_f32(tensors.q_norm_w);
    const float * k_norm_w = tensor_f32(tensors.k_norm_w);
    std::vector<float> temp(static_cast<size_t>(head_dim), 0.0f);
    for (int token = 0; token < tokens; ++token) {
        for (int head = 0; head < n_heads; ++head) {
            float * ptr = q.data() + static_cast<size_t>(token) * q_dim + static_cast<size_t>(head) * head_dim;
            rms_norm_row(ptr, head_dim, q_norm_w, rms_norm_eps, temp.data());
            std::memcpy(ptr, temp.data(), static_cast<size_t>(head_dim) * sizeof(float));
        }
        for (int head = 0; head < n_kv_heads; ++head) {
            float * ptr = k.data() + static_cast<size_t>(token) * kv_dim + static_cast<size_t>(head) * head_dim;
            rms_norm_row(ptr, head_dim, k_norm_w, rms_norm_eps, temp.data());
            std::memcpy(ptr, temp.data(), static_cast<size_t>(head_dim) * sizeof(float));
        }
    }

    apply_rope(&q, tokens, n_heads, head_dim, rope_theta);
    apply_rope(&k, tokens, n_kv_heads, head_dim, rope_theta);

    std::vector<float> attn(static_cast<size_t>(tokens) * q_dim, 0.0f);
    std::vector<float> scores(static_cast<size_t>(tokens), 0.0f);
    const int n_rep = n_heads / n_kv_heads;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    for (int token = 0; token < tokens; ++token) {
        for (int head = 0; head < n_heads; ++head) {
            const int kv_head = head / n_rep;
            const float * qrow = q.data() +
                static_cast<size_t>(token) * q_dim +
                static_cast<size_t>(head) * head_dim;

            float max_score = -std::numeric_limits<float>::infinity();
            for (int key_token = 0; key_token <= token; ++key_token) {
                const float * krow = k.data() +
                    static_cast<size_t>(key_token) * kv_dim +
                    static_cast<size_t>(kv_head) * head_dim;
                float score = 0.0f;
                for (int d = 0; d < head_dim; ++d) {
                    score += qrow[d] * krow[d];
                }
                score *= scale;
                scores[static_cast<size_t>(key_token)] = score;
                max_score = std::max(max_score, score);
            }

            float denom = 0.0f;
            for (int key_token = 0; key_token <= token; ++key_token) {
                const float value = std::exp(scores[static_cast<size_t>(key_token)] - max_score);
                scores[static_cast<size_t>(key_token)] = value;
                denom += value;
            }

            float * dst = attn.data() +
                static_cast<size_t>(token) * q_dim +
                static_cast<size_t>(head) * head_dim;
            for (int d = 0; d < head_dim; ++d) {
                float sum = 0.0f;
                for (int key_token = 0; key_token <= token; ++key_token) {
                    const float prob = scores[static_cast<size_t>(key_token)] / denom;
                    const float * vrow = v.data() +
                        static_cast<size_t>(key_token) * kv_dim +
                        static_cast<size_t>(kv_head) * head_dim;
                    sum += prob * vrow[d];
                }
                dst[d] = sum;
            }
        }
    }

    std::vector<float> projected;
    linear(attn, tokens, q_dim, hidden, tensors.out_w, &projected);

    std::vector<float> hidden_states(static_cast<size_t>(tokens) * hidden, 0.0f);
    for (size_t i = 0; i < hidden_states.size(); ++i) {
        hidden_states[i] = input_values[i] + projected[i];
    }

    const float * ffn_norm_w = tensor_f32(tensors.ffn_norm_w);
    for (int token = 0; token < tokens; ++token) {
        rms_norm_row(
            hidden_states.data() + static_cast<size_t>(token) * hidden,
            hidden,
            ffn_norm_w,
            rms_norm_eps,
            normed.data() + static_cast<size_t>(token) * hidden);
    }

    std::vector<float> gate;
    std::vector<float> up;
    linear(normed, tokens, hidden, intermediate, tensors.gate_w, &gate);
    linear(normed, tokens, hidden, intermediate, tensors.up_w, &up);
    for (size_t i = 0; i < gate.size(); ++i) {
        gate[i] = silu(gate[i]) * up[i];
    }

    std::vector<float> down;
    linear(gate, tokens, intermediate, hidden, tensors.down_w, &down);
    for (size_t i = 0; i < hidden_states.size(); ++i) {
        hidden_states[i] += down[i];
    }

    *out_values = std::move(hidden_states);
    return true;
}

bool qwenasr_text_layer_forward_cpu(
    const QwenAsrGgufModel & model,
    const QwenAsrDecoderInputOutput & input,
    int layer,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int intermediate,
    float rope_theta,
    float rms_norm_eps,
    QwenAsrTextLayerOutput * out,
    std::string * error) {
    if (!out) {
        set_error(error, "qwenasr_text_layer_forward_cpu: out is null");
        return false;
    }

    QwenAsrTextLayerOutput result;
    result.tokens = input.tokens;
    result.hidden = input.hidden;
    if (!text_layer_forward_values_cpu(
            model,
            input.values,
            input.tokens,
            input.hidden,
            layer,
            n_heads,
            n_kv_heads,
            head_dim,
            intermediate,
            rope_theta,
            rms_norm_eps,
            &result.values,
            error)) {
        return false;
    }
    *out = std::move(result);
    return true;
}

bool qwenasr_text_prefill_forward_cpu(
    const QwenAsrGgufModel & model,
    const QwenAsrDecoderInputOutput & input,
    int n_layers,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int intermediate,
    int vocab,
    float rope_theta,
    float rms_norm_eps,
    QwenAsrTextPrefillOutput * out,
    std::string * error) {
    if (!out) {
        set_error(error, "qwenasr_text_prefill_forward_cpu: out is null");
        return false;
    }
    if (input.tokens <= 0 || input.hidden <= 0 || n_layers <= 0 || vocab <= 0 ||
        input.values.size() != static_cast<size_t>(input.tokens) * input.hidden) {
        set_error(error, "invalid text prefill configuration");
        return false;
    }

    std::vector<float> states = input.values;
    std::vector<float> next;
    for (int layer = 0; layer < n_layers; ++layer) {
        if (!text_layer_forward_values_cpu(
                model,
                states,
                input.tokens,
                input.hidden,
                layer,
                n_heads,
                n_kv_heads,
                head_dim,
                intermediate,
                rope_theta,
                rms_norm_eps,
                &next,
                error)) {
            return false;
        }
        states = std::move(next);
        next.clear();
    }

    QwenAsrGgufTensorView norm_w;
    QwenAsrGgufTensorView output_w;
    if (!require_f32_tensor(model, "text.output_norm.weight", &norm_w, error) ||
        !require_f32_tensor(model, "text.output.weight", &output_w, error)) {
        return false;
    }
    if (!require_shape(norm_w, input.hidden, error) ||
        !require_shape(output_w, input.hidden, vocab, error)) {
        return false;
    }

    std::vector<float> normed(static_cast<size_t>(input.hidden), 0.0f);
    rms_norm_row(
        states.data() + static_cast<size_t>(input.tokens - 1) * input.hidden,
        input.hidden,
        tensor_f32(norm_w),
        rms_norm_eps,
        normed.data());

    const float * w = tensor_f32(output_w);
    std::vector<float> logits(static_cast<size_t>(vocab), 0.0f);
    for (int token = 0; token < vocab; ++token) {
        const float * ww = w + static_cast<size_t>(token) * input.hidden;
        float sum = 0.0f;
        for (int i = 0; i < input.hidden; ++i) {
            sum += normed[static_cast<size_t>(i)] * ww[i];
        }
        logits[static_cast<size_t>(token)] = sum;
    }

    QwenAsrTextPrefillOutput result;
    result.tokens = input.tokens;
    result.hidden = input.hidden;
    result.vocab = vocab;
    result.logits = std::move(logits);
    *out = std::move(result);
    return true;
}
