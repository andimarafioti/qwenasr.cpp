#include "audio-encoder.h"

#include "ggml.h"

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshadow"
#endif
#include "ggml-cpu.h"
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
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

struct AudioLayerTensors {
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
    int ffn = 0;
};

static bool load_audio_layer_tensors(
    const QwenAsrGgufModel & model,
    int layer,
    int hidden,
    AudioLayerTensors * tensors,
    std::string * error) {
    const std::string prefix = "audio.blk." + std::to_string(layer) + ".";
    if (!require_f32_tensor(model, prefix + "attn_norm.weight", &tensors->attn_norm_w, error) ||
        !require_f32_tensor(model, prefix + "attn_norm.bias", &tensors->attn_norm_b, error) ||
        !require_f32_tensor(model, prefix + "ffn_norm.weight", &tensors->ffn_norm_w, error) ||
        !require_f32_tensor(model, prefix + "ffn_norm.bias", &tensors->ffn_norm_b, error) ||
        !require_f32_tensor(model, prefix + "attn_q.weight", &tensors->q_w, error) ||
        !require_f32_tensor(model, prefix + "attn_q.bias", &tensors->q_b, error) ||
        !require_f32_tensor(model, prefix + "attn_k.weight", &tensors->k_w, error) ||
        !require_f32_tensor(model, prefix + "attn_k.bias", &tensors->k_b, error) ||
        !require_f32_tensor(model, prefix + "attn_v.weight", &tensors->v_w, error) ||
        !require_f32_tensor(model, prefix + "attn_v.bias", &tensors->v_b, error) ||
        !require_f32_tensor(model, prefix + "attn_output.weight", &tensors->out_w, error) ||
        !require_f32_tensor(model, prefix + "attn_output.bias", &tensors->out_b, error) ||
        !require_f32_tensor(model, prefix + "ffn_up.weight", &tensors->up_w, error) ||
        !require_f32_tensor(model, prefix + "ffn_up.bias", &tensors->up_b, error) ||
        !require_f32_tensor(model, prefix + "ffn_down.weight", &tensors->down_w, error) ||
        !require_f32_tensor(model, prefix + "ffn_down.bias", &tensors->down_b, error)) {
        return false;
    }

    tensors->ffn = static_cast<int>(tensors->up_w.ne.size() == 2 ? tensors->up_w.ne[1] : 0);
    if (!require_shape(tensors->attn_norm_w, hidden, error) ||
        !require_shape(tensors->attn_norm_b, hidden, error) ||
        !require_shape(tensors->ffn_norm_w, hidden, error) ||
        !require_shape(tensors->ffn_norm_b, hidden, error) ||
        !require_shape(tensors->q_w, hidden, hidden, error) ||
        !require_shape(tensors->q_b, hidden, error) ||
        !require_shape(tensors->k_w, hidden, hidden, error) ||
        !require_shape(tensors->k_b, hidden, error) ||
        !require_shape(tensors->v_w, hidden, hidden, error) ||
        !require_shape(tensors->v_b, hidden, error) ||
        !require_shape(tensors->out_w, hidden, hidden, error) ||
        !require_shape(tensors->out_b, hidden, error) ||
        !require_shape(tensors->up_w, hidden, tensors->ffn, error) ||
        !require_shape(tensors->up_b, tensors->ffn, error) ||
        !require_shape(tensors->down_w, tensors->ffn, hidden, error) ||
        !require_shape(tensors->down_b, hidden, error)) {
        return false;
    }
    return true;
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

    AudioLayerTensors tensors;
    if (!load_audio_layer_tensors(model, 0, hidden, &tensors, error)) {
        return false;
    }
    const int ffn = tensors.ffn;

    std::vector<float> normed;
    std::vector<float> q;
    std::vector<float> k;
    std::vector<float> v;
    std::vector<float> attn;
    std::vector<float> projected;
    layer_norm(prep.values, tokens, hidden, tensor_f32(tensors.attn_norm_w), tensor_f32(tensors.attn_norm_b), &normed);
    linear(normed, tokens, hidden, hidden, tensors.q_w, tensors.q_b, &q);
    linear(normed, tokens, hidden, hidden, tensors.k_w, tensors.k_b, &k);
    linear(normed, tokens, hidden, hidden, tensors.v_w, tensors.v_b, &v);
    audio_attention_full(q, k, v, tokens, hidden, n_heads, &attn);
    linear(attn, tokens, hidden, hidden, tensors.out_w, tensors.out_b, &projected);

    std::vector<float> hidden_states = prep.values;
    add_inplace(&hidden_states, projected);

    std::vector<float> ffn_normed;
    std::vector<float> ffn_up;
    std::vector<float> ffn_down;
    layer_norm(hidden_states, tokens, hidden, tensor_f32(tensors.ffn_norm_w), tensor_f32(tensors.ffn_norm_b), &ffn_normed);
    linear(ffn_normed, tokens, hidden, ffn, tensors.up_w, tensors.up_b, &ffn_up);
    for (float & value : ffn_up) {
        value = gelu(value);
    }
    linear(ffn_up, tokens, ffn, hidden, tensors.down_w, tensors.down_b, &ffn_down);
    add_inplace(&hidden_states, ffn_down);

    out->tokens = tokens;
    out->hidden = hidden;
    out->attention_segments = prep.attention_segments;
    out->values = std::move(hidden_states);
    return true;
}

static ggml_tensor * new_f32_tensor_from_view(
    ggml_context * ctx,
    const QwenAsrGgufTensorView & view) {
    ggml_tensor * tensor = ggml_new_tensor(
        ctx,
        GGML_TYPE_F32,
        static_cast<int>(view.ne.size()),
        view.ne.data());
    std::memcpy(tensor->data, view.data, ggml_nbytes(tensor));
    return tensor;
}

static ggml_tensor * linear_ggml(
    ggml_context * ctx,
    ggml_tensor * input,
    ggml_tensor * weight,
    ggml_tensor * bias) {
    ggml_tensor * output = ggml_mul_mat(ctx, weight, input);
    ggml_mul_mat_set_prec(output, GGML_PREC_F32);
    return ggml_add(ctx, output, ggml_repeat(ctx, bias, output));
}

static ggml_tensor * layer_norm_ggml(
    ggml_context * ctx,
    ggml_tensor * input,
    ggml_tensor * weight,
    ggml_tensor * bias) {
    ggml_tensor * output = ggml_norm(ctx, input, 1.0e-5f);
    output = ggml_mul(ctx, output, weight);
    return ggml_add(ctx, output, bias);
}

static bool audio_layer_forward_ggml_values(
    const QwenAsrGgufModel & model,
    const std::vector<float> & input_values,
    int tokens,
    int hidden,
    int n_threads,
    int n_heads,
    int layer,
    std::vector<float> * out_values,
    std::string * error) {
    if (n_threads <= 0) {
        n_threads = 1;
    }
    if (!out_values) {
        set_error(error, "audio_layer_forward_ggml_values: out_values is null");
        return false;
    }
    if (tokens <= 0 || hidden <= 0 || n_heads <= 0 || hidden % n_heads != 0 ||
        input_values.size() != static_cast<size_t>(tokens) * hidden) {
        set_error(error, "invalid audio layer geometry");
        return false;
    }

    AudioLayerTensors tensors;
    if (!load_audio_layer_tensors(model, layer, hidden, &tensors, error)) {
        return false;
    }

    const size_t ctx_size = 1024ull * 1024ull * 1024ull;
    std::vector<uint8_t> ctx_buffer(ctx_size);
    ggml_init_params params;
    params.mem_size = ctx_buffer.size();
    params.mem_buffer = ctx_buffer.data();
    params.no_alloc = false;
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        set_error(error, "failed to initialize GGML context for audio layer");
        return false;
    }

    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, tokens);
    std::memcpy(x->data, input_values.data(), input_values.size() * sizeof(float));

    ggml_tensor * attn_norm_w = new_f32_tensor_from_view(ctx, tensors.attn_norm_w);
    ggml_tensor * attn_norm_b = new_f32_tensor_from_view(ctx, tensors.attn_norm_b);
    ggml_tensor * ffn_norm_w = new_f32_tensor_from_view(ctx, tensors.ffn_norm_w);
    ggml_tensor * ffn_norm_b = new_f32_tensor_from_view(ctx, tensors.ffn_norm_b);
    ggml_tensor * q_w = new_f32_tensor_from_view(ctx, tensors.q_w);
    ggml_tensor * q_b = new_f32_tensor_from_view(ctx, tensors.q_b);
    ggml_tensor * k_w = new_f32_tensor_from_view(ctx, tensors.k_w);
    ggml_tensor * k_b = new_f32_tensor_from_view(ctx, tensors.k_b);
    ggml_tensor * v_w = new_f32_tensor_from_view(ctx, tensors.v_w);
    ggml_tensor * v_b = new_f32_tensor_from_view(ctx, tensors.v_b);
    ggml_tensor * out_w = new_f32_tensor_from_view(ctx, tensors.out_w);
    ggml_tensor * out_b = new_f32_tensor_from_view(ctx, tensors.out_b);
    ggml_tensor * up_w = new_f32_tensor_from_view(ctx, tensors.up_w);
    ggml_tensor * up_b = new_f32_tensor_from_view(ctx, tensors.up_b);
    ggml_tensor * down_w = new_f32_tensor_from_view(ctx, tensors.down_w);
    ggml_tensor * down_b = new_f32_tensor_from_view(ctx, tensors.down_b);

    const int head_dim = hidden / n_heads;
    ggml_tensor * normed = layer_norm_ggml(ctx, x, attn_norm_w, attn_norm_b);
    ggml_tensor * q = linear_ggml(ctx, normed, q_w, q_b);
    ggml_tensor * k = linear_ggml(ctx, normed, k_w, k_b);
    ggml_tensor * v = linear_ggml(ctx, normed, v_w, v_b);

    q = ggml_reshape_3d(ctx, q, head_dim, n_heads, tokens);
    k = ggml_reshape_3d(ctx, k, head_dim, n_heads, tokens);
    v = ggml_reshape_3d(ctx, v, head_dim, n_heads, tokens);

    ggml_tensor * q_p = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
    ggml_tensor * k_p = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
    ggml_tensor * v_p = ggml_cont(ctx, ggml_permute(ctx, v, 1, 2, 0, 3));

    ggml_tensor * scores = ggml_mul_mat(ctx, k_p, q_p);
    ggml_mul_mat_set_prec(scores, GGML_PREC_F32);
    scores = ggml_soft_max_ext(ctx, scores, nullptr, 1.0f / std::sqrt(static_cast<float>(head_dim)), 0.0f);

    ggml_tensor * attn = ggml_mul_mat(ctx, v_p, scores);
    ggml_mul_mat_set_prec(attn, GGML_PREC_F32);
    attn = ggml_cont(ctx, ggml_permute(ctx, attn, 0, 2, 1, 3));
    attn = ggml_reshape_2d(ctx, attn, hidden, tokens);

    ggml_tensor * projected = linear_ggml(ctx, attn, out_w, out_b);
    ggml_tensor * hidden_states = ggml_add(ctx, x, projected);

    ggml_tensor * ffn_normed = layer_norm_ggml(ctx, hidden_states, ffn_norm_w, ffn_norm_b);
    ggml_tensor * ffn_up = linear_ggml(ctx, ffn_normed, up_w, up_b);
    ffn_up = ggml_gelu_erf(ctx, ffn_up);
    ggml_tensor * ffn_down = linear_ggml(ctx, ffn_up, down_w, down_b);
    hidden_states = ggml_add(ctx, hidden_states, ffn_down);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, hidden_states);
    const ggml_status status = ggml_graph_compute_with_ctx(ctx, graph, n_threads);
    if (status != GGML_STATUS_SUCCESS) {
        ggml_free(ctx);
        set_error(error, "GGML audio layer graph compute failed");
        return false;
    }

    out_values->assign(static_cast<size_t>(tokens) * hidden, 0.0f);
    std::memcpy(out_values->data(), ggml_get_data_f32(hidden_states), out_values->size() * sizeof(float));

    ggml_free(ctx);
    return true;
}

bool qwenasr_audio_layer0_forward_ggml(
    const QwenAsrGgufModel & model,
    const QwenAsrFeatures & features,
    int n_threads,
    int n_heads,
    QwenAsrAudioLayerOutput * out,
    std::string * error) {
    if (!out) {
        set_error(error, "qwenasr_audio_layer0_forward_ggml: out is null");
        return false;
    }

    QwenAsrAudioPrepOutput prep;
    if (!qwenasr_audio_prep_forward_ggml(model, features, n_threads, &prep, error)) {
        return false;
    }

    QwenAsrAudioLayerOutput result;
    result.tokens = prep.tokens;
    result.hidden = prep.hidden;
    result.attention_segments = std::move(prep.attention_segments);
    if (!audio_layer_forward_ggml_values(
            model,
            prep.values,
            result.tokens,
            result.hidden,
            n_threads,
            n_heads,
            0,
            &result.values,
            error)) {
        return false;
    }

    *out = std::move(result);
    return true;
}

struct AudioProjectorTensors {
    QwenAsrGgufTensorView post_norm_w;
    QwenAsrGgufTensorView post_norm_b;
    QwenAsrGgufTensorView proj0_w;
    QwenAsrGgufTensorView proj0_b;
    QwenAsrGgufTensorView proj1_w;
    QwenAsrGgufTensorView proj1_b;
};

static bool load_audio_projector_tensors(
    const QwenAsrGgufModel & model,
    int hidden,
    int output_dim,
    AudioProjectorTensors * tensors,
    std::string * error) {
    if (!require_f32_tensor(model, "audio.post_norm.weight", &tensors->post_norm_w, error) ||
        !require_f32_tensor(model, "audio.post_norm.bias", &tensors->post_norm_b, error) ||
        !require_f32_tensor(model, "audio.proj.0.weight", &tensors->proj0_w, error) ||
        !require_f32_tensor(model, "audio.proj.0.bias", &tensors->proj0_b, error) ||
        !require_f32_tensor(model, "audio.proj.1.weight", &tensors->proj1_w, error) ||
        !require_f32_tensor(model, "audio.proj.1.bias", &tensors->proj1_b, error)) {
        return false;
    }
    if (!require_shape(tensors->post_norm_w, hidden, error) ||
        !require_shape(tensors->post_norm_b, hidden, error) ||
        !require_shape(tensors->proj0_w, hidden, hidden, error) ||
        !require_shape(tensors->proj0_b, hidden, error) ||
        !require_shape(tensors->proj1_w, hidden, output_dim, error) ||
        !require_shape(tensors->proj1_b, output_dim, error)) {
        return false;
    }
    return true;
}

static bool audio_projector_forward_ggml_values(
    const QwenAsrGgufModel & model,
    const std::vector<float> & input_values,
    int tokens,
    int hidden,
    int output_dim,
    int n_threads,
    std::vector<float> * out_values,
    std::string * error) {
    if (!out_values) {
        set_error(error, "audio_projector_forward_ggml_values: out_values is null");
        return false;
    }
    if (tokens <= 0 || hidden <= 0 || output_dim <= 0 ||
        input_values.size() != static_cast<size_t>(tokens) * hidden) {
        set_error(error, "invalid audio projector geometry");
        return false;
    }
    if (n_threads <= 0) {
        n_threads = 1;
    }

    AudioProjectorTensors tensors;
    if (!load_audio_projector_tensors(model, hidden, output_dim, &tensors, error)) {
        return false;
    }

    const size_t ctx_size = 512ull * 1024ull * 1024ull;
    std::vector<uint8_t> ctx_buffer(ctx_size);
    ggml_init_params params;
    params.mem_size = ctx_buffer.size();
    params.mem_buffer = ctx_buffer.data();
    params.no_alloc = false;
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        set_error(error, "failed to initialize GGML context for audio projector");
        return false;
    }

    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, tokens);
    std::memcpy(x->data, input_values.data(), input_values.size() * sizeof(float));

    ggml_tensor * post_norm_w = new_f32_tensor_from_view(ctx, tensors.post_norm_w);
    ggml_tensor * post_norm_b = new_f32_tensor_from_view(ctx, tensors.post_norm_b);
    ggml_tensor * proj0_w = new_f32_tensor_from_view(ctx, tensors.proj0_w);
    ggml_tensor * proj0_b = new_f32_tensor_from_view(ctx, tensors.proj0_b);
    ggml_tensor * proj1_w = new_f32_tensor_from_view(ctx, tensors.proj1_w);
    ggml_tensor * proj1_b = new_f32_tensor_from_view(ctx, tensors.proj1_b);

    ggml_tensor * y = layer_norm_ggml(ctx, x, post_norm_w, post_norm_b);
    y = linear_ggml(ctx, y, proj0_w, proj0_b);
    y = ggml_gelu_erf(ctx, y);
    y = linear_ggml(ctx, y, proj1_w, proj1_b);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, y);
    const ggml_status status = ggml_graph_compute_with_ctx(ctx, graph, n_threads);
    if (status != GGML_STATUS_SUCCESS) {
        ggml_free(ctx);
        set_error(error, "GGML audio projector graph compute failed");
        return false;
    }

    out_values->assign(static_cast<size_t>(tokens) * output_dim, 0.0f);
    std::memcpy(out_values->data(), ggml_get_data_f32(y), out_values->size() * sizeof(float));
    ggml_free(ctx);
    return true;
}

bool qwenasr_audio_encoder_forward_ggml(
    const QwenAsrGgufModel & model,
    const QwenAsrFeatures & features,
    int n_threads,
    int n_layers,
    int n_heads,
    int output_dim,
    QwenAsrAudioEncoderOutput * out,
    std::string * error) {
    if (!out) {
        set_error(error, "qwenasr_audio_encoder_forward_ggml: out is null");
        return false;
    }
    if (n_layers <= 0 || output_dim <= 0) {
        set_error(error, "invalid audio encoder configuration");
        return false;
    }

    QwenAsrAudioPrepOutput prep;
    if (!qwenasr_audio_prep_forward_ggml(model, features, n_threads, &prep, error)) {
        return false;
    }
    if (prep.tokens <= 0 || prep.hidden <= 0 || n_heads <= 0 || prep.hidden % n_heads != 0) {
        set_error(error, "invalid audio encoder geometry");
        return false;
    }

    std::vector<float> hidden_states = std::move(prep.values);
    std::vector<float> next;
    for (int layer = 0; layer < n_layers; ++layer) {
        if (!audio_layer_forward_ggml_values(
                model,
                hidden_states,
                prep.tokens,
                prep.hidden,
                n_threads,
                n_heads,
                layer,
                &next,
                error)) {
            return false;
        }
        hidden_states.swap(next);
    }

    std::vector<float> projected;
    if (!audio_projector_forward_ggml_values(
            model,
            hidden_states,
            prep.tokens,
            prep.hidden,
            output_dim,
            n_threads,
            &projected,
            error)) {
        return false;
    }

    QwenAsrAudioEncoderOutput result;
    result.tokens = prep.tokens;
    result.hidden = output_dim;
    result.attention_segments = std::move(prep.attention_segments);
    result.values = std::move(projected);
    *out = std::move(result);
    return true;
}
