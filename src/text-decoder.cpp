#include "text-decoder.h"

#include "ggml.h"

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshadow"
#endif
#include "ggml-backend.h"
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
    ggml_tensor * weight) {
    ggml_tensor * output = ggml_mul_mat(ctx, weight, input);
    ggml_mul_mat_set_prec(output, GGML_PREC_F32);
    return output;
}

static ggml_tensor * rms_norm_ggml(
    ggml_context * ctx,
    ggml_tensor * input,
    ggml_tensor * weight,
    float eps) {
    return ggml_mul(ctx, ggml_rms_norm(ctx, input, eps), weight);
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

struct TextLayerKvCache {
    int tokens = 0;
    int kv_dim = 0;
    std::vector<float> k; // [token, kv_head * head_dim], after RoPE
    std::vector<float> v; // [token, kv_head * head_dim]
};

struct TextDecoderTensorContext {
    std::vector<TextLayerTensors> layers;
    QwenAsrGgufTensorView embd_w;
    QwenAsrGgufTensorView output_norm_w;
    QwenAsrGgufTensorView output_w;
};

struct BackendPendingCopy {
    ggml_tensor * tensor = nullptr;
    const void * data = nullptr;
    size_t nbytes = 0;
};

struct BackendTextLayerTensors {
    ggml_tensor * attn_norm_w = nullptr;
    ggml_tensor * ffn_norm_w = nullptr;
    ggml_tensor * q_w = nullptr;
    ggml_tensor * k_w = nullptr;
    ggml_tensor * v_w = nullptr;
    ggml_tensor * out_w = nullptr;
    ggml_tensor * q_norm_w = nullptr;
    ggml_tensor * k_norm_w = nullptr;
    ggml_tensor * gate_w = nullptr;
    ggml_tensor * up_w = nullptr;
    ggml_tensor * down_w = nullptr;
};

struct QwenAsrTextLayerBackend {
    QwenAsrGgmlBackendSet backends;
    ggml_backend_sched_t sched = nullptr;
    ggml_context * weight_ctx = nullptr;
    ggml_backend_buffer_t weight_buffer = nullptr;
    BackendTextLayerTensors tensors;
    int n_threads = 1;
    int layer = 0;
    int hidden = 0;
    int n_heads = 0;
    int n_kv_heads = 0;
    int head_dim = 0;
    int intermediate = 0;
    float rope_theta = 0.0f;
    float rms_norm_eps = 0.0f;
};

struct QwenAsrTextDecoderBackend {
    QwenAsrGgmlBackendSet backends;
    ggml_backend_sched_t sched = nullptr;
    ggml_context * weight_ctx = nullptr;
    ggml_backend_buffer_t weight_buffer = nullptr;
    ggml_context * kv_ctx = nullptr;
    ggml_backend_buffer_t kv_buffer = nullptr;
    std::vector<BackendTextLayerTensors> layers;
    std::vector<ggml_tensor *> k_cache;
    std::vector<ggml_tensor *> v_cache;
    ggml_tensor * output_norm_w = nullptr;
    ggml_tensor * output_w = nullptr;
    int n_threads = 1;
    int n_layers = 0;
    int hidden = 0;
    int n_heads = 0;
    int n_kv_heads = 0;
    int head_dim = 0;
    int intermediate = 0;
    int vocab = 0;
    int max_seq_len = 0;
    int cur_len = 0;
    float rope_theta = 0.0f;
    float rms_norm_eps = 0.0f;
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

static ggml_tensor * new_backend_weight_from_view(
    ggml_context * ctx,
    const QwenAsrGgufTensorView & view,
    std::vector<BackendPendingCopy> * pending) {
    ggml_tensor * tensor = ggml_new_tensor(
        ctx,
        GGML_TYPE_F32,
        static_cast<int>(view.ne.size()),
        view.ne.data());
    ggml_set_name(tensor, view.name.c_str());
    if (pending) {
        pending->push_back({ tensor, view.data, ggml_nbytes(tensor) });
    }
    return tensor;
}

static BackendTextLayerTensors backend_text_layer_tensors_from_views(
    ggml_context * ctx,
    const TextLayerTensors & views,
    std::vector<BackendPendingCopy> * pending) {
    BackendTextLayerTensors tensors;
    tensors.attn_norm_w = new_backend_weight_from_view(ctx, views.attn_norm_w, pending);
    tensors.ffn_norm_w = new_backend_weight_from_view(ctx, views.ffn_norm_w, pending);
    tensors.q_w = new_backend_weight_from_view(ctx, views.q_w, pending);
    tensors.k_w = new_backend_weight_from_view(ctx, views.k_w, pending);
    tensors.v_w = new_backend_weight_from_view(ctx, views.v_w, pending);
    tensors.out_w = new_backend_weight_from_view(ctx, views.out_w, pending);
    tensors.q_norm_w = new_backend_weight_from_view(ctx, views.q_norm_w, pending);
    tensors.k_norm_w = new_backend_weight_from_view(ctx, views.k_norm_w, pending);
    tensors.gate_w = new_backend_weight_from_view(ctx, views.gate_w, pending);
    tensors.up_w = new_backend_weight_from_view(ctx, views.up_w, pending);
    tensors.down_w = new_backend_weight_from_view(ctx, views.down_w, pending);
    return tensors;
}

static ggml_tensor * backend_text_layer_forward_ggml(
    ggml_context * ctx,
    const BackendTextLayerTensors & tensors,
    ggml_tensor * x,
    ggml_tensor * positions,
    int hidden,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int intermediate,
    float rope_theta,
    float rms_norm_eps) {
    const int tokens = static_cast<int>(x->ne[1]);
    const int q_dim = n_heads * head_dim;
    GGML_UNUSED(hidden);
    GGML_UNUSED(intermediate);

    ggml_tensor * normed = rms_norm_ggml(ctx, x, tensors.attn_norm_w, rms_norm_eps);
    ggml_tensor * q = linear_ggml(ctx, normed, tensors.q_w);
    ggml_tensor * k = linear_ggml(ctx, normed, tensors.k_w);
    ggml_tensor * v = linear_ggml(ctx, normed, tensors.v_w);

    q = ggml_reshape_3d(ctx, q, head_dim, n_heads, tokens);
    k = ggml_reshape_3d(ctx, k, head_dim, n_kv_heads, tokens);
    v = ggml_reshape_3d(ctx, v, head_dim, n_kv_heads, tokens);

    q = rms_norm_ggml(ctx, q, tensors.q_norm_w, rms_norm_eps);
    k = rms_norm_ggml(ctx, k, tensors.k_norm_w, rms_norm_eps);
    q = ggml_rope_ext(
        ctx,
        q,
        positions,
        nullptr,
        head_dim,
        GGML_ROPE_TYPE_NEOX,
        0,
        rope_theta,
        1.0f,
        0.0f,
        1.0f,
        0.0f,
        0.0f);
    k = ggml_rope_ext(
        ctx,
        k,
        positions,
        nullptr,
        head_dim,
        GGML_ROPE_TYPE_NEOX,
        0,
        rope_theta,
        1.0f,
        0.0f,
        1.0f,
        0.0f,
        0.0f);

    ggml_tensor * q_p = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
    ggml_tensor * k_p = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
    ggml_tensor * v_p = ggml_cont(ctx, ggml_permute(ctx, v, 1, 2, 0, 3));

    ggml_tensor * scores = ggml_mul_mat(ctx, k_p, q_p);
    ggml_mul_mat_set_prec(scores, GGML_PREC_F32);
    scores = ggml_diag_mask_inf(ctx, scores, 0);
    scores = ggml_soft_max_ext(ctx, scores, nullptr, 1.0f / std::sqrt(static_cast<float>(head_dim)), 0.0f);

    ggml_tensor * attn = ggml_mul_mat(ctx, v_p, scores);
    ggml_mul_mat_set_prec(attn, GGML_PREC_F32);
    attn = ggml_cont(ctx, ggml_permute(ctx, attn, 0, 2, 1, 3));
    attn = ggml_reshape_2d(ctx, attn, q_dim, tokens);

    ggml_tensor * projected = linear_ggml(ctx, attn, tensors.out_w);
    ggml_tensor * hidden_states = ggml_add(ctx, x, projected);

    ggml_tensor * ffn_normed = rms_norm_ggml(ctx, hidden_states, tensors.ffn_norm_w, rms_norm_eps);
    ggml_tensor * gate = linear_ggml(ctx, ffn_normed, tensors.gate_w);
    ggml_tensor * up = linear_ggml(ctx, ffn_normed, tensors.up_w);
    ggml_tensor * activated = ggml_mul(ctx, ggml_silu(ctx, gate), up);
    ggml_tensor * down = linear_ggml(ctx, activated, tensors.down_w);
    return ggml_add(ctx, hidden_states, down);
}

static ggml_tensor * backend_text_layer_forward_cached_ggml(
    ggml_context * ctx,
    const BackendTextLayerTensors & tensors,
    ggml_tensor * x,
    ggml_tensor * positions,
    ggml_tensor * mask,
    ggml_tensor * k_cache,
    ggml_tensor * v_cache,
    int n_past,
    int hidden,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int intermediate,
    float rope_theta,
    float rms_norm_eps,
    ggml_cgraph * graph) {
    const int tokens = static_cast<int>(x->ne[1]);
    const int full_tokens = n_past + tokens;
    const int q_dim = n_heads * head_dim;
    GGML_UNUSED(hidden);
    GGML_UNUSED(intermediate);

    ggml_tensor * normed = rms_norm_ggml(ctx, x, tensors.attn_norm_w, rms_norm_eps);
    ggml_tensor * q = linear_ggml(ctx, normed, tensors.q_w);
    ggml_tensor * k = linear_ggml(ctx, normed, tensors.k_w);
    ggml_tensor * v = linear_ggml(ctx, normed, tensors.v_w);

    q = ggml_reshape_3d(ctx, q, head_dim, n_heads, tokens);
    k = ggml_reshape_3d(ctx, k, head_dim, n_kv_heads, tokens);
    v = ggml_reshape_3d(ctx, v, head_dim, n_kv_heads, tokens);

    q = rms_norm_ggml(ctx, q, tensors.q_norm_w, rms_norm_eps);
    k = rms_norm_ggml(ctx, k, tensors.k_norm_w, rms_norm_eps);
    q = ggml_rope_ext(
        ctx,
        q,
        positions,
        nullptr,
        head_dim,
        GGML_ROPE_TYPE_NEOX,
        0,
        rope_theta,
        1.0f,
        0.0f,
        1.0f,
        0.0f,
        0.0f);
    k = ggml_rope_ext(
        ctx,
        k,
        positions,
        nullptr,
        head_dim,
        GGML_ROPE_TYPE_NEOX,
        0,
        rope_theta,
        1.0f,
        0.0f,
        1.0f,
        0.0f,
        0.0f);

    ggml_tensor * k_perm = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
    ggml_tensor * v_perm = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));
    const size_t k_offset = static_cast<size_t>(n_past) * k_cache->nb[1];
    const size_t v_offset = static_cast<size_t>(n_past) * v_cache->nb[1];
    ggml_tensor * k_dst = ggml_view_3d(
        ctx,
        k_cache,
        head_dim,
        tokens,
        n_kv_heads,
        k_cache->nb[1],
        k_cache->nb[2],
        k_offset);
    ggml_tensor * v_dst = ggml_view_3d(
        ctx,
        v_cache,
        head_dim,
        tokens,
        n_kv_heads,
        v_cache->nb[1],
        v_cache->nb[2],
        v_offset);
    ggml_build_forward_expand(graph, ggml_cpy(ctx, k_perm, k_dst));
    ggml_build_forward_expand(graph, ggml_cpy(ctx, v_perm, v_dst));

    ggml_tensor * k_full = ggml_view_3d(
        ctx,
        k_cache,
        head_dim,
        full_tokens,
        n_kv_heads,
        k_cache->nb[1],
        k_cache->nb[2],
        0);
    ggml_tensor * v_full = ggml_view_3d(
        ctx,
        v_cache,
        head_dim,
        full_tokens,
        n_kv_heads,
        v_cache->nb[1],
        v_cache->nb[2],
        0);

    ggml_tensor * q_p = ggml_permute(ctx, q, 0, 2, 1, 3);
    ggml_tensor * scores = ggml_mul_mat(ctx, k_full, q_p);
    ggml_mul_mat_set_prec(scores, GGML_PREC_F32);
    scores = ggml_soft_max_ext(ctx, scores, mask, 1.0f / std::sqrt(static_cast<float>(head_dim)), 0.0f);

    ggml_tensor * v_t = ggml_cont(ctx, ggml_transpose(ctx, v_full));
    ggml_tensor * attn = ggml_mul_mat(ctx, v_t, scores);
    ggml_mul_mat_set_prec(attn, GGML_PREC_F32);
    attn = ggml_cont(ctx, ggml_permute(ctx, attn, 0, 2, 1, 3));
    attn = ggml_reshape_2d(ctx, attn, q_dim, tokens);

    ggml_tensor * projected = linear_ggml(ctx, attn, tensors.out_w);
    ggml_tensor * hidden_states = ggml_add(ctx, x, projected);

    ggml_tensor * ffn_normed = rms_norm_ggml(ctx, hidden_states, tensors.ffn_norm_w, rms_norm_eps);
    ggml_tensor * gate = linear_ggml(ctx, ffn_normed, tensors.gate_w);
    ggml_tensor * up = linear_ggml(ctx, ffn_normed, tensors.up_w);
    ggml_tensor * activated = ggml_mul(ctx, ggml_silu(ctx, gate), up);
    ggml_tensor * down = linear_ggml(ctx, activated, tensors.down_w);
    return ggml_add(ctx, hidden_states, down);
}

static ggml_tensor * backend_text_logits_ggml(
    ggml_context * ctx,
    ggml_tensor * states,
    ggml_tensor * output_norm_w,
    ggml_tensor * output_w,
    int hidden,
    int tokens,
    float rms_norm_eps) {
    ggml_tensor * last = ggml_view_2d(
        ctx,
        states,
        hidden,
        1,
        states->nb[1],
        static_cast<size_t>(tokens - 1) * states->nb[1]);
    ggml_tensor * normed = rms_norm_ggml(ctx, last, output_norm_w, rms_norm_eps);
    return linear_ggml(ctx, normed, output_w);
}

static bool load_text_decoder_tensor_context(
    const QwenAsrGgufModel & model,
    int n_layers,
    int hidden,
    int q_dim,
    int kv_dim,
    int head_dim,
    int intermediate,
    int vocab,
    TextDecoderTensorContext * tensors,
    std::string * error) {
    if (!tensors || n_layers <= 0) {
        set_error(error, "invalid text decoder tensor context configuration");
        return false;
    }

    TextDecoderTensorContext result;
    result.layers.resize(static_cast<size_t>(n_layers));
    for (int layer = 0; layer < n_layers; ++layer) {
        if (!load_text_layer_tensors(
                model,
                layer,
                hidden,
                q_dim,
                kv_dim,
                head_dim,
                intermediate,
                &result.layers[static_cast<size_t>(layer)],
                error)) {
            return false;
        }
    }
    if (!require_f32_tensor(model, "text.token_embd.weight", &result.embd_w, error) ||
        !require_f32_tensor(model, "text.output_norm.weight", &result.output_norm_w, error) ||
        !require_f32_tensor(model, "text.output.weight", &result.output_w, error)) {
        return false;
    }
    if (!require_shape(result.embd_w, hidden, vocab, error) ||
        !require_shape(result.output_norm_w, hidden, error) ||
        !require_shape(result.output_w, hidden, vocab, error)) {
        return false;
    }

    *tensors = std::move(result);
    return true;
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

static void linear_row(
    const float * input,
    int in_dim,
    int out_dim,
    const QwenAsrGgufTensorView & weight,
    std::vector<float> * output) {
    const float * w = tensor_f32(weight);
    output->assign(static_cast<size_t>(out_dim), 0.0f);
    for (int out = 0; out < out_dim; ++out) {
        float sum = 0.0f;
        const float * ww = w + static_cast<size_t>(out) * in_dim;
        for (int in = 0; in < in_dim; ++in) {
            sum += input[in] * ww[in];
        }
        (*output)[static_cast<size_t>(out)] = sum;
    }
}

static float silu(float x) {
    return x / (1.0f + std::exp(-x));
}

static void apply_rope_position(
    float * rows,
    int position,
    int n_heads,
    int head_dim,
    float rope_theta) {
    const int half = head_dim / 2;
    const double log_theta = std::log(static_cast<double>(rope_theta));
    for (int i = 0; i < half; ++i) {
        const double inv_freq = std::exp(-log_theta * static_cast<double>(2 * i) / static_cast<double>(head_dim));
        const double angle = static_cast<double>(position) * inv_freq;
        const float c = static_cast<float>(std::cos(angle));
        const float s = static_cast<float>(std::sin(angle));
        for (int head = 0; head < n_heads; ++head) {
            float * row = rows + static_cast<size_t>(head) * head_dim;
            const float a = row[i];
            const float b = row[half + i];
            row[i] = a * c - b * s;
            row[half + i] = b * c + a * s;
        }
    }
}

static void apply_rope(
    std::vector<float> * states,
    int tokens,
    int n_heads,
    int head_dim,
    float rope_theta) {
    for (int token = 0; token < tokens; ++token) {
        apply_rope_position(
            states->data() + static_cast<size_t>(token) * n_heads * head_dim,
            token,
            n_heads,
            head_dim,
            rope_theta);
    }
}

static bool text_layer_forward_values_cpu(
    const QwenAsrGgufModel & model,
    const std::vector<float> & input_values,
    int tokens,
    int hidden,
    int layer,
    const TextLayerTensors * loaded_tensors,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int intermediate,
    float rope_theta,
    float rms_norm_eps,
    std::vector<float> * out_values,
    TextLayerKvCache * cache,
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

    TextLayerTensors local_tensors;
    const TextLayerTensors * tensors_ptr = loaded_tensors;
    if (!tensors_ptr) {
        if (!load_text_layer_tensors(model, layer, hidden, q_dim, kv_dim, head_dim, intermediate, &local_tensors, error)) {
            return false;
        }
        tensors_ptr = &local_tensors;
    }
    const TextLayerTensors & tensors = *tensors_ptr;

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

    if (cache) {
        cache->tokens = tokens;
        cache->kv_dim = kv_dim;
        cache->k = std::move(k);
        cache->v = std::move(v);
    }

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

static bool text_layer_forward_values_ggml(
    const QwenAsrGgufModel & model,
    const std::vector<float> & input_values,
    int tokens,
    int hidden,
    int layer,
    int n_threads,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int intermediate,
    float rope_theta,
    float rms_norm_eps,
    std::vector<float> * out_values,
    std::string * error) {
    if (n_threads <= 0) {
        n_threads = 1;
    }
    if (!out_values) {
        set_error(error, "text_layer_forward_values_ggml: out_values is null");
        return false;
    }
    const int q_dim = n_heads * head_dim;
    const int kv_dim = n_kv_heads * head_dim;
    if (tokens <= 0 || hidden <= 0 || n_heads <= 0 || n_kv_heads <= 0 ||
        head_dim <= 0 || intermediate <= 0 || n_heads % n_kv_heads != 0 ||
        q_dim <= 0 || kv_dim <= 0 || input_values.size() != static_cast<size_t>(tokens) * hidden) {
        set_error(error, "invalid GGML text layer configuration");
        return false;
    }

    TextLayerTensors tensors;
    if (!load_text_layer_tensors(model, layer, hidden, q_dim, kv_dim, head_dim, intermediate, &tensors, error)) {
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
        set_error(error, "failed to initialize GGML context for text layer");
        return false;
    }

    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, tokens);
    std::memcpy(x->data, input_values.data(), input_values.size() * sizeof(float));

    ggml_tensor * positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, tokens);
    int32_t * position_data = static_cast<int32_t *>(positions->data);
    for (int token = 0; token < tokens; ++token) {
        position_data[token] = token;
    }

    BackendTextLayerTensors graph_tensors;
    graph_tensors.attn_norm_w = new_f32_tensor_from_view(ctx, tensors.attn_norm_w);
    graph_tensors.ffn_norm_w = new_f32_tensor_from_view(ctx, tensors.ffn_norm_w);
    graph_tensors.q_w = new_f32_tensor_from_view(ctx, tensors.q_w);
    graph_tensors.k_w = new_f32_tensor_from_view(ctx, tensors.k_w);
    graph_tensors.v_w = new_f32_tensor_from_view(ctx, tensors.v_w);
    graph_tensors.out_w = new_f32_tensor_from_view(ctx, tensors.out_w);
    graph_tensors.q_norm_w = new_f32_tensor_from_view(ctx, tensors.q_norm_w);
    graph_tensors.k_norm_w = new_f32_tensor_from_view(ctx, tensors.k_norm_w);
    graph_tensors.gate_w = new_f32_tensor_from_view(ctx, tensors.gate_w);
    graph_tensors.up_w = new_f32_tensor_from_view(ctx, tensors.up_w);
    graph_tensors.down_w = new_f32_tensor_from_view(ctx, tensors.down_w);

    ggml_tensor * hidden_states = backend_text_layer_forward_ggml(
        ctx,
        graph_tensors,
        x,
        positions,
        hidden,
        n_heads,
        n_kv_heads,
        head_dim,
        intermediate,
        rope_theta,
        rms_norm_eps);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, hidden_states);
    const ggml_status status = ggml_graph_compute_with_ctx(ctx, graph, n_threads);
    if (status != GGML_STATUS_SUCCESS) {
        ggml_free(ctx);
        set_error(error, "GGML text layer graph compute failed");
        return false;
    }

    out_values->assign(static_cast<size_t>(tokens) * hidden, 0.0f);
    std::memcpy(out_values->data(), ggml_get_data_f32(hidden_states), out_values->size() * sizeof(float));

    ggml_free(ctx);
    return true;
}

static bool compute_logits_for_state_cpu(
    const QwenAsrGgufModel & model,
    const float * state,
    int hidden,
    int vocab,
    float rms_norm_eps,
    const QwenAsrGgufTensorView * loaded_norm_w,
    const QwenAsrGgufTensorView * loaded_output_w,
    std::vector<float> * logits,
    std::string * error) {
    if (!state || !logits || hidden <= 0 || vocab <= 0) {
        set_error(error, "invalid text logits configuration");
        return false;
    }

    QwenAsrGgufTensorView local_norm_w;
    QwenAsrGgufTensorView local_output_w;
    const QwenAsrGgufTensorView * norm_w = loaded_norm_w;
    const QwenAsrGgufTensorView * output_w = loaded_output_w;
    if (!norm_w || !output_w) {
        if (!require_f32_tensor(model, "text.output_norm.weight", &local_norm_w, error) ||
            !require_f32_tensor(model, "text.output.weight", &local_output_w, error)) {
            return false;
        }
        norm_w = &local_norm_w;
        output_w = &local_output_w;
    }
    if (!require_shape(*norm_w, hidden, error) ||
        !require_shape(*output_w, hidden, vocab, error)) {
        return false;
    }

    std::vector<float> normed(static_cast<size_t>(hidden), 0.0f);
    rms_norm_row(state, hidden, tensor_f32(*norm_w), rms_norm_eps, normed.data());

    const float * w = tensor_f32(*output_w);
    logits->assign(static_cast<size_t>(vocab), 0.0f);
    for (int token = 0; token < vocab; ++token) {
        const float * ww = w + static_cast<size_t>(token) * hidden;
        float sum = 0.0f;
        for (int i = 0; i < hidden; ++i) {
            sum += normed[static_cast<size_t>(i)] * ww[i];
        }
        (*logits)[static_cast<size_t>(token)] = sum;
    }
    return true;
}

static bool text_layer_decode_cached_cpu(
    const QwenAsrGgufModel & model,
    const std::vector<float> & input_state,
    int layer,
    const TextLayerTensors * loaded_tensors,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int intermediate,
    float rope_theta,
    float rms_norm_eps,
    TextLayerKvCache * cache,
    std::vector<float> * out_state,
    std::string * error) {
    if (!cache || !out_state) {
        set_error(error, "text_layer_decode_cached_cpu: cache or out_state is null");
        return false;
    }
    const int hidden = static_cast<int>(input_state.size());
    const int q_dim = n_heads * head_dim;
    const int kv_dim = n_kv_heads * head_dim;
    if (hidden <= 0 || n_heads <= 0 || n_kv_heads <= 0 || head_dim <= 0 ||
        intermediate <= 0 || n_heads % n_kv_heads != 0 || q_dim <= 0 || kv_dim <= 0 ||
        cache->tokens <= 0 || cache->kv_dim != kv_dim ||
        cache->k.size() != static_cast<size_t>(cache->tokens) * kv_dim ||
        cache->v.size() != static_cast<size_t>(cache->tokens) * kv_dim) {
        set_error(error, "invalid cached text layer configuration");
        return false;
    }

    TextLayerTensors local_tensors;
    const TextLayerTensors * tensors_ptr = loaded_tensors;
    if (!tensors_ptr) {
        if (!load_text_layer_tensors(model, layer, hidden, q_dim, kv_dim, head_dim, intermediate, &local_tensors, error)) {
            return false;
        }
        tensors_ptr = &local_tensors;
    }
    const TextLayerTensors & tensors = *tensors_ptr;

    std::vector<float> normed(static_cast<size_t>(hidden), 0.0f);
    rms_norm_row(input_state.data(), hidden, tensor_f32(tensors.attn_norm_w), rms_norm_eps, normed.data());

    std::vector<float> q;
    std::vector<float> k;
    std::vector<float> v;
    linear_row(normed.data(), hidden, q_dim, tensors.q_w, &q);
    linear_row(normed.data(), hidden, kv_dim, tensors.k_w, &k);
    linear_row(normed.data(), hidden, kv_dim, tensors.v_w, &v);

    const float * q_norm_w = tensor_f32(tensors.q_norm_w);
    const float * k_norm_w = tensor_f32(tensors.k_norm_w);
    std::vector<float> temp(static_cast<size_t>(head_dim), 0.0f);
    for (int head = 0; head < n_heads; ++head) {
        float * ptr = q.data() + static_cast<size_t>(head) * head_dim;
        rms_norm_row(ptr, head_dim, q_norm_w, rms_norm_eps, temp.data());
        std::memcpy(ptr, temp.data(), static_cast<size_t>(head_dim) * sizeof(float));
    }
    for (int head = 0; head < n_kv_heads; ++head) {
        float * ptr = k.data() + static_cast<size_t>(head) * head_dim;
        rms_norm_row(ptr, head_dim, k_norm_w, rms_norm_eps, temp.data());
        std::memcpy(ptr, temp.data(), static_cast<size_t>(head_dim) * sizeof(float));
    }

    const int position = cache->tokens;
    apply_rope_position(q.data(), position, n_heads, head_dim, rope_theta);
    apply_rope_position(k.data(), position, n_kv_heads, head_dim, rope_theta);

    cache->k.insert(cache->k.end(), k.begin(), k.end());
    cache->v.insert(cache->v.end(), v.begin(), v.end());
    ++cache->tokens;

    std::vector<float> attn(static_cast<size_t>(q_dim), 0.0f);
    std::vector<float> scores(static_cast<size_t>(cache->tokens), 0.0f);
    const int n_rep = n_heads / n_kv_heads;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    for (int head = 0; head < n_heads; ++head) {
        const int kv_head = head / n_rep;
        const float * qrow = q.data() + static_cast<size_t>(head) * head_dim;

        float max_score = -std::numeric_limits<float>::infinity();
        for (int key_token = 0; key_token < cache->tokens; ++key_token) {
            const float * krow = cache->k.data() +
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
        for (int key_token = 0; key_token < cache->tokens; ++key_token) {
            const float value = std::exp(scores[static_cast<size_t>(key_token)] - max_score);
            scores[static_cast<size_t>(key_token)] = value;
            denom += value;
        }

        float * dst = attn.data() + static_cast<size_t>(head) * head_dim;
        for (int d = 0; d < head_dim; ++d) {
            float sum = 0.0f;
            for (int key_token = 0; key_token < cache->tokens; ++key_token) {
                const float prob = scores[static_cast<size_t>(key_token)] / denom;
                const float * vrow = cache->v.data() +
                    static_cast<size_t>(key_token) * kv_dim +
                    static_cast<size_t>(kv_head) * head_dim;
                sum += prob * vrow[d];
            }
            dst[d] = sum;
        }
    }

    std::vector<float> projected;
    linear_row(attn.data(), q_dim, hidden, tensors.out_w, &projected);

    std::vector<float> hidden_state(static_cast<size_t>(hidden), 0.0f);
    for (int i = 0; i < hidden; ++i) {
        hidden_state[static_cast<size_t>(i)] = input_state[static_cast<size_t>(i)] + projected[static_cast<size_t>(i)];
    }

    rms_norm_row(hidden_state.data(), hidden, tensor_f32(tensors.ffn_norm_w), rms_norm_eps, normed.data());
    std::vector<float> gate;
    std::vector<float> up;
    linear_row(normed.data(), hidden, intermediate, tensors.gate_w, &gate);
    linear_row(normed.data(), hidden, intermediate, tensors.up_w, &up);
    for (size_t i = 0; i < gate.size(); ++i) {
        gate[i] = silu(gate[i]) * up[i];
    }

    std::vector<float> down;
    linear_row(gate.data(), intermediate, hidden, tensors.down_w, &down);
    for (int i = 0; i < hidden; ++i) {
        hidden_state[static_cast<size_t>(i)] += down[static_cast<size_t>(i)];
    }

    *out_state = std::move(hidden_state);
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
            nullptr,
            n_heads,
            n_kv_heads,
            head_dim,
            intermediate,
            rope_theta,
            rms_norm_eps,
            &result.values,
            nullptr,
            error)) {
        return false;
    }
    *out = std::move(result);
    return true;
}

bool qwenasr_text_layer_forward_ggml(
    const QwenAsrGgufModel & model,
    const QwenAsrDecoderInputOutput & input,
    int layer,
    int n_threads,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int intermediate,
    float rope_theta,
    float rms_norm_eps,
    QwenAsrTextLayerOutput * out,
    std::string * error) {
    if (!out) {
        set_error(error, "qwenasr_text_layer_forward_ggml: out is null");
        return false;
    }

    QwenAsrTextLayerOutput result;
    result.tokens = input.tokens;
    result.hidden = input.hidden;
    if (!text_layer_forward_values_ggml(
            model,
            input.values,
            input.tokens,
            input.hidden,
            layer,
            n_threads,
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

void qwenasr_text_layer_backend_free(QwenAsrTextLayerBackend * backend) {
    if (!backend) {
        return;
    }
    if (backend->sched) {
        ggml_backend_sched_free(backend->sched);
        backend->sched = nullptr;
    }
    if (backend->weight_buffer) {
        ggml_backend_buffer_free(backend->weight_buffer);
        backend->weight_buffer = nullptr;
    }
    if (backend->weight_ctx) {
        ggml_free(backend->weight_ctx);
        backend->weight_ctx = nullptr;
    }
    qwenasr_ggml_backend_set_free(&backend->backends);
    delete backend;
}

bool qwenasr_text_layer_backend_init(
    const QwenAsrGgufModel & model,
    int layer,
    int n_threads,
    int hidden,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int intermediate,
    float rope_theta,
    float rms_norm_eps,
    QwenAsrTextLayerBackend ** out,
    std::string * error,
    QwenAsrGgmlDevice device) {
    if (!out) {
        set_error(error, "qwenasr_text_layer_backend_init: out is null");
        return false;
    }
    *out = nullptr;
    const int q_dim = n_heads * head_dim;
    const int kv_dim = n_kv_heads * head_dim;
    if (layer < 0 || hidden <= 0 || n_heads <= 0 || n_kv_heads <= 0 || head_dim <= 0 ||
        intermediate <= 0 || n_heads % n_kv_heads != 0 || q_dim <= 0 || kv_dim <= 0) {
        set_error(error, "invalid text layer backend configuration");
        return false;
    }
    if (n_threads <= 0) {
        n_threads = 1;
    }

    QwenAsrTextLayerBackend * runtime = new QwenAsrTextLayerBackend();
    runtime->n_threads = n_threads;
    runtime->layer = layer;
    runtime->hidden = hidden;
    runtime->n_heads = n_heads;
    runtime->n_kv_heads = n_kv_heads;
    runtime->head_dim = head_dim;
    runtime->intermediate = intermediate;
    runtime->rope_theta = rope_theta;
    runtime->rms_norm_eps = rms_norm_eps;

    if (!qwenasr_ggml_backend_set_init(device, n_threads, &runtime->backends, error)) {
        qwenasr_text_layer_backend_free(runtime);
        return false;
    }

    constexpr size_t max_nodes = 4096;
    ggml_backend_t backends[2] = {};
    qwenasr_ggml_backend_set_array(runtime->backends, backends);
    runtime->sched = ggml_backend_sched_new(
        backends,
        nullptr,
        qwenasr_ggml_backend_set_count(runtime->backends),
        max_nodes,
        false,
        true);
    if (!runtime->sched) {
        qwenasr_text_layer_backend_free(runtime);
        set_error(error, "failed to initialize GGML text layer backend scheduler");
        return false;
    }

    constexpr int n_weight_tensors = 11;
    const size_t weight_ctx_size = static_cast<size_t>(n_weight_tensors) * ggml_tensor_overhead() + 1024ull * 1024ull;
    ggml_init_params params;
    params.mem_size = weight_ctx_size;
    params.mem_buffer = nullptr;
    params.no_alloc = true;
    runtime->weight_ctx = ggml_init(params);
    if (!runtime->weight_ctx) {
        qwenasr_text_layer_backend_free(runtime);
        set_error(error, "failed to initialize text layer weight context");
        return false;
    }

    TextLayerTensors views;
    if (!load_text_layer_tensors(model, layer, hidden, q_dim, kv_dim, head_dim, intermediate, &views, error)) {
        qwenasr_text_layer_backend_free(runtime);
        return false;
    }

    std::vector<BackendPendingCopy> pending;
    pending.reserve(static_cast<size_t>(n_weight_tensors));
    runtime->tensors = backend_text_layer_tensors_from_views(runtime->weight_ctx, views, &pending);

    runtime->weight_buffer = ggml_backend_alloc_ctx_tensors(runtime->weight_ctx, runtime->backends.primary);
    if (!runtime->weight_buffer) {
        qwenasr_text_layer_backend_free(runtime);
        set_error(error, "failed to allocate text layer backend weight buffer");
        return false;
    }
    ggml_backend_buffer_set_usage(runtime->weight_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    for (const BackendPendingCopy & copy : pending) {
        ggml_backend_tensor_set(copy.tensor, copy.data, 0, copy.nbytes);
    }

    *out = runtime;
    return true;
}

bool qwenasr_text_layer_backend_forward(
    QwenAsrTextLayerBackend * backend,
    const QwenAsrDecoderInputOutput & input,
    QwenAsrTextLayerOutput * out,
    std::string * error) {
    if (!backend || !out) {
        set_error(error, "qwenasr_text_layer_backend_forward: backend or out is null");
        return false;
    }
    if (input.tokens <= 0 || input.hidden != backend->hidden ||
        input.values.size() != static_cast<size_t>(input.tokens) * input.hidden) {
        set_error(error, "invalid decoder input for text layer backend");
        return false;
    }

    constexpr size_t max_nodes = 4096;
    const size_t graph_ctx_size =
        ggml_tensor_overhead() * max_nodes + ggml_graph_overhead_custom(max_nodes, false);
    ggml_init_params params;
    params.mem_size = graph_ctx_size;
    params.mem_buffer = nullptr;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        set_error(error, "failed to initialize text layer graph context");
        return false;
    }

    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, backend->hidden, input.tokens);
    ggml_set_name(x, "text_layer_input");
    ggml_set_input(x);

    ggml_tensor * positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, input.tokens);
    ggml_set_name(positions, "text_layer_positions");
    ggml_set_input(positions);

    ggml_tensor * y = backend_text_layer_forward_ggml(
        ctx,
        backend->tensors,
        x,
        positions,
        backend->hidden,
        backend->n_heads,
        backend->n_kv_heads,
        backend->head_dim,
        backend->intermediate,
        backend->rope_theta,
        backend->rms_norm_eps);
    ggml_set_name(y, "text_layer_output");
    ggml_set_output(y);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, max_nodes, false);
    ggml_build_forward_expand(graph, y);
    if (!ggml_backend_sched_alloc_graph(backend->sched, graph)) {
        ggml_backend_sched_reset(backend->sched);
        ggml_free(ctx);
        set_error(error, "GGML backend text layer graph allocation failed");
        return false;
    }

    std::vector<int32_t> positions_data(static_cast<size_t>(input.tokens), 0);
    for (int token = 0; token < input.tokens; ++token) {
        positions_data[static_cast<size_t>(token)] = token;
    }

    ggml_backend_tensor_set(x, input.values.data(), 0, input.values.size() * sizeof(float));
    ggml_backend_tensor_set(positions, positions_data.data(), 0, positions_data.size() * sizeof(int32_t));
    const ggml_status status = ggml_backend_sched_graph_compute(backend->sched, graph);
    if (status != GGML_STATUS_SUCCESS) {
        ggml_backend_sched_reset(backend->sched);
        ggml_free(ctx);
        set_error(error, "GGML backend text layer graph compute failed");
        return false;
    }

    QwenAsrTextLayerOutput result;
    result.tokens = input.tokens;
    result.hidden = input.hidden;
    result.values.assign(static_cast<size_t>(input.tokens) * input.hidden, 0.0f);
    ggml_backend_tensor_get(y, result.values.data(), 0, result.values.size() * sizeof(float));

    ggml_backend_sched_reset(backend->sched);
    ggml_free(ctx);
    *out = std::move(result);
    return true;
}

void qwenasr_text_decoder_backend_free(QwenAsrTextDecoderBackend * backend) {
    if (!backend) {
        return;
    }
    if (backend->sched) {
        ggml_backend_sched_free(backend->sched);
        backend->sched = nullptr;
    }
    if (backend->weight_buffer) {
        ggml_backend_buffer_free(backend->weight_buffer);
        backend->weight_buffer = nullptr;
    }
    if (backend->kv_buffer) {
        ggml_backend_buffer_free(backend->kv_buffer);
        backend->kv_buffer = nullptr;
    }
    if (backend->weight_ctx) {
        ggml_free(backend->weight_ctx);
        backend->weight_ctx = nullptr;
    }
    if (backend->kv_ctx) {
        ggml_free(backend->kv_ctx);
        backend->kv_ctx = nullptr;
    }
    qwenasr_ggml_backend_set_free(&backend->backends);
    delete backend;
}

static bool text_decoder_backend_alloc_kv_cache(
    QwenAsrTextDecoderBackend * backend,
    int max_seq_len,
    std::string * error) {
    if (!backend || max_seq_len <= 0) {
        set_error(error, "invalid text decoder KV-cache configuration");
        return false;
    }
    if (backend->kv_ctx || backend->kv_buffer) {
        set_error(error, "text decoder KV-cache is already allocated");
        return false;
    }

    backend->max_seq_len = max_seq_len;
    backend->cur_len = 0;
    backend->k_cache.assign(static_cast<size_t>(backend->n_layers), nullptr);
    backend->v_cache.assign(static_cast<size_t>(backend->n_layers), nullptr);

    const size_t kv_ctx_size =
        ggml_tensor_overhead() * static_cast<size_t>(2 * backend->n_layers + 4) + 1024ull * 1024ull;
    ggml_init_params params;
    params.mem_size = kv_ctx_size;
    params.mem_buffer = nullptr;
    params.no_alloc = true;
    backend->kv_ctx = ggml_init(params);
    if (!backend->kv_ctx) {
        set_error(error, "failed to initialize text decoder KV-cache context");
        return false;
    }

    for (int layer = 0; layer < backend->n_layers; ++layer) {
        backend->k_cache[static_cast<size_t>(layer)] = ggml_new_tensor_3d(
            backend->kv_ctx,
            GGML_TYPE_F32,
            backend->head_dim,
            max_seq_len,
            backend->n_kv_heads);
        backend->v_cache[static_cast<size_t>(layer)] = ggml_new_tensor_3d(
            backend->kv_ctx,
            GGML_TYPE_F32,
            backend->head_dim,
            max_seq_len,
            backend->n_kv_heads);
        ggml_set_name(backend->k_cache[static_cast<size_t>(layer)], ("text_k_cache_" + std::to_string(layer)).c_str());
        ggml_set_name(backend->v_cache[static_cast<size_t>(layer)], ("text_v_cache_" + std::to_string(layer)).c_str());
    }

    backend->kv_buffer = ggml_backend_alloc_ctx_tensors(backend->kv_ctx, backend->backends.primary);
    if (!backend->kv_buffer) {
        set_error(error, "failed to allocate text decoder KV-cache backend buffer");
        return false;
    }
    ggml_backend_buffer_clear(backend->kv_buffer, 0);
    return true;
}

bool qwenasr_text_decoder_backend_init(
    const QwenAsrGgufModel & model,
    int n_threads,
    int n_layers,
    int hidden,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int intermediate,
    int vocab,
    float rope_theta,
    float rms_norm_eps,
    QwenAsrTextDecoderBackend ** out,
    std::string * error,
    QwenAsrGgmlDevice device) {
    if (!out) {
        set_error(error, "qwenasr_text_decoder_backend_init: out is null");
        return false;
    }
    *out = nullptr;
    const int q_dim = n_heads * head_dim;
    const int kv_dim = n_kv_heads * head_dim;
    if (n_layers <= 0 || hidden <= 0 || n_heads <= 0 || n_kv_heads <= 0 ||
        head_dim <= 0 || intermediate <= 0 || vocab <= 0 ||
        n_heads % n_kv_heads != 0 || q_dim <= 0 || kv_dim <= 0) {
        set_error(error, "invalid text decoder backend configuration");
        return false;
    }
    if (n_threads <= 0) {
        n_threads = 1;
    }

    QwenAsrTextDecoderBackend * runtime = new QwenAsrTextDecoderBackend();
    runtime->n_threads = n_threads;
    runtime->n_layers = n_layers;
    runtime->hidden = hidden;
    runtime->n_heads = n_heads;
    runtime->n_kv_heads = n_kv_heads;
    runtime->head_dim = head_dim;
    runtime->intermediate = intermediate;
    runtime->vocab = vocab;
    runtime->rope_theta = rope_theta;
    runtime->rms_norm_eps = rms_norm_eps;

    if (!qwenasr_ggml_backend_set_init(device, n_threads, &runtime->backends, error)) {
        qwenasr_text_decoder_backend_free(runtime);
        return false;
    }

    constexpr size_t max_nodes = 8192;
    ggml_backend_t backends[2] = {};
    qwenasr_ggml_backend_set_array(runtime->backends, backends);
    runtime->sched = ggml_backend_sched_new(
        backends,
        nullptr,
        qwenasr_ggml_backend_set_count(runtime->backends),
        max_nodes,
        false,
        true);
    if (!runtime->sched) {
        qwenasr_text_decoder_backend_free(runtime);
        set_error(error, "failed to initialize GGML text decoder backend scheduler");
        return false;
    }

    const int n_weight_tensors = n_layers * 11 + 2;
    const size_t weight_ctx_size = static_cast<size_t>(n_weight_tensors) * ggml_tensor_overhead() + 1024ull * 1024ull;
    ggml_init_params params;
    params.mem_size = weight_ctx_size;
    params.mem_buffer = nullptr;
    params.no_alloc = true;
    runtime->weight_ctx = ggml_init(params);
    if (!runtime->weight_ctx) {
        qwenasr_text_decoder_backend_free(runtime);
        set_error(error, "failed to initialize text decoder weight context");
        return false;
    }

    std::vector<BackendPendingCopy> pending;
    pending.reserve(static_cast<size_t>(n_weight_tensors));
    runtime->layers.reserve(static_cast<size_t>(n_layers));
    for (int layer = 0; layer < n_layers; ++layer) {
        TextLayerTensors views;
        if (!load_text_layer_tensors(model, layer, hidden, q_dim, kv_dim, head_dim, intermediate, &views, error)) {
            qwenasr_text_decoder_backend_free(runtime);
            return false;
        }
        runtime->layers.push_back(backend_text_layer_tensors_from_views(runtime->weight_ctx, views, &pending));
    }

    QwenAsrGgufTensorView output_norm_w;
    QwenAsrGgufTensorView output_w;
    if (!require_f32_tensor(model, "text.output_norm.weight", &output_norm_w, error) ||
        !require_f32_tensor(model, "text.output.weight", &output_w, error) ||
        !require_shape(output_norm_w, hidden, error) ||
        !require_shape(output_w, hidden, vocab, error)) {
        qwenasr_text_decoder_backend_free(runtime);
        return false;
    }
    runtime->output_norm_w = new_backend_weight_from_view(runtime->weight_ctx, output_norm_w, &pending);
    runtime->output_w = new_backend_weight_from_view(runtime->weight_ctx, output_w, &pending);

    runtime->weight_buffer = ggml_backend_alloc_ctx_tensors(runtime->weight_ctx, runtime->backends.primary);
    if (!runtime->weight_buffer) {
        qwenasr_text_decoder_backend_free(runtime);
        set_error(error, "failed to allocate text decoder backend weight buffer");
        return false;
    }
    ggml_backend_buffer_set_usage(runtime->weight_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    for (const BackendPendingCopy & copy : pending) {
        ggml_backend_tensor_set(copy.tensor, copy.data, 0, copy.nbytes);
    }

    *out = runtime;
    return true;
}

bool qwenasr_text_decoder_backend_init_cached(
    const QwenAsrGgufModel & model,
    int n_threads,
    int n_layers,
    int hidden,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int intermediate,
    int vocab,
    int max_seq_len,
    float rope_theta,
    float rms_norm_eps,
    QwenAsrTextDecoderBackend ** out,
    std::string * error,
    QwenAsrGgmlDevice device) {
    if (!out) {
        set_error(error, "qwenasr_text_decoder_backend_init_cached: out is null");
        return false;
    }
    *out = nullptr;
    QwenAsrTextDecoderBackend * backend = nullptr;
    if (!qwenasr_text_decoder_backend_init(
            model,
            n_threads,
            n_layers,
            hidden,
            n_heads,
            n_kv_heads,
            head_dim,
            intermediate,
            vocab,
            rope_theta,
            rms_norm_eps,
            &backend,
            error,
            device)) {
        return false;
    }
    if (!text_decoder_backend_alloc_kv_cache(backend, max_seq_len, error)) {
        qwenasr_text_decoder_backend_free(backend);
        return false;
    }
    *out = backend;
    return true;
}

static bool text_decoder_backend_cached_logits(
    QwenAsrTextDecoderBackend * backend,
    const float * input_values,
    int tokens,
    int n_past,
    std::vector<float> * logits,
    std::string * error) {
    if (!backend || !input_values || !logits) {
        set_error(error, "text_decoder_backend_cached_logits: invalid arguments");
        return false;
    }
    if (!backend->kv_ctx || !backend->kv_buffer || tokens <= 0 || n_past < 0 ||
        n_past + tokens > backend->max_seq_len) {
        set_error(error, "invalid text decoder cached forward configuration");
        return false;
    }

    constexpr size_t max_nodes = 8192;
    const size_t graph_ctx_size =
        ggml_tensor_overhead() * max_nodes + ggml_graph_overhead_custom(max_nodes, false);
    ggml_init_params params;
    params.mem_size = graph_ctx_size;
    params.mem_buffer = nullptr;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        set_error(error, "failed to initialize text cached decode graph context");
        return false;
    }

    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, backend->hidden, tokens);
    ggml_set_name(x, "text_cached_input");
    ggml_set_input(x);

    ggml_tensor * positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, tokens);
    ggml_set_name(positions, "text_cached_positions");
    ggml_set_input(positions);

    ggml_tensor * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_past + tokens, tokens);
    ggml_set_name(mask, "text_cached_mask");
    ggml_set_input(mask);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, max_nodes, false);
    ggml_tensor * y = x;
    for (int layer = 0; layer < backend->n_layers; ++layer) {
        y = backend_text_layer_forward_cached_ggml(
            ctx,
            backend->layers[static_cast<size_t>(layer)],
            y,
            positions,
            mask,
            backend->k_cache[static_cast<size_t>(layer)],
            backend->v_cache[static_cast<size_t>(layer)],
            n_past,
            backend->hidden,
            backend->n_heads,
            backend->n_kv_heads,
            backend->head_dim,
            backend->intermediate,
            backend->rope_theta,
            backend->rms_norm_eps,
            graph);
    }
    y = backend_text_logits_ggml(
        ctx,
        y,
        backend->output_norm_w,
        backend->output_w,
        backend->hidden,
        tokens,
        backend->rms_norm_eps);
    ggml_set_name(y, "text_cached_logits");
    ggml_set_output(y);
    ggml_build_forward_expand(graph, y);

    ggml_backend_sched_reset(backend->sched);
    if (!ggml_backend_sched_alloc_graph(backend->sched, graph)) {
        ggml_backend_sched_reset(backend->sched);
        ggml_free(ctx);
        set_error(error, "GGML backend text cached graph allocation failed");
        return false;
    }

    std::vector<int32_t> position_data(static_cast<size_t>(tokens), 0);
    for (int token = 0; token < tokens; ++token) {
        position_data[static_cast<size_t>(token)] = n_past + token;
    }

    std::vector<float> mask_data(static_cast<size_t>(tokens) * static_cast<size_t>(n_past + tokens), -INFINITY);
    for (int query = 0; query < tokens; ++query) {
        const int max_key = n_past + query;
        for (int key = 0; key <= max_key; ++key) {
            mask_data[static_cast<size_t>(query) * static_cast<size_t>(n_past + tokens) + static_cast<size_t>(key)] = 0.0f;
        }
    }

    ggml_backend_tensor_set(x, input_values, 0, static_cast<size_t>(tokens) * backend->hidden * sizeof(float));
    ggml_backend_tensor_set(positions, position_data.data(), 0, position_data.size() * sizeof(int32_t));
    ggml_backend_tensor_set(mask, mask_data.data(), 0, mask_data.size() * sizeof(float));
    const ggml_status status = ggml_backend_sched_graph_compute(backend->sched, graph);
    if (status != GGML_STATUS_SUCCESS) {
        ggml_backend_sched_reset(backend->sched);
        ggml_free(ctx);
        set_error(error, "GGML backend text cached graph compute failed");
        return false;
    }

    logits->assign(static_cast<size_t>(backend->vocab), 0.0f);
    ggml_backend_tensor_get(y, logits->data(), 0, logits->size() * sizeof(float));
    backend->cur_len = n_past + tokens;

    ggml_backend_sched_reset(backend->sched);
    ggml_free(ctx);
    return true;
}

bool qwenasr_text_prefill_backend_forward(
    QwenAsrTextDecoderBackend * backend,
    const QwenAsrDecoderInputOutput & input,
    QwenAsrTextPrefillOutput * out,
    std::string * error) {
    if (!backend || !out) {
        set_error(error, "qwenasr_text_prefill_backend_forward: backend or out is null");
        return false;
    }
    if (input.tokens <= 0 || input.hidden != backend->hidden ||
        input.values.size() != static_cast<size_t>(input.tokens) * input.hidden) {
        set_error(error, "invalid decoder input for text prefill backend");
        return false;
    }

    constexpr size_t max_nodes = 8192;
    const size_t graph_ctx_size =
        ggml_tensor_overhead() * max_nodes + ggml_graph_overhead_custom(max_nodes, false);
    ggml_init_params params;
    params.mem_size = graph_ctx_size;
    params.mem_buffer = nullptr;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        set_error(error, "failed to initialize text prefill graph context");
        return false;
    }

    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, backend->hidden, input.tokens);
    ggml_set_name(x, "text_prefill_input");
    ggml_set_input(x);

    ggml_tensor * positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, input.tokens);
    ggml_set_name(positions, "text_prefill_positions");
    ggml_set_input(positions);

    ggml_tensor * y = x;
    for (int layer = 0; layer < backend->n_layers; ++layer) {
        y = backend_text_layer_forward_ggml(
            ctx,
            backend->layers[static_cast<size_t>(layer)],
            y,
            positions,
            backend->hidden,
            backend->n_heads,
            backend->n_kv_heads,
            backend->head_dim,
            backend->intermediate,
            backend->rope_theta,
            backend->rms_norm_eps);
    }
    y = backend_text_logits_ggml(
        ctx,
        y,
        backend->output_norm_w,
        backend->output_w,
        backend->hidden,
        input.tokens,
        backend->rms_norm_eps);
    ggml_set_name(y, "text_prefill_logits");
    ggml_set_output(y);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, max_nodes, false);
    ggml_build_forward_expand(graph, y);
    if (!ggml_backend_sched_alloc_graph(backend->sched, graph)) {
        ggml_backend_sched_reset(backend->sched);
        ggml_free(ctx);
        set_error(error, "GGML backend text prefill graph allocation failed");
        return false;
    }

    std::vector<int32_t> positions_data(static_cast<size_t>(input.tokens), 0);
    for (int token = 0; token < input.tokens; ++token) {
        positions_data[static_cast<size_t>(token)] = token;
    }

    ggml_backend_tensor_set(x, input.values.data(), 0, input.values.size() * sizeof(float));
    ggml_backend_tensor_set(positions, positions_data.data(), 0, positions_data.size() * sizeof(int32_t));
    const ggml_status status = ggml_backend_sched_graph_compute(backend->sched, graph);
    if (status != GGML_STATUS_SUCCESS) {
        ggml_backend_sched_reset(backend->sched);
        ggml_free(ctx);
        set_error(error, "GGML backend text prefill graph compute failed");
        return false;
    }

    QwenAsrTextPrefillOutput result;
    result.tokens = input.tokens;
    result.hidden = input.hidden;
    result.vocab = backend->vocab;
    result.logits.assign(static_cast<size_t>(backend->vocab), 0.0f);
    ggml_backend_tensor_get(y, result.logits.data(), 0, result.logits.size() * sizeof(float));

    ggml_backend_sched_reset(backend->sched);
    ggml_free(ctx);
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
                nullptr,
                n_heads,
                n_kv_heads,
                head_dim,
                intermediate,
                rope_theta,
                rms_norm_eps,
                &next,
                nullptr,
                error)) {
            return false;
        }
        states = std::move(next);
        next.clear();
    }

    std::vector<float> logits;
    if (!compute_logits_for_state_cpu(
            model,
            states.data() + static_cast<size_t>(input.tokens - 1) * input.hidden,
            input.hidden,
            vocab,
            rms_norm_eps,
            nullptr,
            nullptr,
            &logits,
            error)) {
        return false;
    }

    QwenAsrTextPrefillOutput result;
    result.tokens = input.tokens;
    result.hidden = input.hidden;
    result.vocab = vocab;
    result.logits = std::move(logits);
    *out = std::move(result);
    return true;
}

bool qwenasr_text_generate_greedy_cpu(
    const QwenAsrGgufModel & model,
    const QwenAsrDecoderInputOutput & input,
    int max_new_tokens,
    int n_layers,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int intermediate,
    int vocab,
    float rope_theta,
    float rms_norm_eps,
    const std::vector<int32_t> & stop_ids,
    QwenAsrTextGenerateOutput * out,
    std::string * error) {
    if (!out) {
        set_error(error, "qwenasr_text_generate_greedy_cpu: out is null");
        return false;
    }
    if (max_new_tokens < 0) {
        set_error(error, "max_new_tokens must be non-negative");
        return false;
    }
    if (input.tokens <= 0 || input.hidden <= 0 ||
        input.values.size() != static_cast<size_t>(input.tokens) * input.hidden) {
        set_error(error, "invalid text generation input");
        return false;
    }

    QwenAsrGgufTensorView embd;
    if (!require_f32_tensor(model, "text.token_embd.weight", &embd, error)) {
        return false;
    }
    if (!require_shape(embd, input.hidden, vocab, error)) {
        return false;
    }
    const float * embedding = tensor_f32(embd);

    QwenAsrDecoderInputOutput state = input;
    QwenAsrTextGenerateOutput result;
    result.prompt_tokens = input.tokens;
    result.total_tokens = input.tokens;
    result.hidden = input.hidden;
    result.vocab = vocab;
    result.generated_ids.reserve(static_cast<size_t>(max_new_tokens));

    for (int step = 0; step < max_new_tokens; ++step) {
        QwenAsrTextPrefillOutput prefill;
        if (!qwenasr_text_prefill_forward_cpu(
                model,
                state,
                n_layers,
                n_heads,
                n_kv_heads,
                head_dim,
                intermediate,
                vocab,
                rope_theta,
                rms_norm_eps,
                &prefill,
                error)) {
            return false;
        }

        int32_t next_id = -1;
        float best = -std::numeric_limits<float>::infinity();
        for (int i = 0; i < prefill.vocab; ++i) {
            const float value = prefill.logits[static_cast<size_t>(i)];
            if (value > best) {
                best = value;
                next_id = static_cast<int32_t>(i);
            }
        }
        if (next_id < 0 || next_id >= vocab) {
            set_error(error, "generated token id is out of range");
            return false;
        }

        result.generated_ids.push_back(next_id);
        result.total_tokens = input.tokens + static_cast<int>(result.generated_ids.size());
        bool stop = false;
        for (const int32_t stop_id : stop_ids) {
            if (next_id == stop_id) {
                stop = true;
                break;
            }
        }
        if (stop) {
            result.stopped = true;
            break;
        }

        const size_t old_values = state.values.size();
        state.input_ids.push_back(next_id);
        state.values.resize(old_values + static_cast<size_t>(state.hidden));
        std::memcpy(
            state.values.data() + old_values,
            embedding + static_cast<size_t>(next_id) * state.hidden,
            static_cast<size_t>(state.hidden) * sizeof(float));
        ++state.tokens;
    }

    *out = std::move(result);
    return true;
}

bool qwenasr_text_generate_cached_cpu(
    const QwenAsrGgufModel & model,
    const QwenAsrDecoderInputOutput & input,
    int max_new_tokens,
    int n_layers,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int intermediate,
    int vocab,
    float rope_theta,
    float rms_norm_eps,
    const std::vector<int32_t> & stop_ids,
    QwenAsrTextGenerateOutput * out,
    std::string * error) {
    if (!out) {
        set_error(error, "qwenasr_text_generate_cached_cpu: out is null");
        return false;
    }
    if (max_new_tokens < 0) {
        set_error(error, "max_new_tokens must be non-negative");
        return false;
    }
    if (input.tokens <= 0 || input.hidden <= 0 || n_layers <= 0 ||
        input.values.size() != static_cast<size_t>(input.tokens) * input.hidden) {
        set_error(error, "invalid cached text generation input");
        return false;
    }

    QwenAsrTextGenerateOutput result;
    result.prompt_tokens = input.tokens;
    result.total_tokens = input.tokens;
    result.hidden = input.hidden;
    result.vocab = vocab;
    result.generated_ids.reserve(static_cast<size_t>(max_new_tokens));
    if (max_new_tokens == 0) {
        *out = std::move(result);
        return true;
    }

    const int q_dim = n_heads * head_dim;
    const int kv_dim = n_kv_heads * head_dim;
    TextDecoderTensorContext tensors;
    if (!load_text_decoder_tensor_context(
            model,
            n_layers,
            input.hidden,
            q_dim,
            kv_dim,
            head_dim,
            intermediate,
            vocab,
            &tensors,
            error)) {
        return false;
    }
    const float * embedding = tensor_f32(tensors.embd_w);

    std::vector<TextLayerKvCache> caches(static_cast<size_t>(n_layers));
    std::vector<float> states = input.values;
    std::vector<float> next;
    for (int layer = 0; layer < n_layers; ++layer) {
        if (!text_layer_forward_values_cpu(
                model,
                states,
                input.tokens,
                input.hidden,
                layer,
                &tensors.layers[static_cast<size_t>(layer)],
                n_heads,
                n_kv_heads,
                head_dim,
                intermediate,
                rope_theta,
                rms_norm_eps,
                &next,
                &caches[static_cast<size_t>(layer)],
                error)) {
            return false;
        }
        caches[static_cast<size_t>(layer)].k.reserve(
            static_cast<size_t>(input.tokens + max_new_tokens) * kv_dim);
        caches[static_cast<size_t>(layer)].v.reserve(
            static_cast<size_t>(input.tokens + max_new_tokens) * kv_dim);
        states = std::move(next);
        next.clear();
    }

    std::vector<float> logits;
    if (!compute_logits_for_state_cpu(
            model,
            states.data() + static_cast<size_t>(input.tokens - 1) * input.hidden,
            input.hidden,
            vocab,
            rms_norm_eps,
            &tensors.output_norm_w,
            &tensors.output_w,
            &logits,
            error)) {
        return false;
    }

    for (int step = 0; step < max_new_tokens; ++step) {
        int32_t next_id = -1;
        float best = -std::numeric_limits<float>::infinity();
        for (int i = 0; i < vocab; ++i) {
            const float value = logits[static_cast<size_t>(i)];
            if (value > best) {
                best = value;
                next_id = static_cast<int32_t>(i);
            }
        }
        if (next_id < 0 || next_id >= vocab) {
            set_error(error, "generated token id is out of range");
            return false;
        }

        result.generated_ids.push_back(next_id);
        result.total_tokens = input.tokens + static_cast<int>(result.generated_ids.size());
        bool stop = false;
        for (const int32_t stop_id : stop_ids) {
            if (next_id == stop_id) {
                stop = true;
                break;
            }
        }
        if (stop) {
            result.stopped = true;
            break;
        }
        if (step + 1 >= max_new_tokens) {
            break;
        }

        std::vector<float> state(
            embedding + static_cast<size_t>(next_id) * input.hidden,
            embedding + static_cast<size_t>(next_id + 1) * input.hidden);
        for (int layer = 0; layer < n_layers; ++layer) {
            std::vector<float> layer_out;
            if (!text_layer_decode_cached_cpu(
                    model,
                    state,
                    layer,
                    &tensors.layers[static_cast<size_t>(layer)],
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    intermediate,
                    rope_theta,
                    rms_norm_eps,
                    &caches[static_cast<size_t>(layer)],
                    &layer_out,
                    error)) {
                return false;
            }
            state = std::move(layer_out);
        }

        if (!compute_logits_for_state_cpu(
                model,
                state.data(),
                input.hidden,
                vocab,
                rms_norm_eps,
                &tensors.output_norm_w,
                &tensors.output_w,
                &logits,
                error)) {
            return false;
        }
    }

    *out = std::move(result);
    return true;
}

bool qwenasr_text_generate_cached_backend(
    QwenAsrTextDecoderBackend * backend,
    const QwenAsrGgufModel & model,
    const QwenAsrDecoderInputOutput & input,
    int max_new_tokens,
    const std::vector<int32_t> & stop_ids,
    QwenAsrTextGenerateOutput * out,
    std::string * error) {
    if (!backend || !out) {
        set_error(error, "qwenasr_text_generate_cached_backend: backend or out is null");
        return false;
    }
    if (max_new_tokens < 0) {
        set_error(error, "max_new_tokens must be non-negative");
        return false;
    }
    if (input.tokens <= 0 || input.hidden != backend->hidden ||
        input.values.size() != static_cast<size_t>(input.tokens) * input.hidden) {
        set_error(error, "invalid backend cached text generation input");
        return false;
    }
    if (input.tokens + max_new_tokens > backend->max_seq_len) {
        set_error(error, "backend text generation exceeds KV-cache capacity");
        return false;
    }

    QwenAsrTextGenerateOutput result;
    result.prompt_tokens = input.tokens;
    result.total_tokens = input.tokens;
    result.hidden = input.hidden;
    result.vocab = backend->vocab;
    result.generated_ids.reserve(static_cast<size_t>(max_new_tokens));
    if (max_new_tokens == 0) {
        *out = std::move(result);
        return true;
    }

    QwenAsrGgufTensorView embd;
    if (!require_f32_tensor(model, "text.token_embd.weight", &embd, error) ||
        !require_shape(embd, backend->hidden, backend->vocab, error)) {
        return false;
    }
    const float * embedding = tensor_f32(embd);

    backend->cur_len = 0;
    std::vector<float> logits;
    if (!text_decoder_backend_cached_logits(
            backend,
            input.values.data(),
            input.tokens,
            0,
            &logits,
            error)) {
        return false;
    }

    for (int step = 0; step < max_new_tokens; ++step) {
        int32_t next_id = -1;
        float best = -std::numeric_limits<float>::infinity();
        for (int i = 0; i < backend->vocab; ++i) {
            const float value = logits[static_cast<size_t>(i)];
            if (value > best) {
                best = value;
                next_id = static_cast<int32_t>(i);
            }
        }
        if (next_id < 0 || next_id >= backend->vocab) {
            set_error(error, "generated token id is out of range");
            return false;
        }

        result.generated_ids.push_back(next_id);
        result.total_tokens = input.tokens + static_cast<int>(result.generated_ids.size());
        bool stop = false;
        for (const int32_t stop_id : stop_ids) {
            if (next_id == stop_id) {
                stop = true;
                break;
            }
        }
        if (stop) {
            result.stopped = true;
            break;
        }
        if (step + 1 >= max_new_tokens) {
            break;
        }

        const float * row = embedding + static_cast<size_t>(next_id) * backend->hidden;
        if (!text_decoder_backend_cached_logits(
                backend,
                row,
                1,
                backend->cur_len,
                &logits,
                error)) {
            return false;
        }
    }

    *out = std::move(result);
    return true;
}
