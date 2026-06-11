#include "audio-encoder.h"

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
#include <initializer_list>
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

static bool shape_eq(const QwenAsrGgufTensorView & tensor, std::initializer_list<int64_t> shape) {
    return tensor.ne == std::vector<int64_t>(shape);
}

static int conv_stride2_pad1_len(int input) {
    return (input + 1) / 2;
}

static bool validate_features(const QwenAsrFeatures & features, const char * caller, std::string * error) {
    if (features.n_mels <= 0 || features.n_frames <= 0) {
        set_error(error, std::string(caller) + ": features are empty");
        return false;
    }
    if (static_cast<size_t>(features.n_mels * features.n_frames) != features.values.size()) {
        set_error(error, std::string(caller) + ": feature size mismatch");
        return false;
    }
    return true;
}

static std::vector<float> build_chunked_feature_input(
    const QwenAsrFeatures & features,
    const QwenAsrAudioGeometry & geometry) {
    const int chunks = geometry.n_chunks;
    const int frames_in = geometry.max_chunk_input_len;
    std::vector<float> input_values(
        static_cast<size_t>(frames_in) * features.n_mels * chunks,
        0.0f);
    for (int chunk = 0; chunk < chunks; ++chunk) {
        const int chunk_len = geometry.chunk_input_lengths[static_cast<size_t>(chunk)];
        for (int mel = 0; mel < features.n_mels; ++mel) {
            for (int frame = 0; frame < chunk_len; ++frame) {
                const int src_frame = chunk * geometry.chunk_window + frame;
                if (src_frame >= features.n_frames) {
                    continue;
                }
                const size_t dst =
                    static_cast<size_t>(frame) +
                    static_cast<size_t>(frames_in) *
                        (static_cast<size_t>(mel) + static_cast<size_t>(features.n_mels) * chunk);
                input_values[dst] = features.values[static_cast<size_t>(mel) * features.n_frames + src_frame];
            }
        }
    }
    return input_values;
}

static std::vector<float> build_audio_positional_values(int frames, int hidden) {
    const int half = hidden / 2;
    const double log_timescale_increment = std::log(10000.0) / static_cast<double>(half - 1);
    std::vector<float> positional(static_cast<size_t>(frames) * hidden, 0.0f);
    for (int frame = 0; frame < frames; ++frame) {
        for (int i = 0; i < half; ++i) {
            const double inv_timescale = std::exp(-log_timescale_increment * static_cast<double>(i));
            const double scaled_time = static_cast<double>(frame) * inv_timescale;
            positional[static_cast<size_t>(frame) * hidden + i] = static_cast<float>(std::sin(scaled_time));
            positional[static_cast<size_t>(frame) * hidden + half + i] = static_cast<float>(std::cos(scaled_time));
        }
    }
    return positional;
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

static ggml_tensor * attention_ggml(
    ggml_context * ctx,
    ggml_tensor * q,
    ggml_tensor * k,
    ggml_tensor * v,
    int head_dim,
    int n_heads,
    int tokens,
    bool use_flash_attn) {
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    if (use_flash_attn) {
        ggml_tensor * q_f = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
        ggml_tensor * k_f = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
        ggml_tensor * v_f = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));
        ggml_tensor * attn = ggml_flash_attn_ext(ctx, q_f, k_f, v_f, nullptr, scale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(attn, GGML_PREC_F32);
        attn = ggml_cont(ctx, attn);
        return ggml_reshape_2d(ctx, attn, head_dim * n_heads, tokens);
    }

    ggml_tensor * q_p = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
    ggml_tensor * k_p = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
    ggml_tensor * v_p = ggml_cont(ctx, ggml_permute(ctx, v, 1, 2, 0, 3));

    ggml_tensor * scores = ggml_mul_mat(ctx, k_p, q_p);
    ggml_mul_mat_set_prec(scores, GGML_PREC_F32);
    scores = ggml_soft_max_ext(ctx, scores, nullptr, scale, 0.0f);

    ggml_tensor * attn = ggml_mul_mat(ctx, v_p, scores);
    ggml_mul_mat_set_prec(attn, GGML_PREC_F32);
    attn = ggml_cont(ctx, ggml_permute(ctx, attn, 0, 2, 1, 3));
    return ggml_reshape_2d(ctx, attn, head_dim * n_heads, tokens);
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

    ggml_tensor * attn = attention_ggml(
        ctx,
        q,
        k,
        v,
        head_dim,
        n_heads,
        tokens,
        qwenasr_ggml_flash_attn_enabled());

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

struct BackendPendingCopy {
    ggml_tensor * tensor = nullptr;
    const void * data = nullptr;
    size_t nbytes = 0;
};

struct BackendAudioLayerTensors {
    ggml_tensor * attn_norm_w = nullptr;
    ggml_tensor * attn_norm_b = nullptr;
    ggml_tensor * ffn_norm_w = nullptr;
    ggml_tensor * ffn_norm_b = nullptr;
    ggml_tensor * q_w = nullptr;
    ggml_tensor * q_b = nullptr;
    ggml_tensor * k_w = nullptr;
    ggml_tensor * k_b = nullptr;
    ggml_tensor * v_w = nullptr;
    ggml_tensor * v_b = nullptr;
    ggml_tensor * out_w = nullptr;
    ggml_tensor * out_b = nullptr;
    ggml_tensor * up_w = nullptr;
    ggml_tensor * up_b = nullptr;
    ggml_tensor * down_w = nullptr;
    ggml_tensor * down_b = nullptr;
};

struct BackendAudioProjectorTensors {
    ggml_tensor * post_norm_w = nullptr;
    ggml_tensor * post_norm_b = nullptr;
    ggml_tensor * proj0_w = nullptr;
    ggml_tensor * proj0_b = nullptr;
    ggml_tensor * proj1_w = nullptr;
    ggml_tensor * proj1_b = nullptr;
};

struct BackendAudioPrepTensors {
    ggml_tensor * w0 = nullptr;
    ggml_tensor * b0 = nullptr;
    ggml_tensor * w1 = nullptr;
    ggml_tensor * b1 = nullptr;
    ggml_tensor * w2 = nullptr;
    ggml_tensor * b2 = nullptr;
    ggml_tensor * wout = nullptr;
};

struct QwenAsrAudioEncoderBackend {
    QwenAsrGgmlBackendSet backends;
    ggml_backend_sched_t sched = nullptr;
    ggml_context * weight_ctx = nullptr;
    ggml_backend_buffer_t weight_buffer = nullptr;
    int n_threads = 1;
    int n_layers = 0;
    int n_heads = 0;
    int hidden = 0;
    int output_dim = 0;
    bool has_prep = false;
    int prep_n_mels = 0;
    int prep_channels = 0;
    std::vector<BackendAudioLayerTensors> layers;
    BackendAudioPrepTensors prep;
    BackendAudioProjectorTensors projector;
};

static ggml_tensor * new_backend_weight_from_view(
    ggml_context * ctx,
    const QwenAsrGgufTensorView & view,
    std::vector<BackendPendingCopy> * pending) {
    ggml_tensor * tensor = ggml_new_tensor(
        ctx,
        view.type,
        static_cast<int>(view.ne.size()),
        view.ne.data());
    ggml_set_name(tensor, view.name.c_str());
    pending->push_back(BackendPendingCopy { tensor, view.data, ggml_nbytes(tensor) });
    return tensor;
}

static ggml_tensor * new_backend_conv_bias_from_view(
    ggml_context * ctx,
    const QwenAsrGgufTensorView & view,
    int channels,
    std::vector<BackendPendingCopy> * pending) {
    ggml_tensor * tensor = ggml_new_tensor_4d(ctx, view.type, 1, 1, channels, 1);
    ggml_set_name(tensor, view.name.c_str());
    pending->push_back(BackendPendingCopy { tensor, view.data, ggml_nbytes(tensor) });
    return tensor;
}

static BackendAudioLayerTensors backend_audio_layer_tensors_from_views(
    ggml_context * ctx,
    const AudioLayerTensors & views,
    std::vector<BackendPendingCopy> * pending) {
    BackendAudioLayerTensors tensors;
    tensors.attn_norm_w = new_backend_weight_from_view(ctx, views.attn_norm_w, pending);
    tensors.attn_norm_b = new_backend_weight_from_view(ctx, views.attn_norm_b, pending);
    tensors.ffn_norm_w = new_backend_weight_from_view(ctx, views.ffn_norm_w, pending);
    tensors.ffn_norm_b = new_backend_weight_from_view(ctx, views.ffn_norm_b, pending);
    tensors.q_w = new_backend_weight_from_view(ctx, views.q_w, pending);
    tensors.q_b = new_backend_weight_from_view(ctx, views.q_b, pending);
    tensors.k_w = new_backend_weight_from_view(ctx, views.k_w, pending);
    tensors.k_b = new_backend_weight_from_view(ctx, views.k_b, pending);
    tensors.v_w = new_backend_weight_from_view(ctx, views.v_w, pending);
    tensors.v_b = new_backend_weight_from_view(ctx, views.v_b, pending);
    tensors.out_w = new_backend_weight_from_view(ctx, views.out_w, pending);
    tensors.out_b = new_backend_weight_from_view(ctx, views.out_b, pending);
    tensors.up_w = new_backend_weight_from_view(ctx, views.up_w, pending);
    tensors.up_b = new_backend_weight_from_view(ctx, views.up_b, pending);
    tensors.down_w = new_backend_weight_from_view(ctx, views.down_w, pending);
    tensors.down_b = new_backend_weight_from_view(ctx, views.down_b, pending);
    return tensors;
}

struct AudioPrepTensorViews {
    QwenAsrGgufTensorView w0;
    QwenAsrGgufTensorView b0;
    QwenAsrGgufTensorView w1;
    QwenAsrGgufTensorView b1;
    QwenAsrGgufTensorView w2;
    QwenAsrGgufTensorView b2;
    QwenAsrGgufTensorView wout;
    int channels = 0;
};

static bool load_audio_prep_tensors(
    const QwenAsrGgufModel & model,
    int n_mels,
    int hidden,
    AudioPrepTensorViews * views,
    std::string * error) {
    if (!views || n_mels <= 0 || hidden <= 0) {
        set_error(error, "invalid audio prep tensor configuration");
        return false;
    }
    if (!require_f32_tensor(model, "audio.conv.0.weight", &views->w0, error) ||
        !require_f32_tensor(model, "audio.conv.0.bias", &views->b0, error) ||
        !require_f32_tensor(model, "audio.conv.1.weight", &views->w1, error) ||
        !require_f32_tensor(model, "audio.conv.1.bias", &views->b1, error) ||
        !require_f32_tensor(model, "audio.conv.2.weight", &views->w2, error) ||
        !require_f32_tensor(model, "audio.conv.2.bias", &views->b2, error) ||
        !require_f32_tensor(model, "audio.conv_out.weight", &views->wout, error)) {
        return false;
    }

    views->channels = static_cast<int>(views->w0.ne.size() == 4 ? views->w0.ne[3] : 0);
    const int freq_out = conv_stride2_pad1_len(conv_stride2_pad1_len(conv_stride2_pad1_len(n_mels)));
    const int conv_out_width = views->channels * freq_out;
    if (views->channels <= 0 || freq_out <= 0 ||
        !shape_eq(views->w0, { 3, 3, 1, views->channels }) ||
        !shape_eq(views->b0, { views->channels }) ||
        !shape_eq(views->w1, { 3, 3, views->channels, views->channels }) ||
        !shape_eq(views->b1, { views->channels }) ||
        !shape_eq(views->w2, { 3, 3, views->channels, views->channels }) ||
        !shape_eq(views->b2, { views->channels }) ||
        views->wout.ne.size() != 2 ||
        views->wout.ne[0] != conv_out_width ||
        views->wout.ne[1] != hidden) {
        set_error(error, "audio prep tensor shape mismatch");
        return false;
    }
    return true;
}

static BackendAudioPrepTensors backend_audio_prep_tensors_from_views(
    ggml_context * ctx,
    const AudioPrepTensorViews & views,
    std::vector<BackendPendingCopy> * pending) {
    BackendAudioPrepTensors tensors;
    tensors.w0 = new_backend_weight_from_view(ctx, views.w0, pending);
    tensors.b0 = new_backend_conv_bias_from_view(ctx, views.b0, views.channels, pending);
    tensors.w1 = new_backend_weight_from_view(ctx, views.w1, pending);
    tensors.b1 = new_backend_conv_bias_from_view(ctx, views.b1, views.channels, pending);
    tensors.w2 = new_backend_weight_from_view(ctx, views.w2, pending);
    tensors.b2 = new_backend_conv_bias_from_view(ctx, views.b2, views.channels, pending);
    tensors.wout = new_backend_weight_from_view(ctx, views.wout, pending);
    return tensors;
}

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

static BackendAudioProjectorTensors backend_audio_projector_tensors_from_views(
    ggml_context * ctx,
    const AudioProjectorTensors & views,
    std::vector<BackendPendingCopy> * pending) {
    BackendAudioProjectorTensors tensors;
    tensors.post_norm_w = new_backend_weight_from_view(ctx, views.post_norm_w, pending);
    tensors.post_norm_b = new_backend_weight_from_view(ctx, views.post_norm_b, pending);
    tensors.proj0_w = new_backend_weight_from_view(ctx, views.proj0_w, pending);
    tensors.proj0_b = new_backend_weight_from_view(ctx, views.proj0_b, pending);
    tensors.proj1_w = new_backend_weight_from_view(ctx, views.proj1_w, pending);
    tensors.proj1_b = new_backend_weight_from_view(ctx, views.proj1_b, pending);
    return tensors;
}

static ggml_tensor * backend_audio_layer_forward_ggml(
    ggml_context * ctx,
    const BackendAudioLayerTensors & tensors,
    ggml_tensor * x,
    int hidden,
    int n_heads) {
    const int tokens = static_cast<int>(x->ne[1]);
    const int head_dim = hidden / n_heads;

    ggml_tensor * normed = layer_norm_ggml(ctx, x, tensors.attn_norm_w, tensors.attn_norm_b);
    ggml_tensor * q = linear_ggml(ctx, normed, tensors.q_w, tensors.q_b);
    ggml_tensor * k = linear_ggml(ctx, normed, tensors.k_w, tensors.k_b);
    ggml_tensor * v = linear_ggml(ctx, normed, tensors.v_w, tensors.v_b);

    q = ggml_reshape_3d(ctx, q, head_dim, n_heads, tokens);
    k = ggml_reshape_3d(ctx, k, head_dim, n_heads, tokens);
    v = ggml_reshape_3d(ctx, v, head_dim, n_heads, tokens);

    ggml_tensor * attn = attention_ggml(
        ctx,
        q,
        k,
        v,
        head_dim,
        n_heads,
        tokens,
        qwenasr_ggml_flash_attn_enabled());

    ggml_tensor * projected = linear_ggml(ctx, attn, tensors.out_w, tensors.out_b);
    ggml_tensor * hidden_states = ggml_add(ctx, x, projected);

    ggml_tensor * ffn_normed = layer_norm_ggml(ctx, hidden_states, tensors.ffn_norm_w, tensors.ffn_norm_b);
    ggml_tensor * ffn_up = linear_ggml(ctx, ffn_normed, tensors.up_w, tensors.up_b);
    ffn_up = ggml_gelu_erf(ctx, ffn_up);
    ggml_tensor * ffn_down = linear_ggml(ctx, ffn_up, tensors.down_w, tensors.down_b);
    return ggml_add(ctx, hidden_states, ffn_down);
}

static ggml_tensor * backend_audio_projector_forward_ggml(
    ggml_context * ctx,
    const BackendAudioProjectorTensors & tensors,
    ggml_tensor * x) {
    ggml_tensor * y = layer_norm_ggml(ctx, x, tensors.post_norm_w, tensors.post_norm_b);
    y = linear_ggml(ctx, y, tensors.proj0_w, tensors.proj0_b);
    y = ggml_gelu_erf(ctx, y);
    return linear_ggml(ctx, y, tensors.proj1_w, tensors.proj1_b);
}

static ggml_tensor * backend_audio_prep_forward_ggml(
    ggml_context * ctx,
    const BackendAudioPrepTensors & tensors,
    ggml_tensor * x,
    ggml_tensor * positional,
    int channels,
    int hidden,
    int n_mels,
    int frames_out,
    int chunks) {
    const int freq_out = conv_stride2_pad1_len(conv_stride2_pad1_len(conv_stride2_pad1_len(n_mels)));
    const int conv_out_width = channels * freq_out;

    auto conv_gelu = [&](ggml_tensor * input, ggml_tensor * weight, ggml_tensor * bias) -> ggml_tensor * {
        ggml_tensor * y = ggml_conv_2d(ctx, weight, input, 2, 2, 1, 1, 1, 1);
        y = ggml_add(ctx, y, ggml_repeat(ctx, bias, y));
        return ggml_gelu_erf(ctx, y);
    };

    ggml_tensor * y = conv_gelu(x, tensors.w0, tensors.b0);
    y = conv_gelu(y, tensors.w1, tensors.b1);
    y = conv_gelu(y, tensors.w2, tensors.b2);
    y = ggml_cont(ctx, ggml_permute(ctx, y, 2, 0, 1, 3));
    y = ggml_reshape_2d(ctx, y, conv_out_width, frames_out * chunks);
    y = ggml_mul_mat(ctx, tensors.wout, y);
    ggml_mul_mat_set_prec(y, GGML_PREC_F32);
    y = ggml_reshape_3d(ctx, y, hidden, frames_out, chunks);
    y = ggml_add(ctx, y, ggml_repeat(ctx, positional, y));
    return ggml_reshape_2d(ctx, y, hidden, frames_out * chunks);
}

void qwenasr_audio_encoder_backend_free(QwenAsrAudioEncoderBackend * backend) {
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

bool qwenasr_audio_encoder_backend_init(
    const QwenAsrGgufModel & model,
    int n_threads,
    int n_layers,
    int n_heads,
    int hidden,
    int output_dim,
    QwenAsrAudioEncoderBackend ** out,
    std::string * error,
    QwenAsrGgmlDevice device,
    int n_mels) {
    if (!out) {
        set_error(error, "qwenasr_audio_encoder_backend_init: out is null");
        return false;
    }
    *out = nullptr;
    if (n_layers <= 0 || n_heads <= 0 || hidden <= 0 || output_dim <= 0 || hidden % n_heads != 0) {
        set_error(error, "invalid audio encoder backend configuration");
        return false;
    }
    if (n_threads <= 0) {
        n_threads = 1;
    }

    QwenAsrAudioEncoderBackend * runtime = new QwenAsrAudioEncoderBackend();
    runtime->n_threads = n_threads;
    runtime->n_layers = n_layers;
    runtime->n_heads = n_heads;
    runtime->hidden = hidden;
    runtime->output_dim = output_dim;
    runtime->has_prep = n_mels > 0;
    runtime->prep_n_mels = n_mels;

    if (!qwenasr_ggml_backend_set_init(device, n_threads, &runtime->backends, error)) {
        qwenasr_audio_encoder_backend_free(runtime);
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
        qwenasr_audio_encoder_backend_free(runtime);
        set_error(error, "failed to initialize GGML backend scheduler");
        return false;
    }

    const int n_weight_tensors = n_layers * 16 + 6 + (runtime->has_prep ? 7 : 0);
    const size_t weight_ctx_size = static_cast<size_t>(n_weight_tensors) * ggml_tensor_overhead() + 1024ull * 1024ull;
    ggml_init_params params;
    params.mem_size = weight_ctx_size;
    params.mem_buffer = nullptr;
    params.no_alloc = true;
    runtime->weight_ctx = ggml_init(params);
    if (!runtime->weight_ctx) {
        qwenasr_audio_encoder_backend_free(runtime);
        set_error(error, "failed to initialize audio encoder weight context");
        return false;
    }

    std::vector<BackendPendingCopy> pending;
    pending.reserve(static_cast<size_t>(n_weight_tensors));
    if (runtime->has_prep) {
        AudioPrepTensorViews prep_views;
        if (!load_audio_prep_tensors(model, n_mels, hidden, &prep_views, error)) {
            qwenasr_audio_encoder_backend_free(runtime);
            return false;
        }
        runtime->prep_channels = prep_views.channels;
        runtime->prep = backend_audio_prep_tensors_from_views(runtime->weight_ctx, prep_views, &pending);
    }

    runtime->layers.reserve(static_cast<size_t>(n_layers));
    for (int layer = 0; layer < n_layers; ++layer) {
        AudioLayerTensors views;
        if (!load_audio_layer_tensors(model, layer, hidden, &views, error)) {
            qwenasr_audio_encoder_backend_free(runtime);
            return false;
        }
        runtime->layers.push_back(backend_audio_layer_tensors_from_views(runtime->weight_ctx, views, &pending));
    }

    AudioProjectorTensors projector_views;
    if (!load_audio_projector_tensors(model, hidden, output_dim, &projector_views, error)) {
        qwenasr_audio_encoder_backend_free(runtime);
        return false;
    }
    runtime->projector = backend_audio_projector_tensors_from_views(runtime->weight_ctx, projector_views, &pending);

    runtime->weight_buffer = ggml_backend_alloc_ctx_tensors(runtime->weight_ctx, runtime->backends.primary);
    if (!runtime->weight_buffer) {
        qwenasr_audio_encoder_backend_free(runtime);
        set_error(error, "failed to allocate audio encoder backend weight buffer");
        return false;
    }
    ggml_backend_buffer_set_usage(runtime->weight_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    for (const BackendPendingCopy & copy : pending) {
        ggml_backend_tensor_set(copy.tensor, copy.data, 0, copy.nbytes);
    }

    *out = runtime;
    return true;
}

bool qwenasr_audio_encoder_backend_forward(
    QwenAsrAudioEncoderBackend * backend,
    const QwenAsrAudioPrepOutput & prep,
    QwenAsrAudioEncoderOutput * out,
    std::string * error) {
    if (!backend || !out) {
        set_error(error, "qwenasr_audio_encoder_backend_forward: backend or out is null");
        return false;
    }
    if (prep.tokens <= 0 || prep.hidden != backend->hidden ||
        prep.values.size() != static_cast<size_t>(prep.tokens) * prep.hidden) {
        set_error(error, "invalid audio prep input for backend encoder");
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
        set_error(error, "failed to initialize audio encoder graph context");
        return false;
    }

    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, backend->hidden, prep.tokens);
    ggml_set_name(x, "audio_encoder_input");
    ggml_set_input(x);

    ggml_tensor * y = x;
    for (int layer = 0; layer < backend->n_layers; ++layer) {
        y = backend_audio_layer_forward_ggml(ctx, backend->layers[static_cast<size_t>(layer)], y, backend->hidden, backend->n_heads);
    }
    y = backend_audio_projector_forward_ggml(ctx, backend->projector, y);
    ggml_set_name(y, "audio_encoder_output");
    ggml_set_output(y);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, max_nodes, false);
    ggml_build_forward_expand(graph, y);
    if (!ggml_backend_sched_alloc_graph(backend->sched, graph)) {
        ggml_backend_sched_reset(backend->sched);
        ggml_free(ctx);
        set_error(error, "GGML backend audio encoder graph allocation failed");
        return false;
    }

    ggml_backend_tensor_set(x, prep.values.data(), 0, prep.values.size() * sizeof(float));
    const ggml_status status = ggml_backend_sched_graph_compute(backend->sched, graph);
    if (status != GGML_STATUS_SUCCESS) {
        ggml_backend_sched_reset(backend->sched);
        ggml_free(ctx);
        set_error(error, "GGML backend audio encoder graph compute failed");
        return false;
    }

    QwenAsrAudioEncoderOutput result;
    result.tokens = prep.tokens;
    result.hidden = backend->output_dim;
    result.attention_segments = prep.attention_segments;
    result.values.assign(static_cast<size_t>(prep.tokens) * backend->output_dim, 0.0f);
    ggml_backend_tensor_get(y, result.values.data(), 0, result.values.size() * sizeof(float));

    ggml_backend_sched_reset(backend->sched);
    ggml_free(ctx);
    *out = std::move(result);
    return true;
}

bool qwenasr_audio_encoder_backend_forward_features(
    QwenAsrAudioEncoderBackend * backend,
    const QwenAsrFeatures & features,
    QwenAsrAudioEncoderOutput * out,
    std::string * error) {
    if (!backend || !out) {
        set_error(error, "qwenasr_audio_encoder_backend_forward_features: backend or out is null");
        return false;
    }
    if (!backend->has_prep) {
        set_error(error, "audio encoder backend was not initialized with prep tensors");
        return false;
    }
    if (!qwenasr_ggml_combined_audio_enabled()) {
        set_error(error, "audio encoder combined fast path is disabled");
        return false;
    }
    if (!validate_features(features, "qwenasr_audio_encoder_backend_forward_features", error)) {
        return false;
    }
    if (features.n_mels != backend->prep_n_mels) {
        set_error(error, "audio encoder backend mel count mismatch");
        return false;
    }

    const QwenAsrAudioGeometry geometry = qwenasr_audio_geometry(features.n_frames);
    const int chunks = geometry.n_chunks;
    const int frames_in = geometry.max_chunk_input_len;
    const int frames_out = qwenasr_audio_output_length(frames_in);
    if (chunks <= 0 || frames_in <= 0 || frames_out <= 0) {
        set_error(error, "audio encoder combined output geometry is empty");
        return false;
    }
    if (geometry.audio_tokens != chunks * frames_out) {
        set_error(error, "audio encoder combined fast path requires full audio chunks");
        return false;
    }

    std::vector<float> input_values = build_chunked_feature_input(features, geometry);
    std::vector<float> positional = build_audio_positional_values(frames_out, backend->hidden);

    constexpr size_t max_nodes = 8192;
    const size_t graph_ctx_size =
        ggml_tensor_overhead() * max_nodes + ggml_graph_overhead_custom(max_nodes, false);
    ggml_init_params params;
    params.mem_size = graph_ctx_size;
    params.mem_buffer = nullptr;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        set_error(error, "failed to initialize combined audio encoder graph context");
        return false;
    }

    ggml_tensor * x = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, frames_in, features.n_mels, 1, chunks);
    ggml_set_name(x, "audio_encoder_features_input");
    ggml_set_input(x);

    ggml_tensor * pos = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, backend->hidden, frames_out, 1);
    ggml_set_name(pos, "audio_encoder_positional");
    ggml_set_input(pos);

    ggml_tensor * y = backend_audio_prep_forward_ggml(
        ctx,
        backend->prep,
        x,
        pos,
        backend->prep_channels,
        backend->hidden,
        features.n_mels,
        frames_out,
        chunks);
    for (int layer = 0; layer < backend->n_layers; ++layer) {
        y = backend_audio_layer_forward_ggml(ctx, backend->layers[static_cast<size_t>(layer)], y, backend->hidden, backend->n_heads);
    }
    y = backend_audio_projector_forward_ggml(ctx, backend->projector, y);
    ggml_set_name(y, "audio_encoder_combined_output");
    ggml_set_output(y);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, max_nodes, false);
    ggml_build_forward_expand(graph, y);
    if (!ggml_backend_sched_alloc_graph(backend->sched, graph)) {
        ggml_backend_sched_reset(backend->sched);
        ggml_free(ctx);
        set_error(error, "GGML backend combined audio encoder graph allocation failed");
        return false;
    }

    ggml_backend_tensor_set(x, input_values.data(), 0, input_values.size() * sizeof(float));
    ggml_backend_tensor_set(pos, positional.data(), 0, positional.size() * sizeof(float));
    const ggml_status status = ggml_backend_sched_graph_compute(backend->sched, graph);
    if (status != GGML_STATUS_SUCCESS) {
        ggml_backend_sched_reset(backend->sched);
        ggml_free(ctx);
        set_error(error, "GGML backend combined audio encoder graph compute failed");
        return false;
    }

    QwenAsrAudioEncoderOutput result;
    result.tokens = geometry.audio_tokens;
    result.hidden = backend->output_dim;
    result.attention_segments = geometry.attention_segments;
    result.values.assign(static_cast<size_t>(geometry.audio_tokens) * backend->output_dim, 0.0f);
    ggml_backend_tensor_get(y, result.values.data(), 0, result.values.size() * sizeof(float));

    ggml_backend_sched_reset(backend->sched);
    ggml_free(ctx);
    *out = std::move(result);
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
