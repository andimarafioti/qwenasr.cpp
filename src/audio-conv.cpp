#include "audio-conv.h"

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
#include <cstdint>
#include <cmath>
#include <cstring>
#include <initializer_list>
#include <string>
#include <utility>
#include <vector>

static void set_error(std::string * error, const std::string & message) {
    if (error) {
        *error = message;
    }
}

static bool require_f32_tensor(
    const QwenAsrGgufModel & model,
    const char * name,
    QwenAsrGgufTensorView * out,
    std::string * error) {
    if (!qwenasr_gguf_model_tensor_by_name(model, name, out, error)) {
        return false;
    }
    if (out->type != GGML_TYPE_F32) {
        set_error(error, std::string("tensor is not f32: ") + name);
        return false;
    }
    return true;
}

static bool shape_eq(const QwenAsrGgufTensorView & tensor, std::initializer_list<int64_t> shape) {
    return tensor.ne == std::vector<int64_t>(shape);
}

static float read_f32(const void * data, size_t index) {
    float value = 0.0f;
    std::memcpy(&value, static_cast<const float *>(data) + index, sizeof(float));
    return value;
}

static float gelu(float x) {
    return 0.5f * x * (1.0f + std::erf(x * 0.70710678118654752440f));
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

static bool load_conv0_tensors(
    const QwenAsrGgufModel & model,
    QwenAsrGgufTensorView * weight,
    QwenAsrGgufTensorView * bias,
    std::string * error) {
    if (!require_f32_tensor(model, "audio.conv.0.weight", weight, error) ||
        !require_f32_tensor(model, "audio.conv.0.bias", bias, error)) {
        return false;
    }
    if (!shape_eq(*weight, { 3, 3, 1, 480 }) || !shape_eq(*bias, { 480 })) {
        set_error(error, "audio.conv.0 tensor shape mismatch");
        return false;
    }
    return true;
}

bool qwenasr_audio_conv0_forward(
    const QwenAsrGgufModel & model,
    const QwenAsrFeatures & features,
    QwenAsrConv2dOutput * out,
    std::string * error) {
    if (!out) {
        set_error(error, "qwenasr_audio_conv0_forward: out is null");
        return false;
    }
    if (!validate_features(features, "qwenasr_audio_conv0_forward", error)) {
        return false;
    }

    QwenAsrGgufTensorView weight;
    QwenAsrGgufTensorView bias;
    if (!load_conv0_tensors(model, &weight, &bias, error)) {
        return false;
    }

    const QwenAsrAudioGeometry geometry = qwenasr_audio_geometry(features.n_frames);
    const int chunks = geometry.n_chunks;
    const int channels = 480;
    const int freq_out = conv_stride2_pad1_len(features.n_mels);
    const int frames_out = conv_stride2_pad1_len(geometry.max_chunk_input_len);
    if (chunks <= 0 || freq_out <= 0 || frames_out <= 0) {
        set_error(error, "audio.conv.0 output geometry is empty");
        return false;
    }

    QwenAsrConv2dOutput result;
    result.chunks = chunks;
    result.channels = channels;
    result.freq = freq_out;
    result.frames = frames_out;
    result.chunk_input_lengths = geometry.chunk_input_lengths;
    result.chunk_output_lengths.reserve(result.chunk_input_lengths.size());
    for (int len : result.chunk_input_lengths) {
        result.chunk_output_lengths.push_back(conv_stride2_pad1_len(len));
    }
    result.values.assign(
        static_cast<size_t>(chunks) * channels * freq_out * frames_out,
        0.0f);

    auto input_at = [&](int chunk, int mel, int frame) -> float {
        if (mel < 0 || mel >= features.n_mels || frame < 0 || frame >= geometry.max_chunk_input_len) {
            return 0.0f;
        }
        const int chunk_len = result.chunk_input_lengths[static_cast<size_t>(chunk)];
        if (frame >= chunk_len) {
            return 0.0f;
        }
        const int src_frame = chunk * geometry.chunk_window + frame;
        if (src_frame < 0 || src_frame >= features.n_frames) {
            return 0.0f;
        }
        return features.values[static_cast<size_t>(mel) * features.n_frames + src_frame];
    };

    auto out_index = [&](int chunk, int channel, int freq, int frame) -> size_t {
        return (((static_cast<size_t>(chunk) * channels + channel) * freq_out + freq) * frames_out + frame);
    };

    for (int chunk = 0; chunk < chunks; ++chunk) {
        for (int oc = 0; oc < channels; ++oc) {
            const float b = read_f32(bias.data, static_cast<size_t>(oc));
            for (int of = 0; of < freq_out; ++of) {
                for (int ot = 0; ot < frames_out; ++ot) {
                    float sum = b;
                    for (int kh = 0; kh < 3; ++kh) {
                        const int in_f = of * 2 + kh - 1;
                        for (int kw = 0; kw < 3; ++kw) {
                            const int in_t = ot * 2 + kw - 1;
                            const size_t widx =
                                static_cast<size_t>(kw) +
                                3u * (static_cast<size_t>(kh) + 3u * static_cast<size_t>(oc));
                            sum += input_at(chunk, in_f, in_t) * read_f32(weight.data, widx);
                        }
                    }
                    result.values[out_index(chunk, oc, of, ot)] = gelu(sum);
                }
            }
        }
    }

    *out = std::move(result);
    return true;
}

bool qwenasr_audio_conv0_forward_ggml(
    const QwenAsrGgufModel & model,
    const QwenAsrFeatures & features,
    int n_threads,
    QwenAsrConv2dOutput * out,
    std::string * error) {
    if (!out) {
        set_error(error, "qwenasr_audio_conv0_forward_ggml: out is null");
        return false;
    }
    if (!validate_features(features, "qwenasr_audio_conv0_forward_ggml", error)) {
        return false;
    }
    if (n_threads <= 0) {
        n_threads = 1;
    }

    QwenAsrGgufTensorView weight;
    QwenAsrGgufTensorView bias;
    if (!load_conv0_tensors(model, &weight, &bias, error)) {
        return false;
    }

    const QwenAsrAudioGeometry geometry = qwenasr_audio_geometry(features.n_frames);
    const int chunks = geometry.n_chunks;
    const int channels = 480;
    const int freq_out = conv_stride2_pad1_len(features.n_mels);
    const int frames_out = conv_stride2_pad1_len(geometry.max_chunk_input_len);
    if (chunks <= 0 || freq_out <= 0 || frames_out <= 0) {
        set_error(error, "audio.conv.0 output geometry is empty");
        return false;
    }

    QwenAsrConv2dOutput result;
    result.chunks = chunks;
    result.channels = channels;
    result.freq = freq_out;
    result.frames = frames_out;
    result.chunk_input_lengths = geometry.chunk_input_lengths;
    result.chunk_output_lengths.reserve(result.chunk_input_lengths.size());
    for (int len : result.chunk_input_lengths) {
        result.chunk_output_lengths.push_back(conv_stride2_pad1_len(len));
    }
    result.values.assign(
        static_cast<size_t>(chunks) * channels * freq_out * frames_out,
        0.0f);

    const int frames_in = geometry.max_chunk_input_len;
    std::vector<float> input_values = build_chunked_feature_input(features, geometry);

    const size_t ctx_size = 768ull * 1024ull * 1024ull;
    std::vector<uint8_t> ctx_buffer(ctx_size);
    ggml_init_params params;
    params.mem_size = ctx_buffer.size();
    params.mem_buffer = ctx_buffer.data();
    params.no_alloc = false;
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        set_error(error, "failed to initialize GGML context for audio.conv.0");
        return false;
    }

    ggml_tensor * w = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 3, 3, 1, channels);
    ggml_tensor * b = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, 1, channels, 1);
    ggml_tensor * x = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, frames_in, features.n_mels, 1, chunks);
    std::memcpy(w->data, weight.data, ggml_nbytes(w));
    std::memcpy(b->data, bias.data, ggml_nbytes(b));
    std::memcpy(x->data, input_values.data(), ggml_nbytes(x));

    ggml_tensor * y = ggml_conv_2d(ctx, w, x, 2, 2, 1, 1, 1, 1);
    y = ggml_add(ctx, y, ggml_repeat(ctx, b, y));
    y = ggml_gelu_erf(ctx, y);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, y);
    const ggml_status status = ggml_graph_compute_with_ctx(ctx, graph, n_threads);
    if (status != GGML_STATUS_SUCCESS) {
        ggml_free(ctx);
        set_error(error, "GGML audio.conv.0 graph compute failed");
        return false;
    }

    const float * ggml_out = ggml_get_data_f32(y);
    for (int chunk = 0; chunk < chunks; ++chunk) {
        for (int channel = 0; channel < channels; ++channel) {
            for (int freq = 0; freq < freq_out; ++freq) {
                for (int frame = 0; frame < frames_out; ++frame) {
                    const size_t src =
                        static_cast<size_t>(frame) +
                        static_cast<size_t>(frames_out) *
                            (static_cast<size_t>(freq) +
                             static_cast<size_t>(freq_out) *
                                 (static_cast<size_t>(channel) + static_cast<size_t>(channels) * chunk));
                    const size_t dst =
                        (((static_cast<size_t>(chunk) * channels + channel) * freq_out + freq) * frames_out + frame);
                    result.values[dst] = ggml_out[src];
                }
            }
        }
    }

    ggml_free(ctx);
    *out = std::move(result);
    return true;
}

bool qwenasr_audio_cnn_forward_ggml(
    const QwenAsrGgufModel & model,
    const QwenAsrFeatures & features,
    int n_threads,
    QwenAsrAudioCnnOutput * out,
    std::string * error) {
    if (!out) {
        set_error(error, "qwenasr_audio_cnn_forward_ggml: out is null");
        return false;
    }
    if (!validate_features(features, "qwenasr_audio_cnn_forward_ggml", error)) {
        return false;
    }
    if (n_threads <= 0) {
        n_threads = 1;
    }

    QwenAsrGgufTensorView w0;
    QwenAsrGgufTensorView b0;
    QwenAsrGgufTensorView w1;
    QwenAsrGgufTensorView b1;
    QwenAsrGgufTensorView w2;
    QwenAsrGgufTensorView b2;
    QwenAsrGgufTensorView wout;
    if (!require_f32_tensor(model, "audio.conv.0.weight", &w0, error) ||
        !require_f32_tensor(model, "audio.conv.0.bias", &b0, error) ||
        !require_f32_tensor(model, "audio.conv.1.weight", &w1, error) ||
        !require_f32_tensor(model, "audio.conv.1.bias", &b1, error) ||
        !require_f32_tensor(model, "audio.conv.2.weight", &w2, error) ||
        !require_f32_tensor(model, "audio.conv.2.bias", &b2, error) ||
        !require_f32_tensor(model, "audio.conv_out.weight", &wout, error)) {
        return false;
    }

    const int channels = static_cast<int>(w0.ne[3]);
    if (channels <= 0 ||
        !shape_eq(w0, { 3, 3, 1, channels }) ||
        !shape_eq(b0, { channels }) ||
        !shape_eq(w1, { 3, 3, channels, channels }) ||
        !shape_eq(b1, { channels }) ||
        !shape_eq(w2, { 3, 3, channels, channels }) ||
        !shape_eq(b2, { channels })) {
        set_error(error, "audio CNN tensor shape mismatch");
        return false;
    }

    const QwenAsrAudioGeometry geometry = qwenasr_audio_geometry(features.n_frames);
    const int chunks = geometry.n_chunks;
    const int frames_in = geometry.max_chunk_input_len;
    const int freq_out = conv_stride2_pad1_len(conv_stride2_pad1_len(conv_stride2_pad1_len(features.n_mels)));
    const int frames_out = qwenasr_audio_output_length(frames_in);
    const int conv_out_width = channels * freq_out;
    if (chunks <= 0 || frames_in <= 0 || freq_out <= 0 || frames_out <= 0) {
        set_error(error, "audio CNN output geometry is empty");
        return false;
    }
    if (wout.ne.size() != 2 || wout.ne[0] != conv_out_width || wout.ne[1] <= 0) {
        set_error(error, "audio.conv_out tensor shape mismatch");
        return false;
    }
    const int hidden = static_cast<int>(wout.ne[1]);

    QwenAsrAudioCnnOutput result;
    result.chunks = chunks;
    result.frames = frames_out;
    result.hidden = hidden;
    result.chunk_input_lengths = geometry.chunk_input_lengths;
    result.chunk_output_lengths = geometry.chunk_output_lengths;
    result.values.assign(
        static_cast<size_t>(chunks) * frames_out * hidden,
        0.0f);

    std::vector<float> input_values = build_chunked_feature_input(features, geometry);

    const size_t ctx_size = 1536ull * 1024ull * 1024ull;
    std::vector<uint8_t> ctx_buffer(ctx_size);
    ggml_init_params params;
    params.mem_size = ctx_buffer.size();
    params.mem_buffer = ctx_buffer.data();
    params.no_alloc = false;
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        set_error(error, "failed to initialize GGML context for audio CNN");
        return false;
    }

    ggml_tensor * tw0 = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 3, 3, 1, channels);
    ggml_tensor * tb0 = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, 1, channels, 1);
    ggml_tensor * tw1 = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 3, 3, channels, channels);
    ggml_tensor * tb1 = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, 1, channels, 1);
    ggml_tensor * tw2 = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 3, 3, channels, channels);
    ggml_tensor * tb2 = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, 1, channels, 1);
    ggml_tensor * tout = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, conv_out_width, hidden);
    ggml_tensor * x = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, frames_in, features.n_mels, 1, chunks);
    std::memcpy(tw0->data, w0.data, ggml_nbytes(tw0));
    std::memcpy(tb0->data, b0.data, ggml_nbytes(tb0));
    std::memcpy(tw1->data, w1.data, ggml_nbytes(tw1));
    std::memcpy(tb1->data, b1.data, ggml_nbytes(tb1));
    std::memcpy(tw2->data, w2.data, ggml_nbytes(tw2));
    std::memcpy(tb2->data, b2.data, ggml_nbytes(tb2));
    std::memcpy(tout->data, wout.data, ggml_nbytes(tout));
    std::memcpy(x->data, input_values.data(), ggml_nbytes(x));

    auto conv_gelu = [&](ggml_tensor * input, ggml_tensor * weight, ggml_tensor * bias) -> ggml_tensor * {
        ggml_tensor * y = ggml_conv_2d(ctx, weight, input, 2, 2, 1, 1, 1, 1);
        y = ggml_add(ctx, y, ggml_repeat(ctx, bias, y));
        return ggml_gelu_erf(ctx, y);
    };

    ggml_tensor * y = conv_gelu(x, tw0, tb0);
    y = conv_gelu(y, tw1, tb1);
    y = conv_gelu(y, tw2, tb2);
    y = ggml_cont(ctx, ggml_permute(ctx, y, 2, 0, 1, 3));
    y = ggml_reshape_2d(ctx, y, conv_out_width, frames_out * chunks);
    y = ggml_mul_mat(ctx, tout, y);
    y = ggml_reshape_3d(ctx, y, hidden, frames_out, chunks);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, y);
    const ggml_status status = ggml_graph_compute_with_ctx(ctx, graph, n_threads);
    if (status != GGML_STATUS_SUCCESS) {
        ggml_free(ctx);
        set_error(error, "GGML audio CNN graph compute failed");
        return false;
    }

    std::memcpy(result.values.data(), ggml_get_data_f32(y), result.values.size() * sizeof(float));
    ggml_free(ctx);
    *out = std::move(result);
    return true;
}

static bool audio_prep_from_cnn(
    int feature_frames,
    const QwenAsrAudioCnnOutput & cnn,
    QwenAsrAudioPrepOutput * out,
    std::string * error) {
    if (!out) {
        set_error(error, "audio_prep_from_cnn: out is null");
        return false;
    }
    if (cnn.hidden <= 0 || (cnn.hidden % 2) != 0) {
        set_error(error, "audio prep hidden size must be positive and even");
        return false;
    }

    QwenAsrAudioPrepOutput result;
    result.hidden = cnn.hidden;
    result.tokens = 0;
    for (int len : cnn.chunk_output_lengths) {
        result.tokens += len;
    }
    result.values.assign(static_cast<size_t>(result.tokens) * result.hidden, 0.0f);
    result.attention_segments = qwenasr_audio_geometry(feature_frames).attention_segments;

    const int half = result.hidden / 2;
    const double log_timescale_increment = std::log(10000.0) / static_cast<double>(half - 1);
    std::vector<float> positional(static_cast<size_t>(cnn.frames) * result.hidden, 0.0f);
    for (int frame = 0; frame < cnn.frames; ++frame) {
        for (int i = 0; i < half; ++i) {
            const double inv_timescale = std::exp(-log_timescale_increment * static_cast<double>(i));
            const double scaled_time = static_cast<double>(frame) * inv_timescale;
            positional[static_cast<size_t>(frame) * result.hidden + i] = static_cast<float>(std::sin(scaled_time));
            positional[static_cast<size_t>(frame) * result.hidden + half + i] = static_cast<float>(std::cos(scaled_time));
        }
    }

    size_t token = 0;
    for (int chunk = 0; chunk < cnn.chunks; ++chunk) {
        const int valid_frames = cnn.chunk_output_lengths[static_cast<size_t>(chunk)];
        for (int frame = 0; frame < valid_frames; ++frame) {
            const size_t src =
                (static_cast<size_t>(chunk) * cnn.frames + frame) * cnn.hidden;
            const size_t pos = static_cast<size_t>(frame) * cnn.hidden;
            const size_t dst = token * result.hidden;
            for (int hidden = 0; hidden < result.hidden; ++hidden) {
                result.values[dst + hidden] = cnn.values[src + hidden] + positional[pos + hidden];
            }
            ++token;
        }
    }
    if (token != static_cast<size_t>(result.tokens)) {
        set_error(error, "audio prep token count mismatch");
        return false;
    }

    *out = std::move(result);
    return true;
}

bool qwenasr_audio_prep_forward_ggml(
    const QwenAsrGgufModel & model,
    const QwenAsrFeatures & features,
    int n_threads,
    QwenAsrAudioPrepOutput * out,
    std::string * error) {
    if (!out) {
        set_error(error, "qwenasr_audio_prep_forward_ggml: out is null");
        return false;
    }

    QwenAsrAudioCnnOutput cnn;
    if (!qwenasr_audio_cnn_forward_ggml(model, features, n_threads, &cnn, error)) {
        return false;
    }
    return audio_prep_from_cnn(features.n_frames, cnn, out, error);
}

struct PrepBackendPendingCopy {
    ggml_tensor * tensor = nullptr;
    const void * data = nullptr;
    size_t nbytes = 0;
};

struct QwenAsrAudioPrepBackend {
    ggml_backend_t backend = nullptr;
    ggml_backend_sched_t sched = nullptr;
    ggml_context * weight_ctx = nullptr;
    ggml_backend_buffer_t weight_buffer = nullptr;
    int n_threads = 1;
    int n_mels = 0;
    int channels = 0;
    int hidden = 0;
    ggml_tensor * w0 = nullptr;
    ggml_tensor * b0 = nullptr;
    ggml_tensor * w1 = nullptr;
    ggml_tensor * b1 = nullptr;
    ggml_tensor * w2 = nullptr;
    ggml_tensor * b2 = nullptr;
    ggml_tensor * wout = nullptr;
};

static ggml_tensor * new_prep_backend_weight_from_view(
    ggml_context * ctx,
    const QwenAsrGgufTensorView & view,
    std::vector<PrepBackendPendingCopy> * pending) {
    ggml_tensor * tensor = ggml_new_tensor(
        ctx,
        view.type,
        static_cast<int>(view.ne.size()),
        view.ne.data());
    ggml_set_name(tensor, view.name.c_str());
    pending->push_back(PrepBackendPendingCopy { tensor, view.data, ggml_nbytes(tensor) });
    return tensor;
}

static ggml_tensor * new_prep_backend_conv_bias_from_view(
    ggml_context * ctx,
    const QwenAsrGgufTensorView & view,
    int channels,
    std::vector<PrepBackendPendingCopy> * pending) {
    ggml_tensor * tensor = ggml_new_tensor_4d(ctx, view.type, 1, 1, channels, 1);
    ggml_set_name(tensor, view.name.c_str());
    pending->push_back(PrepBackendPendingCopy { tensor, view.data, ggml_nbytes(tensor) });
    return tensor;
}

void qwenasr_audio_prep_backend_free(QwenAsrAudioPrepBackend * backend) {
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
    if (backend->backend) {
        ggml_backend_free(backend->backend);
        backend->backend = nullptr;
    }
    delete backend;
}

bool qwenasr_audio_prep_backend_init(
    const QwenAsrGgufModel & model,
    int n_threads,
    int n_mels,
    QwenAsrAudioPrepBackend ** out,
    std::string * error) {
    if (!out) {
        set_error(error, "qwenasr_audio_prep_backend_init: out is null");
        return false;
    }
    *out = nullptr;
    if (n_mels <= 0) {
        set_error(error, "invalid audio prep backend configuration");
        return false;
    }
    if (n_threads <= 0) {
        n_threads = 1;
    }

    QwenAsrGgufTensorView w0;
    QwenAsrGgufTensorView b0;
    QwenAsrGgufTensorView w1;
    QwenAsrGgufTensorView b1;
    QwenAsrGgufTensorView w2;
    QwenAsrGgufTensorView b2;
    QwenAsrGgufTensorView wout;
    if (!require_f32_tensor(model, "audio.conv.0.weight", &w0, error) ||
        !require_f32_tensor(model, "audio.conv.0.bias", &b0, error) ||
        !require_f32_tensor(model, "audio.conv.1.weight", &w1, error) ||
        !require_f32_tensor(model, "audio.conv.1.bias", &b1, error) ||
        !require_f32_tensor(model, "audio.conv.2.weight", &w2, error) ||
        !require_f32_tensor(model, "audio.conv.2.bias", &b2, error) ||
        !require_f32_tensor(model, "audio.conv_out.weight", &wout, error)) {
        return false;
    }

    const int channels = static_cast<int>(w0.ne.size() == 4 ? w0.ne[3] : 0);
    const int freq_out = conv_stride2_pad1_len(conv_stride2_pad1_len(conv_stride2_pad1_len(n_mels)));
    const int conv_out_width = channels * freq_out;
    if (channels <= 0 || freq_out <= 0 ||
        !shape_eq(w0, { 3, 3, 1, channels }) ||
        !shape_eq(b0, { channels }) ||
        !shape_eq(w1, { 3, 3, channels, channels }) ||
        !shape_eq(b1, { channels }) ||
        !shape_eq(w2, { 3, 3, channels, channels }) ||
        !shape_eq(b2, { channels }) ||
        wout.ne.size() != 2 || wout.ne[0] != conv_out_width || wout.ne[1] <= 0) {
        set_error(error, "audio prep backend tensor shape mismatch");
        return false;
    }

    QwenAsrAudioPrepBackend * runtime = new QwenAsrAudioPrepBackend();
    runtime->n_threads = n_threads;
    runtime->n_mels = n_mels;
    runtime->channels = channels;
    runtime->hidden = static_cast<int>(wout.ne[1]);

    runtime->backend = ggml_backend_cpu_init();
    if (!runtime->backend) {
        qwenasr_audio_prep_backend_free(runtime);
        set_error(error, "failed to initialize GGML CPU backend");
        return false;
    }
    ggml_backend_cpu_set_n_threads(runtime->backend, n_threads);

    constexpr size_t max_nodes = 1024;
    ggml_backend_t backends[] = { runtime->backend };
    runtime->sched = ggml_backend_sched_new(backends, nullptr, 1, max_nodes, false, true);
    if (!runtime->sched) {
        qwenasr_audio_prep_backend_free(runtime);
        set_error(error, "failed to initialize GGML prep backend scheduler");
        return false;
    }

    const int n_weight_tensors = 7;
    const size_t weight_ctx_size = static_cast<size_t>(n_weight_tensors) * ggml_tensor_overhead() + 1024ull * 1024ull;
    ggml_init_params params;
    params.mem_size = weight_ctx_size;
    params.mem_buffer = nullptr;
    params.no_alloc = true;
    runtime->weight_ctx = ggml_init(params);
    if (!runtime->weight_ctx) {
        qwenasr_audio_prep_backend_free(runtime);
        set_error(error, "failed to initialize audio prep weight context");
        return false;
    }

    std::vector<PrepBackendPendingCopy> pending;
    pending.reserve(n_weight_tensors);
    runtime->w0 = new_prep_backend_weight_from_view(runtime->weight_ctx, w0, &pending);
    runtime->b0 = new_prep_backend_conv_bias_from_view(runtime->weight_ctx, b0, channels, &pending);
    runtime->w1 = new_prep_backend_weight_from_view(runtime->weight_ctx, w1, &pending);
    runtime->b1 = new_prep_backend_conv_bias_from_view(runtime->weight_ctx, b1, channels, &pending);
    runtime->w2 = new_prep_backend_weight_from_view(runtime->weight_ctx, w2, &pending);
    runtime->b2 = new_prep_backend_conv_bias_from_view(runtime->weight_ctx, b2, channels, &pending);
    runtime->wout = new_prep_backend_weight_from_view(runtime->weight_ctx, wout, &pending);

    runtime->weight_buffer = ggml_backend_alloc_ctx_tensors(runtime->weight_ctx, runtime->backend);
    if (!runtime->weight_buffer) {
        qwenasr_audio_prep_backend_free(runtime);
        set_error(error, "failed to allocate audio prep backend weight buffer");
        return false;
    }
    ggml_backend_buffer_set_usage(runtime->weight_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    for (const PrepBackendPendingCopy & copy : pending) {
        ggml_backend_tensor_set(copy.tensor, copy.data, 0, copy.nbytes);
    }

    *out = runtime;
    return true;
}

bool qwenasr_audio_prep_backend_forward(
    QwenAsrAudioPrepBackend * backend,
    const QwenAsrFeatures & features,
    QwenAsrAudioPrepOutput * out,
    std::string * error) {
    if (!backend || !out) {
        set_error(error, "qwenasr_audio_prep_backend_forward: backend or out is null");
        return false;
    }
    if (!validate_features(features, "qwenasr_audio_prep_backend_forward", error)) {
        return false;
    }
    if (features.n_mels != backend->n_mels) {
        set_error(error, "audio prep backend mel count mismatch");
        return false;
    }

    const QwenAsrAudioGeometry geometry = qwenasr_audio_geometry(features.n_frames);
    const int chunks = geometry.n_chunks;
    const int frames_in = geometry.max_chunk_input_len;
    const int freq_out = conv_stride2_pad1_len(conv_stride2_pad1_len(conv_stride2_pad1_len(features.n_mels)));
    const int frames_out = qwenasr_audio_output_length(frames_in);
    const int conv_out_width = backend->channels * freq_out;
    if (chunks <= 0 || frames_in <= 0 || freq_out <= 0 || frames_out <= 0) {
        set_error(error, "audio prep backend output geometry is empty");
        return false;
    }

    std::vector<float> input_values = build_chunked_feature_input(features, geometry);

    constexpr size_t max_nodes = 1024;
    const size_t graph_ctx_size =
        ggml_tensor_overhead() * max_nodes + ggml_graph_overhead_custom(max_nodes, false);
    ggml_init_params params;
    params.mem_size = graph_ctx_size;
    params.mem_buffer = nullptr;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        set_error(error, "failed to initialize audio prep graph context");
        return false;
    }

    ggml_tensor * x = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, frames_in, features.n_mels, 1, chunks);
    ggml_set_name(x, "audio_prep_input");
    ggml_set_input(x);

    auto conv_gelu = [&](ggml_tensor * input, ggml_tensor * weight, ggml_tensor * bias) -> ggml_tensor * {
        ggml_tensor * y = ggml_conv_2d(ctx, weight, input, 2, 2, 1, 1, 1, 1);
        y = ggml_add(ctx, y, ggml_repeat(ctx, bias, y));
        return ggml_gelu_erf(ctx, y);
    };

    ggml_tensor * y = conv_gelu(x, backend->w0, backend->b0);
    y = conv_gelu(y, backend->w1, backend->b1);
    y = conv_gelu(y, backend->w2, backend->b2);
    y = ggml_cont(ctx, ggml_permute(ctx, y, 2, 0, 1, 3));
    y = ggml_reshape_2d(ctx, y, conv_out_width, frames_out * chunks);
    y = ggml_mul_mat(ctx, backend->wout, y);
    y = ggml_reshape_3d(ctx, y, backend->hidden, frames_out, chunks);
    ggml_set_name(y, "audio_prep_cnn_output");
    ggml_set_output(y);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, max_nodes, false);
    ggml_build_forward_expand(graph, y);
    if (!ggml_backend_sched_alloc_graph(backend->sched, graph)) {
        ggml_backend_sched_reset(backend->sched);
        ggml_free(ctx);
        set_error(error, "GGML backend audio prep graph allocation failed");
        return false;
    }
    ggml_backend_tensor_set(x, input_values.data(), 0, input_values.size() * sizeof(float));
    const ggml_status status = ggml_backend_sched_graph_compute(backend->sched, graph);
    if (status != GGML_STATUS_SUCCESS) {
        ggml_backend_sched_reset(backend->sched);
        ggml_free(ctx);
        set_error(error, "GGML backend audio prep graph compute failed");
        return false;
    }

    QwenAsrAudioCnnOutput cnn;
    cnn.chunks = chunks;
    cnn.frames = frames_out;
    cnn.hidden = backend->hidden;
    cnn.chunk_input_lengths = geometry.chunk_input_lengths;
    cnn.chunk_output_lengths = geometry.chunk_output_lengths;
    cnn.values.assign(static_cast<size_t>(chunks) * frames_out * backend->hidden, 0.0f);
    ggml_backend_tensor_get(y, cnn.values.data(), 0, cnn.values.size() * sizeof(float));

    ggml_backend_sched_reset(backend->sched);
    ggml_free(ctx);
    return audio_prep_from_cnn(features.n_frames, cnn, out, error);
}
