#include "audio-conv.h"

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
    std::vector<float> input_values(
        static_cast<size_t>(frames_in) * features.n_mels * chunks,
        0.0f);
    for (int chunk = 0; chunk < chunks; ++chunk) {
        const int chunk_len = result.chunk_input_lengths[static_cast<size_t>(chunk)];
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
