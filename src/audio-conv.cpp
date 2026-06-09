#include "audio-conv.h"

#include <algorithm>
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

bool qwenasr_audio_conv0_forward(
    const QwenAsrGgufModel & model,
    const QwenAsrFeatures & features,
    QwenAsrConv2dOutput * out,
    std::string * error) {
    if (!out) {
        set_error(error, "qwenasr_audio_conv0_forward: out is null");
        return false;
    }
    if (features.n_mels <= 0 || features.n_frames <= 0) {
        set_error(error, "qwenasr_audio_conv0_forward: features are empty");
        return false;
    }
    if (static_cast<size_t>(features.n_mels * features.n_frames) != features.values.size()) {
        set_error(error, "qwenasr_audio_conv0_forward: feature size mismatch");
        return false;
    }

    QwenAsrGgufTensorView weight;
    QwenAsrGgufTensorView bias;
    if (!require_f32_tensor(model, "audio.conv.0.weight", &weight, error) ||
        !require_f32_tensor(model, "audio.conv.0.bias", &bias, error)) {
        return false;
    }
    if (!shape_eq(weight, { 3, 3, 1, 480 }) || !shape_eq(bias, { 480 })) {
        set_error(error, "audio.conv.0 tensor shape mismatch");
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
