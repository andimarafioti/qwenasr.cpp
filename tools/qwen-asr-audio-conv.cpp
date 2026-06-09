#include "audio-conv.h"
#include "audio-features.h"
#include "gguf-model.h"
#include "native-model.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

static void usage(const char * argv0) {
    std::cerr
        << "Usage: " << argv0 << " model.gguf audio.wav [--out conv0.f32]\n"
        << "\n"
        << "Run the native mapped-weight Qwen3-ASR first audio Conv2D layer.\n";
}

static bool write_raw(const std::string & path, const std::vector<float> & values, std::string * error) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        if (error) {
            *error = "failed to open output file: " + path;
        }
        return false;
    }
    out.write(reinterpret_cast<const char *>(values.data()), static_cast<std::streamsize>(values.size() * sizeof(float)));
    if (!out) {
        if (error) {
            *error = "failed to write output file: " + path;
        }
        return false;
    }
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        usage(argv[0]);
        return 2;
    }

    std::string model_path;
    std::string audio_path;
    std::string out_path;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            usage(argv[0]);
            return 0;
        }
        if (arg == "--out") {
            if (++i >= argc) {
                std::cerr << "--out requires a path\n";
                return 2;
            }
            out_path = argv[i];
            continue;
        }
        if (model_path.empty()) {
            model_path = arg;
            continue;
        }
        if (audio_path.empty()) {
            audio_path = arg;
            continue;
        }
        std::cerr << "too many positional arguments\n";
        return 2;
    }

    if (model_path.empty() || audio_path.empty()) {
        std::cerr << "model GGUF and audio WAV are required\n";
        return 2;
    }

    std::string error;
    QwenAsrNativeConfig cfg;
    if (!qwenasr_load_gguf_metadata(model_path.c_str(), false, &cfg, &error)) {
        std::cerr << error << "\n";
        return 1;
    }

    std::vector<float> samples;
    int sample_rate = 0;
    if (!qwenasr_read_wav_16k_mono(audio_path.c_str(), &samples, &sample_rate, &error)) {
        std::cerr << error << "\n";
        return 1;
    }
    QwenAsrFeatures features;
    if (!qwenasr_extract_whisper_features(samples, &features, &error)) {
        std::cerr << error << "\n";
        return 1;
    }

    QwenAsrGgufModel model {};
    if (!qwenasr_gguf_model_open(model_path.c_str(), &model, &error)) {
        std::cerr << error << "\n";
        return 1;
    }

    QwenAsrConv2dOutput conv;
    if (!qwenasr_audio_conv0_forward(model, features, &conv, &error)) {
        qwenasr_gguf_model_close(&model);
        std::cerr << error << "\n";
        return 1;
    }
    qwenasr_gguf_model_close(&model);

    if (!out_path.empty() && !write_raw(out_path, conv.values, &error)) {
        std::cerr << error << "\n";
        return 1;
    }

    double sum = 0.0;
    float min_value = std::numeric_limits<float>::infinity();
    float max_value = -std::numeric_limits<float>::infinity();
    for (float value : conv.values) {
        min_value = std::min(min_value, value);
        max_value = std::max(max_value, value);
        sum += value;
    }

    std::cout << "sample_rate=" << sample_rate << "\n";
    std::cout << "samples=" << samples.size() << "\n";
    std::cout << "feature_mels=" << features.n_mels << "\n";
    std::cout << "feature_frames=" << features.n_frames << "\n";
    std::cout << "chunks=" << conv.chunks << "\n";
    std::cout << "channels=" << conv.channels << "\n";
    std::cout << "freq=" << conv.freq << "\n";
    std::cout << "frames=" << conv.frames << "\n";
    std::cout << "values=" << conv.values.size() << "\n";
    std::cout << "min=" << min_value << "\n";
    std::cout << "max=" << max_value << "\n";
    std::cout << "mean=" << (sum / static_cast<double>(conv.values.size())) << "\n";
    for (size_t i = 0; i < conv.chunk_input_lengths.size(); ++i) {
        std::cout << "chunk." << i << ".input=" << conv.chunk_input_lengths[i] << "\n";
        std::cout << "chunk." << i << ".output=" << conv.chunk_output_lengths[i] << "\n";
    }
    return 0;
}
