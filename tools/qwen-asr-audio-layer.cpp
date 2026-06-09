#include "audio-encoder.h"
#include "audio-features.h"
#include "gguf-model.h"
#include "native-model.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <thread>
#include <vector>

static void usage(const char * argv0) {
    std::cerr
        << "Usage: " << argv0 << " model.gguf audio.wav [--out layer0.f32] [--threads N]\n"
        << "\n"
        << "Run native Qwen3-ASR audio prep plus encoder layer 0.\n";
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

    const unsigned hw_threads = std::thread::hardware_concurrency();
    int n_threads = hw_threads == 0 ? 1 : static_cast<int>(hw_threads);
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
        if (arg == "--threads") {
            if (++i >= argc) {
                std::cerr << "--threads requires a positive integer\n";
                return 2;
            }
            char * end = nullptr;
            const long parsed = std::strtol(argv[i], &end, 10);
            if (!end || *end != '\0' || parsed <= 0) {
                std::cerr << "--threads requires a positive integer\n";
                return 2;
            }
            n_threads = static_cast<int>(parsed);
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

    QwenAsrAudioLayerOutput layer;
    const auto layer_start = std::chrono::steady_clock::now();
    const bool layer_ok = qwenasr_audio_layer0_forward_cpu(
        model,
        features,
        n_threads,
        static_cast<int>(cfg.audio_encoder_attention_heads),
        &layer,
        &error);
    const auto layer_end = std::chrono::steady_clock::now();
    qwenasr_gguf_model_close(&model);
    if (!layer_ok) {
        std::cerr << error << "\n";
        return 1;
    }

    if (!out_path.empty() && !write_raw(out_path, layer.values, &error)) {
        std::cerr << error << "\n";
        return 1;
    }

    double sum = 0.0;
    float min_value = std::numeric_limits<float>::infinity();
    float max_value = -std::numeric_limits<float>::infinity();
    for (float value : layer.values) {
        min_value = std::min(min_value, value);
        max_value = std::max(max_value, value);
        sum += value;
    }

    const double layer_ms =
        std::chrono::duration<double, std::milli>(layer_end - layer_start).count();

    std::cout << "backend=cpu\n";
    std::cout << "threads=" << n_threads << "\n";
    std::cout << "layer_ms=" << layer_ms << "\n";
    std::cout << "sample_rate=" << sample_rate << "\n";
    std::cout << "samples=" << samples.size() << "\n";
    std::cout << "feature_mels=" << features.n_mels << "\n";
    std::cout << "feature_frames=" << features.n_frames << "\n";
    std::cout << "tokens=" << layer.tokens << "\n";
    std::cout << "hidden=" << layer.hidden << "\n";
    std::cout << "heads=" << cfg.audio_encoder_attention_heads << "\n";
    std::cout << "values=" << layer.values.size() << "\n";
    std::cout << "attention_segments=" << layer.attention_segments.size() << "\n";
    std::cout << "min=" << min_value << "\n";
    std::cout << "max=" << max_value << "\n";
    std::cout << "mean=" << (sum / static_cast<double>(layer.values.size())) << "\n";
    return 0;
}
