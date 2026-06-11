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
        << "Usage: " << argv0 << " model.gguf audio.wav [--out encoder.f32] [--threads N]\n"
        << "       " << argv0 << " model.gguf audio.wav [--backend ggml|sched] [--device auto|cpu|gpu]\n"
        << "\n"
        << "Run native Qwen3-ASR audio CNN, transformer encoder, and projector.\n"
        << "Defaults: --backend sched --device auto.\n";
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
    std::string backend = "sched";
    std::string error;
    QwenAsrGgmlDevice device = qwenasr_ggml_device_auto();
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
        if (arg == "--backend") {
            if (++i >= argc) {
                std::cerr << "--backend requires ggml or sched\n";
                return 2;
            }
            backend = argv[i];
            if (backend != "ggml" && backend != "sched") {
                std::cerr << "--backend requires ggml or sched\n";
                return 2;
            }
            continue;
        }
        if (arg == "--device") {
            if (++i >= argc) {
                std::cerr << "--device requires auto, cpu, or gpu\n";
                return 2;
            }
            if (!qwenasr_ggml_device_from_string(argv[i], &device, &error)) {
                std::cerr << error << "\n";
                return 2;
            }
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

    QwenAsrAudioEncoderOutput encoder;
    double init_ms = 0.0;
    auto encoder_start = std::chrono::steady_clock::now();
    auto encoder_end = encoder_start;
    bool encoder_ok = false;
    std::string audio_path_name = "direct";
    if (backend == "ggml") {
        encoder_ok = qwenasr_audio_encoder_forward_ggml(
            model,
            features,
            n_threads,
            static_cast<int>(cfg.audio_encoder_layers),
            static_cast<int>(cfg.audio_encoder_attention_heads),
            static_cast<int>(cfg.audio_output_dim),
            &encoder,
            &error);
        encoder_end = std::chrono::steady_clock::now();
    } else {
        const auto init_start = std::chrono::steady_clock::now();
        QwenAsrAudioEncoderBackend * encoder_backend = nullptr;
        if (!qwenasr_audio_encoder_backend_init(
                model,
                n_threads,
                static_cast<int>(cfg.audio_encoder_layers),
                static_cast<int>(cfg.audio_encoder_attention_heads),
                static_cast<int>(cfg.audio_d_model),
                static_cast<int>(cfg.audio_output_dim),
                &encoder_backend,
                &error,
                device,
                static_cast<int>(cfg.audio_num_mel_bins))) {
            qwenasr_gguf_model_close(&model);
            std::cerr << error << "\n";
            return 1;
        }
        const auto init_end = std::chrono::steady_clock::now();
        init_ms = std::chrono::duration<double, std::milli>(init_end - init_start).count();

        audio_path_name = "combined";
        encoder_start = std::chrono::steady_clock::now();
        encoder_ok = qwenasr_audio_encoder_backend_forward_features(encoder_backend, features, &encoder, &error);
        encoder_end = std::chrono::steady_clock::now();
        if (!encoder_ok) {
            const auto prep_init_start = std::chrono::steady_clock::now();
            QwenAsrAudioPrepBackend * prep_backend = nullptr;
            if (!qwenasr_audio_prep_backend_init(
                    model,
                    n_threads,
                    static_cast<int>(cfg.audio_num_mel_bins),
                    &prep_backend,
                    &error,
                    device)) {
                qwenasr_audio_encoder_backend_free(encoder_backend);
                qwenasr_gguf_model_close(&model);
                std::cerr << error << "\n";
                return 1;
            }
            const auto prep_init_end = std::chrono::steady_clock::now();
            init_ms += std::chrono::duration<double, std::milli>(prep_init_end - prep_init_start).count();

            QwenAsrAudioPrepOutput prep;
            audio_path_name = "two-stage";
            encoder_start = std::chrono::steady_clock::now();
            encoder_ok =
                qwenasr_audio_prep_backend_forward(prep_backend, features, &prep, &error) &&
                qwenasr_audio_encoder_backend_forward(encoder_backend, prep, &encoder, &error);
            encoder_end = std::chrono::steady_clock::now();
            qwenasr_audio_prep_backend_free(prep_backend);
        }
        qwenasr_audio_encoder_backend_free(encoder_backend);
    }
    qwenasr_gguf_model_close(&model);
    if (!encoder_ok) {
        std::cerr << error << "\n";
        return 1;
    }

    if (!out_path.empty() && !write_raw(out_path, encoder.values, &error)) {
        std::cerr << error << "\n";
        return 1;
    }

    double sum = 0.0;
    float min_value = std::numeric_limits<float>::infinity();
    float max_value = -std::numeric_limits<float>::infinity();
    for (float value : encoder.values) {
        min_value = std::min(min_value, value);
        max_value = std::max(max_value, value);
        sum += value;
    }

    const double encoder_ms =
        std::chrono::duration<double, std::milli>(encoder_end - encoder_start).count();

    std::cout << "backend=" << backend << "\n";
    std::cout << "device=" << qwenasr_ggml_device_name(device) << "\n";
    std::cout << "audio_path=" << audio_path_name << "\n";
    std::cout << "threads=" << n_threads << "\n";
    std::cout << "init_ms=" << init_ms << "\n";
    std::cout << "encoder_ms=" << encoder_ms << "\n";
    std::cout << "sample_rate=" << sample_rate << "\n";
    std::cout << "samples=" << samples.size() << "\n";
    std::cout << "feature_mels=" << features.n_mels << "\n";
    std::cout << "feature_frames=" << features.n_frames << "\n";
    std::cout << "layers=" << cfg.audio_encoder_layers << "\n";
    std::cout << "tokens=" << encoder.tokens << "\n";
    std::cout << "hidden=" << encoder.hidden << "\n";
    std::cout << "heads=" << cfg.audio_encoder_attention_heads << "\n";
    std::cout << "values=" << encoder.values.size() << "\n";
    std::cout << "attention_segments=" << encoder.attention_segments.size() << "\n";
    std::cout << "min=" << min_value << "\n";
    std::cout << "max=" << max_value << "\n";
    std::cout << "mean=" << (sum / static_cast<double>(encoder.values.size())) << "\n";
    return 0;
}
