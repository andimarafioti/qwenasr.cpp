#include "audio-features.h"
#include "decoder-input.h"
#include "gguf-model.h"
#include "native-model.h"
#include "tokenizer.h"

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
        << "Usage: " << argv0 << " model.gguf audio.wav [--out decoder-input.f32] [--threads N]\n"
        << "       " << argv0 << " model.gguf audio.wav [--language NAME] [--system TEXT]\n"
        << "       " << argv0 << " model.gguf audio.wav [--audio-backend ggml|sched]\n"
        << "\n"
        << "Build native Qwen3-ASR decoder input embeddings from text prompt and audio embeddings.\n";
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
    std::string language;
    std::string system_text;
    std::string audio_backend = "ggml";
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
        if (arg == "--language") {
            if (++i >= argc) {
                std::cerr << "--language requires a value\n";
                return 2;
            }
            language = argv[i];
            continue;
        }
        if (arg == "--system") {
            if (++i >= argc) {
                std::cerr << "--system requires a value\n";
                return 2;
            }
            system_text = argv[i];
            continue;
        }
        if (arg == "--audio-backend") {
            if (++i >= argc) {
                std::cerr << "--audio-backend requires ggml or sched\n";
                return 2;
            }
            audio_backend = argv[i];
            if (audio_backend != "ggml" && audio_backend != "sched") {
                std::cerr << "--audio-backend requires ggml or sched\n";
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

    std::string error;
    QwenAsrNativeConfig cfg;
    if (!qwenasr_load_gguf_metadata(model_path.c_str(), false, &cfg, &error)) {
        std::cerr << error << "\n";
        return 1;
    }

    QwenAsrTokenizer tokenizer;
    if (!qwenasr_tokenizer_load_gguf(model_path.c_str(), &tokenizer, &error)) {
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

    QwenAsrDecoderInputOutput decoder_input;
    double init_ms = 0.0;
    auto start = std::chrono::steady_clock::now();
    auto end = start;
    bool ok = false;
    if (audio_backend == "ggml") {
        ok = qwenasr_decoder_input_forward_ggml(
            model,
            tokenizer,
            features,
            n_threads,
            static_cast<int>(cfg.audio_encoder_layers),
            static_cast<int>(cfg.audio_encoder_attention_heads),
            static_cast<int>(cfg.audio_output_dim),
            static_cast<int>(cfg.token_audio),
            system_text,
            language,
            &decoder_input,
            &error);
        end = std::chrono::steady_clock::now();
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
                &error)) {
            qwenasr_gguf_model_close(&model);
            std::cerr << error << "\n";
            return 1;
        }
        const auto init_end = std::chrono::steady_clock::now();
        init_ms = std::chrono::duration<double, std::milli>(init_end - init_start).count();

        QwenAsrAudioPrepOutput prep;
        QwenAsrAudioEncoderOutput audio;
        start = std::chrono::steady_clock::now();
        ok =
            qwenasr_audio_prep_forward_ggml(model, features, n_threads, &prep, &error) &&
            qwenasr_audio_encoder_backend_forward(encoder_backend, prep, &audio, &error) &&
            qwenasr_decoder_input_from_audio(
                model,
                tokenizer,
                audio,
                static_cast<int>(cfg.token_audio),
                system_text,
                language,
                &decoder_input,
                &error);
        end = std::chrono::steady_clock::now();
        qwenasr_audio_encoder_backend_free(encoder_backend);
    }
    qwenasr_gguf_model_close(&model);
    if (!ok) {
        std::cerr << error << "\n";
        return 1;
    }

    if (!out_path.empty() && !write_raw(out_path, decoder_input.values, &error)) {
        std::cerr << error << "\n";
        return 1;
    }

    double sum = 0.0;
    float min_value = std::numeric_limits<float>::infinity();
    float max_value = -std::numeric_limits<float>::infinity();
    for (float value : decoder_input.values) {
        min_value = std::min(min_value, value);
        max_value = std::max(max_value, value);
        sum += value;
    }

    const double decoder_input_ms = std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << "backend=ggml\n";
    std::cout << "audio_backend=" << audio_backend << "\n";
    std::cout << "threads=" << n_threads << "\n";
    std::cout << "init_ms=" << init_ms << "\n";
    std::cout << "decoder_input_ms=" << decoder_input_ms << "\n";
    std::cout << "sample_rate=" << sample_rate << "\n";
    std::cout << "samples=" << samples.size() << "\n";
    std::cout << "feature_frames=" << features.n_frames << "\n";
    std::cout << "tokens=" << decoder_input.tokens << "\n";
    std::cout << "hidden=" << decoder_input.hidden << "\n";
    std::cout << "audio_tokens=" << decoder_input.audio_tokens << "\n";
    std::cout << "values=" << decoder_input.values.size() << "\n";
    std::cout << "min=" << min_value << "\n";
    std::cout << "max=" << max_value << "\n";
    std::cout << "mean=" << (sum / static_cast<double>(decoder_input.values.size())) << "\n";
    std::cout << "ids=";
    for (size_t i = 0; i < decoder_input.input_ids.size(); ++i) {
        if (i > 0) {
            std::cout << ",";
        }
        std::cout << decoder_input.input_ids[i];
    }
    std::cout << "\n";
    return 0;
}
