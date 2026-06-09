#include "audio-features.h"
#include "decoder-input.h"
#include "gguf-model.h"
#include "native-model.h"
#include "text-decoder.h"
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
        << "Usage: " << argv0 << " model.gguf audio.wav [--out text-layer0.f32] [--threads N]\n"
        << "       " << argv0 << " model.gguf audio.wav [--language NAME] [--system TEXT]\n"
        << "       " << argv0 << " model.gguf audio.wav [--audio-backend ggml|sched] [--layer N]\n"
        << "       " << argv0 << " model.gguf audio.wav --prefill [--out next-token-logits.f32]\n"
        << "       " << argv0 << " model.gguf audio.wav --generate N [--kv-cache]\n"
        << "\n"
        << "Run native Qwen3-ASR decoder input assembly, text prefill, or slow greedy generation.\n";
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

static bool write_text(const std::string & path, const std::string & value, std::string * error) {
    std::ofstream out(path);
    if (!out) {
        if (error) {
            *error = "failed to open output file: " + path;
        }
        return false;
    }
    out << value;
    if (!out) {
        if (error) {
            *error = "failed to write output file: " + path;
        }
        return false;
    }
    return true;
}

static std::string json_escape(const std::string & value) {
    std::string out;
    out.reserve(value.size() + 2);
    out.push_back('"');
    for (const unsigned char c : value) {
        switch (c) {
        case '\\': out += "\\\\"; break;
        case '"': out += "\\\""; break;
        case '\n': out += "\\n"; break;
        case '\r': out += "\\r"; break;
        case '\t': out += "\\t"; break;
        default:
            if (c < 0x20) {
                const char hex[] = "0123456789abcdef";
                out += "\\u00";
                out.push_back(hex[c >> 4]);
                out.push_back(hex[c & 0x0F]);
            } else {
                out.push_back(static_cast<char>(c));
            }
            break;
        }
    }
    out.push_back('"');
    return out;
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        usage(argv[0]);
        return 2;
    }

    const unsigned hw_threads = std::thread::hardware_concurrency();
    int n_threads = hw_threads == 0 ? 1 : static_cast<int>(hw_threads);
    int layer = 0;
    std::string model_path;
    std::string audio_path;
    std::string out_path;
    std::string language;
    std::string system_text;
    std::string audio_backend = "sched";
    bool prefill = false;
    bool use_kv_cache = false;
    int generate_tokens = -1;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            usage(argv[0]);
            return 0;
        }
        if (arg == "--prefill") {
            prefill = true;
            continue;
        }
        if (arg == "--kv-cache") {
            use_kv_cache = true;
            continue;
        }
        if (arg == "--generate") {
            if (++i >= argc) {
                std::cerr << "--generate requires a non-negative integer\n";
                return 2;
            }
            char * end = nullptr;
            const long parsed = std::strtol(argv[i], &end, 10);
            if (!end || *end != '\0' || parsed < 0 || parsed > 1024) {
                std::cerr << "--generate requires a non-negative integer up to 1024\n";
                return 2;
            }
            generate_tokens = static_cast<int>(parsed);
            continue;
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
        if (arg == "--layer") {
            if (++i >= argc) {
                std::cerr << "--layer requires a non-negative integer\n";
                return 2;
            }
            char * end = nullptr;
            const long parsed = std::strtol(argv[i], &end, 10);
            if (!end || *end != '\0' || parsed < 0) {
                std::cerr << "--layer requires a non-negative integer\n";
                return 2;
            }
            layer = static_cast<int>(parsed);
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
    if (prefill && generate_tokens >= 0) {
        std::cerr << "--prefill and --generate are mutually exclusive\n";
        return 2;
    }
    if (use_kv_cache && generate_tokens < 0) {
        std::cerr << "--kv-cache requires --generate\n";
        return 2;
    }

    std::string error;
    QwenAsrNativeConfig cfg;
    if (!qwenasr_load_gguf_metadata(model_path.c_str(), false, &cfg, &error)) {
        std::cerr << error << "\n";
        return 1;
    }
    if (!prefill && (layer < 0 || layer >= static_cast<int>(cfg.text_num_hidden_layers))) {
        std::cerr << "--layer is out of range for model\n";
        return 2;
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
    auto decoder_start = std::chrono::steady_clock::now();
    auto decoder_end = decoder_start;
    bool decoder_ok = false;
    if (audio_backend == "ggml") {
        decoder_ok = qwenasr_decoder_input_forward_ggml(
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
        decoder_end = std::chrono::steady_clock::now();
    } else {
        const auto init_start = std::chrono::steady_clock::now();
        QwenAsrAudioPrepBackend * prep_backend = nullptr;
        QwenAsrAudioEncoderBackend * encoder_backend = nullptr;
        if (!qwenasr_audio_prep_backend_init(
                model,
                n_threads,
                static_cast<int>(cfg.audio_num_mel_bins),
                &prep_backend,
                &error)) {
            qwenasr_gguf_model_close(&model);
            std::cerr << error << "\n";
            return 1;
        }
        if (!qwenasr_audio_encoder_backend_init(
                model,
                n_threads,
                static_cast<int>(cfg.audio_encoder_layers),
                static_cast<int>(cfg.audio_encoder_attention_heads),
                static_cast<int>(cfg.audio_d_model),
                static_cast<int>(cfg.audio_output_dim),
                &encoder_backend,
                &error)) {
            qwenasr_audio_prep_backend_free(prep_backend);
            qwenasr_gguf_model_close(&model);
            std::cerr << error << "\n";
            return 1;
        }
        const auto init_end = std::chrono::steady_clock::now();
        init_ms = std::chrono::duration<double, std::milli>(init_end - init_start).count();

        QwenAsrAudioPrepOutput prep;
        QwenAsrAudioEncoderOutput audio;
        decoder_start = std::chrono::steady_clock::now();
        decoder_ok =
            qwenasr_audio_prep_backend_forward(prep_backend, features, &prep, &error) &&
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
        decoder_end = std::chrono::steady_clock::now();
        qwenasr_audio_encoder_backend_free(encoder_backend);
        qwenasr_audio_prep_backend_free(prep_backend);
    }
    if (!decoder_ok) {
        qwenasr_gguf_model_close(&model);
        std::cerr << error << "\n";
        return 1;
    }

    if (generate_tokens >= 0) {
        QwenAsrTextGenerateOutput generated;
        const auto generate_start = std::chrono::steady_clock::now();
        const std::vector<int32_t> stop_ids = {
            static_cast<int32_t>(cfg.token_im_end),
            static_cast<int32_t>(cfg.token_endoftext),
            static_cast<int32_t>(cfg.tokenizer_eos),
        };
        const bool generate_ok = use_kv_cache ?
            qwenasr_text_generate_cached_cpu(
                model,
                decoder_input,
                generate_tokens,
                static_cast<int>(cfg.text_num_hidden_layers),
                static_cast<int>(cfg.text_num_attention_heads),
                static_cast<int>(cfg.text_num_key_value_heads),
                static_cast<int>(cfg.text_head_dim),
                static_cast<int>(cfg.text_intermediate_size),
                static_cast<int>(cfg.text_vocab_size),
                cfg.text_rope_theta,
                cfg.text_rms_norm_eps,
                stop_ids,
                &generated,
                &error) :
            qwenasr_text_generate_greedy_cpu(
                model,
                decoder_input,
                generate_tokens,
                static_cast<int>(cfg.text_num_hidden_layers),
                static_cast<int>(cfg.text_num_attention_heads),
                static_cast<int>(cfg.text_num_key_value_heads),
                static_cast<int>(cfg.text_head_dim),
                static_cast<int>(cfg.text_intermediate_size),
                static_cast<int>(cfg.text_vocab_size),
                cfg.text_rope_theta,
                cfg.text_rms_norm_eps,
                stop_ids,
                &generated,
                &error);
        const auto generate_end = std::chrono::steady_clock::now();
        qwenasr_gguf_model_close(&model);
        if (!generate_ok) {
            std::cerr << error << "\n";
            return 1;
        }

        const std::string generated_text = qwenasr_tokenizer_decode(tokenizer, generated.generated_ids, true);
        if (!out_path.empty() && !write_text(out_path, generated_text, &error)) {
            std::cerr << error << "\n";
            return 1;
        }

        const double decoder_input_ms = std::chrono::duration<double, std::milli>(decoder_end - decoder_start).count();
        const double generate_ms = std::chrono::duration<double, std::milli>(generate_end - generate_start).count();

        std::cout << "backend=cpu\n";
        std::cout << "mode=generate\n";
        std::cout << "decode_backend=" << (use_kv_cache ? "kv-cache" : "recompute") << "\n";
        std::cout << "audio_backend=" << audio_backend << "\n";
        std::cout << "threads=" << n_threads << "\n";
        std::cout << "init_ms=" << init_ms << "\n";
        std::cout << "decoder_input_ms=" << decoder_input_ms << "\n";
        std::cout << "generate_ms=" << generate_ms << "\n";
        std::cout << "sample_rate=" << sample_rate << "\n";
        std::cout << "samples=" << samples.size() << "\n";
        std::cout << "feature_frames=" << features.n_frames << "\n";
        std::cout << "layers=" << cfg.text_num_hidden_layers << "\n";
        std::cout << "prompt_tokens=" << generated.prompt_tokens << "\n";
        std::cout << "total_tokens=" << generated.total_tokens << "\n";
        std::cout << "hidden=" << generated.hidden << "\n";
        std::cout << "vocab=" << generated.vocab << "\n";
        std::cout << "requested_tokens=" << generate_tokens << "\n";
        std::cout << "generated_tokens=" << generated.generated_ids.size() << "\n";
        std::cout << "stopped=" << (generated.stopped ? "true" : "false") << "\n";
        std::cout << "generated_ids=";
        for (size_t i = 0; i < generated.generated_ids.size(); ++i) {
            if (i > 0) {
                std::cout << ",";
            }
            std::cout << generated.generated_ids[i];
        }
        std::cout << "\n";
        std::cout << "text_json=" << json_escape(generated_text) << "\n";
        return 0;
    }

    if (prefill) {
        QwenAsrTextPrefillOutput prefill_out;
        const auto prefill_start = std::chrono::steady_clock::now();
        const bool prefill_ok = qwenasr_text_prefill_forward_cpu(
            model,
            decoder_input,
            static_cast<int>(cfg.text_num_hidden_layers),
            static_cast<int>(cfg.text_num_attention_heads),
            static_cast<int>(cfg.text_num_key_value_heads),
            static_cast<int>(cfg.text_head_dim),
            static_cast<int>(cfg.text_intermediate_size),
            static_cast<int>(cfg.text_vocab_size),
            cfg.text_rope_theta,
            cfg.text_rms_norm_eps,
            &prefill_out,
            &error);
        const auto prefill_end = std::chrono::steady_clock::now();
        qwenasr_gguf_model_close(&model);
        if (!prefill_ok) {
            std::cerr << error << "\n";
            return 1;
        }

        if (!out_path.empty() && !write_raw(out_path, prefill_out.logits, &error)) {
            std::cerr << error << "\n";
            return 1;
        }

        double sum = 0.0;
        float min_value = std::numeric_limits<float>::infinity();
        float max_value = -std::numeric_limits<float>::infinity();
        int top_id = -1;
        float top_value = -std::numeric_limits<float>::infinity();
        for (size_t i = 0; i < prefill_out.logits.size(); ++i) {
            const float value = prefill_out.logits[i];
            min_value = std::min(min_value, value);
            max_value = std::max(max_value, value);
            if (value > top_value) {
                top_value = value;
                top_id = static_cast<int>(i);
            }
            sum += value;
        }

        const double decoder_input_ms = std::chrono::duration<double, std::milli>(decoder_end - decoder_start).count();
        const double prefill_ms = std::chrono::duration<double, std::milli>(prefill_end - prefill_start).count();

        std::cout << "backend=cpu\n";
        std::cout << "mode=prefill\n";
        std::cout << "audio_backend=" << audio_backend << "\n";
        std::cout << "threads=" << n_threads << "\n";
        std::cout << "init_ms=" << init_ms << "\n";
        std::cout << "decoder_input_ms=" << decoder_input_ms << "\n";
        std::cout << "prefill_ms=" << prefill_ms << "\n";
        std::cout << "sample_rate=" << sample_rate << "\n";
        std::cout << "samples=" << samples.size() << "\n";
        std::cout << "feature_frames=" << features.n_frames << "\n";
        std::cout << "layers=" << cfg.text_num_hidden_layers << "\n";
        std::cout << "tokens=" << prefill_out.tokens << "\n";
        std::cout << "hidden=" << prefill_out.hidden << "\n";
        std::cout << "vocab=" << prefill_out.vocab << "\n";
        std::cout << "values=" << prefill_out.logits.size() << "\n";
        std::cout << "top_id=" << top_id << "\n";
        std::cout << "top_logit=" << top_value << "\n";
        std::cout << "min=" << min_value << "\n";
        std::cout << "max=" << max_value << "\n";
        std::cout << "mean=" << (sum / static_cast<double>(prefill_out.logits.size())) << "\n";
        return 0;
    }

    QwenAsrTextLayerOutput text_layer;
    const auto text_start = std::chrono::steady_clock::now();
    const bool text_ok = qwenasr_text_layer_forward_cpu(
        model,
        decoder_input,
        layer,
        static_cast<int>(cfg.text_num_attention_heads),
        static_cast<int>(cfg.text_num_key_value_heads),
        static_cast<int>(cfg.text_head_dim),
        static_cast<int>(cfg.text_intermediate_size),
        cfg.text_rope_theta,
        cfg.text_rms_norm_eps,
        &text_layer,
        &error);
    const auto text_end = std::chrono::steady_clock::now();
    qwenasr_gguf_model_close(&model);
    if (!text_ok) {
        std::cerr << error << "\n";
        return 1;
    }

    if (!out_path.empty() && !write_raw(out_path, text_layer.values, &error)) {
        std::cerr << error << "\n";
        return 1;
    }

    double sum = 0.0;
    float min_value = std::numeric_limits<float>::infinity();
    float max_value = -std::numeric_limits<float>::infinity();
    for (float value : text_layer.values) {
        min_value = std::min(min_value, value);
        max_value = std::max(max_value, value);
        sum += value;
    }

    const double decoder_input_ms = std::chrono::duration<double, std::milli>(decoder_end - decoder_start).count();
    const double text_layer_ms = std::chrono::duration<double, std::milli>(text_end - text_start).count();

    std::cout << "backend=cpu\n";
    std::cout << "audio_backend=" << audio_backend << "\n";
    std::cout << "threads=" << n_threads << "\n";
    std::cout << "init_ms=" << init_ms << "\n";
    std::cout << "decoder_input_ms=" << decoder_input_ms << "\n";
    std::cout << "text_layer_ms=" << text_layer_ms << "\n";
    std::cout << "sample_rate=" << sample_rate << "\n";
    std::cout << "samples=" << samples.size() << "\n";
    std::cout << "feature_frames=" << features.n_frames << "\n";
    std::cout << "layer=" << layer << "\n";
    std::cout << "tokens=" << text_layer.tokens << "\n";
    std::cout << "hidden=" << text_layer.hidden << "\n";
    std::cout << "values=" << text_layer.values.size() << "\n";
    std::cout << "min=" << min_value << "\n";
    std::cout << "max=" << max_value << "\n";
    std::cout << "mean=" << (sum / static_cast<double>(text_layer.values.size())) << "\n";
    return 0;
}
