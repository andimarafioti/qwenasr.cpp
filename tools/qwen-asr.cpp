#include "qwenasr.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

struct CliArgs {
    std::string audio;
    std::string model;
    std::string size                = "0.6B";
    std::string backend             = "torch";
    std::string device              = "auto";
    std::string dtype               = "auto";
    std::string attn_implementation = "auto";
    std::string language;
    std::string context;
    std::string python_path;
    int         max_new_tokens    = 256;
    int         batch_size        = 32;
    int         cuda_graph_stride = 128;
    int         warmup            = 1;
    int         runs              = 5;
    bool        use_cuda_graph    = true;
    bool        local_files_only  = false;
    bool        json              = false;
};

static void usage(const char * argv0) {
    std::cerr
        << "Usage: " << argv0 << " audio.wav [options]\n"
        << "\n"
        << "C++ qwenasr.cpp bridge CLI. It exposes the qwentts.cpp-style C ABI while\n"
        << "delegating inference to the installed qwenasr_cpp Python backend.\n"
        << "\n"
        << "Options:\n"
        << "  --model PATH_OR_ID\n"
        << "  --size 0.6B|1.7B|small|large       default: 0.6B\n"
        << "  --backend auto|torch|transformers|vllm  default: torch\n"
        << "  --device auto|cuda:0|cpu           default: auto\n"
        << "  --dtype auto|bf16|fp16|fp32        default: auto\n"
        << "  --attn-implementation NAME         default: auto\n"
        << "  --language NAME\n"
        << "  --context TEXT\n"
        << "  --max-new-tokens N                 default: 256\n"
        << "  --batch-size N                     default: 32\n"
        << "  --no-cuda-graph\n"
        << "  --cuda-graph-stride N              default: 128\n"
        << "  --local-files-only\n"
        << "  --python-path PATHS                prepend to embedded Python sys.path\n"
        << "  --warmup N                         default: 1\n"
        << "  --runs N                           default: 5\n"
        << "  --json\n";
}

static bool arg_value(int & i, int argc, char ** argv, std::string * out) {
    if (i + 1 >= argc) {
        std::cerr << "missing value for " << argv[i] << "\n";
        return false;
    }
    *out = argv[++i];
    return true;
}

static bool arg_value_int(int & i, int argc, char ** argv, int * out) {
    std::string value;
    if (!arg_value(i, argc, argv, &value)) {
        return false;
    }
    try {
        *out = std::stoi(value);
        return true;
    } catch (...) {
        std::cerr << "invalid integer for " << argv[i - 1] << ": " << value << "\n";
        return false;
    }
}

static bool parse_args(int argc, char ** argv, CliArgs * args) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "-h" || a == "--help") {
            usage(argv[0]);
            std::exit(0);
        } else if (a == "--model") {
            if (!arg_value(i, argc, argv, &args->model)) return false;
        } else if (a == "--size") {
            if (!arg_value(i, argc, argv, &args->size)) return false;
        } else if (a == "--backend") {
            if (!arg_value(i, argc, argv, &args->backend)) return false;
        } else if (a == "--device") {
            if (!arg_value(i, argc, argv, &args->device)) return false;
        } else if (a == "--dtype") {
            if (!arg_value(i, argc, argv, &args->dtype)) return false;
        } else if (a == "--attn-implementation") {
            if (!arg_value(i, argc, argv, &args->attn_implementation)) return false;
        } else if (a == "--language") {
            if (!arg_value(i, argc, argv, &args->language)) return false;
        } else if (a == "--context") {
            if (!arg_value(i, argc, argv, &args->context)) return false;
        } else if (a == "--python-path") {
            if (!arg_value(i, argc, argv, &args->python_path)) return false;
        } else if (a == "--max-new-tokens") {
            if (!arg_value_int(i, argc, argv, &args->max_new_tokens)) return false;
        } else if (a == "--batch-size") {
            if (!arg_value_int(i, argc, argv, &args->batch_size)) return false;
        } else if (a == "--cuda-graph-stride") {
            if (!arg_value_int(i, argc, argv, &args->cuda_graph_stride)) return false;
        } else if (a == "--warmup") {
            if (!arg_value_int(i, argc, argv, &args->warmup)) return false;
        } else if (a == "--runs") {
            if (!arg_value_int(i, argc, argv, &args->runs)) return false;
        } else if (a == "--no-cuda-graph") {
            args->use_cuda_graph = false;
        } else if (a == "--local-files-only") {
            args->local_files_only = true;
        } else if (a == "--json") {
            args->json = true;
        } else if (!a.empty() && a[0] == '-') {
            std::cerr << "unknown option: " << a << "\n";
            return false;
        } else if (args->audio.empty()) {
            args->audio = a;
        } else {
            std::cerr << "only one audio input is supported by the C++ bridge CLI\n";
            return false;
        }
    }
    if (args->audio.empty()) {
        std::cerr << "audio input is required\n";
        return false;
    }
    args->runs   = std::max(1, args->runs);
    args->warmup = std::max(0, args->warmup);
    return true;
}

static uint16_t read_u16_le(std::istream & in) {
    unsigned char b[2] = { 0, 0 };
    in.read(reinterpret_cast<char *>(b), 2);
    return static_cast<uint16_t>(b[0] | (b[1] << 8));
}

static uint32_t read_u32_le(std::istream & in) {
    unsigned char b[4] = { 0, 0, 0, 0 };
    in.read(reinterpret_cast<char *>(b), 4);
    return static_cast<uint32_t>(b[0] | (b[1] << 8) | (b[2] << 16) | (b[3] << 24));
}

static double wav_duration_seconds(const std::string & path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return 0.0;
    }
    char riff[4] = {};
    char wave[4] = {};
    in.read(riff, 4);
    (void) read_u32_le(in);
    in.read(wave, 4);
    if (std::strncmp(riff, "RIFF", 4) != 0 || std::strncmp(wave, "WAVE", 4) != 0) {
        return 0.0;
    }

    uint16_t channels        = 0;
    uint32_t sample_rate     = 0;
    uint16_t bits_per_sample = 0;
    uint32_t data_bytes      = 0;

    while (in && (!sample_rate || !data_bytes)) {
        char id[4] = {};
        in.read(id, 4);
        if (!in) {
            break;
        }
        uint32_t size = read_u32_le(in);
        std::streampos data_start = in.tellg();
        if (std::strncmp(id, "fmt ", 4) == 0 && size >= 16) {
            (void) read_u16_le(in);
            channels        = read_u16_le(in);
            sample_rate     = read_u32_le(in);
            (void) read_u32_le(in);
            (void) read_u16_le(in);
            bits_per_sample = read_u16_le(in);
        } else if (std::strncmp(id, "data", 4) == 0) {
            data_bytes = size;
        }
        in.seekg(data_start + static_cast<std::streamoff>(size + (size & 1U)));
    }

    if (!channels || !sample_rate || !bits_per_sample || !data_bytes) {
        return 0.0;
    }
    double bytes_per_sample = static_cast<double>(bits_per_sample) / 8.0;
    return static_cast<double>(data_bytes) / (static_cast<double>(sample_rate) * channels * bytes_per_sample);
}

static std::string json_escape(const std::string & s) {
    std::ostringstream out;
    for (unsigned char c : s) {
        switch (c) {
        case '\\': out << "\\\\"; break;
        case '"': out << "\\\""; break;
        case '\n': out << "\\n"; break;
        case '\r': out << "\\r"; break;
        case '\t': out << "\\t"; break;
        default:
            if (c < 0x20) {
                out << "\\u";
                const char * hex = "0123456789abcdef";
                out << "00" << hex[(c >> 4) & 0xf] << hex[c & 0xf];
            } else {
                out << static_cast<char>(c);
            }
        }
    }
    return out.str();
}

int main(int argc, char ** argv) {
    CliArgs args;
    if (!parse_args(argc, argv, &args)) {
        usage(argv[0]);
        return 2;
    }

    qa_init_params init;
    qa_init_default_params(&init);
    init.model               = args.model.empty() ? nullptr : args.model.c_str();
    init.size                = args.model.empty() ? args.size.c_str() : nullptr;
    init.backend             = args.backend.c_str();
    init.device              = args.device.c_str();
    init.dtype               = args.dtype.c_str();
    init.attn_implementation = args.attn_implementation.c_str();
    init.python_path         = args.python_path.empty() ? nullptr : args.python_path.c_str();
    init.max_new_tokens      = args.max_new_tokens;
    init.max_batch_size      = args.batch_size;
    init.use_cuda_graph      = args.use_cuda_graph;
    init.cuda_graph_stride   = args.cuda_graph_stride;
    init.local_files_only    = args.local_files_only;

    auto load_start = std::chrono::steady_clock::now();
    qa_context * ctx = qa_init(&init);
    if (!ctx) {
        std::cerr << qa_last_error() << "\n";
        return 1;
    }
    auto load_end = std::chrono::steady_clock::now();

    qa_transcribe_params tp;
    qa_transcribe_default_params(&tp);
    tp.audio    = args.audio.c_str();
    tp.language = args.language.empty() ? nullptr : args.language.c_str();
    tp.context  = args.context.c_str();

    for (int i = 0; i < args.warmup; ++i) {
        qa_transcription tmp {};
        qa_status rc = qa_transcribe(ctx, &tp, &tmp);
        qa_transcription_free(&tmp);
        if (rc != QA_STATUS_OK) {
            std::cerr << qa_last_error() << "\n";
            qa_free(ctx);
            return 1;
        }
    }
    (void) qa_synchronize();

    double              best = std::numeric_limits<double>::infinity();
    qa_transcription    last {};
    std::vector<double> times;
    times.reserve(static_cast<size_t>(args.runs));
    for (int i = 0; i < args.runs; ++i) {
        qa_transcription_free(&last);
        (void) qa_synchronize();
        auto start = std::chrono::steady_clock::now();
        qa_status rc = qa_transcribe(ctx, &tp, &last);
        (void) qa_synchronize();
        auto end = std::chrono::steady_clock::now();
        if (rc != QA_STATUS_OK) {
            std::cerr << qa_last_error() << "\n";
            qa_free(ctx);
            return 1;
        }
        double sec = std::chrono::duration<double>(end - start).count();
        times.push_back(sec);
        best = std::min(best, sec);
    }

    double load_sec  = std::chrono::duration<double>(load_end - load_start).count();
    double audio_sec = wav_duration_seconds(args.audio);
    double rtf       = (audio_sec > 0.0 && best > 0.0) ? (audio_sec / best) : 0.0;

    if (args.json) {
        std::cout << "{\n"
                  << "  \"model\": \"" << json_escape(qa_model_id(ctx)) << "\",\n"
                  << "  \"backend\": \"" << json_escape(qa_backend(ctx)) << "\",\n"
                  << "  \"load_sec\": " << load_sec << ",\n"
                  << "  \"best_sec\": " << best << ",\n"
                  << "  \"audio_sec\": " << audio_sec << ",\n"
                  << "  \"rtf\": " << rtf << ",\n"
                  << "  \"text\": \"" << json_escape(last.text ? last.text : "") << "\"\n"
                  << "}\n";
    } else {
        std::cout << "model=" << qa_model_id(ctx) << "\n";
        std::cout << "backend=" << qa_backend(ctx) << "\n";
        std::cout << "bridge=python\n";
        std::cout << "load_sec=" << load_sec << "\n";
        std::cout << "best_sec=" << best << "\n";
        if (audio_sec > 0.0) {
            std::cout << "audio_sec=" << audio_sec << "\n";
            std::cout << "rtf=" << rtf << "\n";
        }
        std::cout << "text=" << (last.text ? last.text : "") << "\n";
    }

    qa_transcription_free(&last);
    qa_free(ctx);
    return 0;
}
