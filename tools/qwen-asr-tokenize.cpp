#include "tokenizer.h"

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

static void usage(const char * argv0) {
    std::cerr
        << "Usage: " << argv0 << " model.gguf [--audio-tokens N] [--system TEXT]\n"
        << "       " << argv0 << " model.gguf [--audio-tokens N] [--language NAME]\n"
        << "\n"
        << "Load the native GGUF tokenizer and print ASR prompt token ids.\n";
}

static bool parse_int(const std::string & text, int * out) {
    char * end = nullptr;
    const long value = std::strtol(text.c_str(), &end, 10);
    if (!end || *end != '\0' || value < 0 || value > 10000000L) {
        return false;
    }
    *out = static_cast<int>(value);
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        usage(argv[0]);
        return 2;
    }

    std::string path;
    std::string system_text;
    std::string language;
    int audio_tokens = 1;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            usage(argv[0]);
            return 0;
        }
        if (arg == "--audio-tokens") {
            if (++i >= argc || !parse_int(argv[i], &audio_tokens)) {
                std::cerr << "--audio-tokens requires a non-negative integer\n";
                return 2;
            }
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
        if (arg == "--language") {
            if (++i >= argc) {
                std::cerr << "--language requires a value\n";
                return 2;
            }
            language = argv[i];
            continue;
        }
        if (!path.empty()) {
            std::cerr << "only one GGUF path is supported\n";
            return 2;
        }
        path = arg;
    }

    if (path.empty()) {
        std::cerr << "GGUF path is required\n";
        return 2;
    }

    QwenAsrTokenizer tokenizer;
    std::string error;
    if (!qwenasr_tokenizer_load_gguf(path.c_str(), &tokenizer, &error)) {
        std::cerr << error << "\n";
        return 1;
    }

    const std::string prompt = qwenasr_build_asr_prompt(audio_tokens, system_text, language);
    const std::vector<int32_t> ids = qwenasr_tokenizer_encode(tokenizer, prompt, false);

    std::cout << "vocab=" << tokenizer.id_to_token.size() << "\n";
    std::cout << "merges=" << tokenizer.merges.size() << "\n";
    std::cout << "specials=" << tokenizer.specials.size() << "\n";
    std::cout << "audio_tokens=" << audio_tokens << "\n";
    std::cout << "prompt_bytes=" << prompt.size() << "\n";
    std::cout << "tokens=" << ids.size() << "\n";
    std::cout << "ids=";
    for (size_t i = 0; i < ids.size(); ++i) {
        if (i > 0) {
            std::cout << ",";
        }
        std::cout << ids[i];
    }
    std::cout << "\n";
    return 0;
}
