#include "native-model.h"

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>

static void usage(const char * argv0) {
    std::cerr
        << "Usage: " << argv0 << " model.gguf [--allow-metadata-only]\n"
        << "       " << argv0 << " --self-test\n"
        << "\n"
        << "Load and validate qwenasr.cpp native GGUF metadata.\n";
}

static void print_config(const QwenAsrNativeConfig & cfg) {
    std::cout << "name=" << cfg.name << "\n";
    std::cout << "arch=" << cfg.arch << "\n";
    std::cout << "kv=" << cfg.n_kv << "\n";
    std::cout << "tensors=" << cfg.n_tensors << "\n";
    std::cout << "alignment=" << cfg.alignment << "\n";
    std::cout << "data_offset=" << cfg.data_offset << "\n";
    std::cout << "audio_sample_rate=" << cfg.audio_sample_rate << "\n";
    std::cout << "audio_num_mel_bins=" << cfg.audio_num_mel_bins << "\n";
    std::cout << "audio_d_model=" << cfg.audio_d_model << "\n";
    std::cout << "audio_encoder_layers=" << cfg.audio_encoder_layers << "\n";
    std::cout << "audio_encoder_attention_heads=" << cfg.audio_encoder_attention_heads << "\n";
    std::cout << "audio_encoder_ffn_dim=" << cfg.audio_encoder_ffn_dim << "\n";
    std::cout << "audio_downsample_hidden_size=" << cfg.audio_downsample_hidden_size << "\n";
    std::cout << "audio_output_dim=" << cfg.audio_output_dim << "\n";
    std::cout << "text_vocab_size=" << cfg.text_vocab_size << "\n";
    std::cout << "text_hidden_size=" << cfg.text_hidden_size << "\n";
    std::cout << "text_intermediate_size=" << cfg.text_intermediate_size << "\n";
    std::cout << "text_num_hidden_layers=" << cfg.text_num_hidden_layers << "\n";
    std::cout << "text_num_attention_heads=" << cfg.text_num_attention_heads << "\n";
    std::cout << "text_num_key_value_heads=" << cfg.text_num_key_value_heads << "\n";
    std::cout << "text_head_dim=" << cfg.text_head_dim << "\n";
    std::cout << "text_rope_theta=" << cfg.text_rope_theta << "\n";
    std::cout << "text_rms_norm_eps=" << cfg.text_rms_norm_eps << "\n";
    std::cout << "token_audio=" << cfg.token_audio << "\n";
    std::cout << "token_audio_start=" << cfg.token_audio_start << "\n";
    std::cout << "token_audio_end=" << cfg.token_audio_end << "\n";
    for (size_t i = 0; i < cfg.first_tensors.size(); ++i) {
        std::cout << "tensor." << i << "=" << cfg.first_tensors[i] << "\n";
    }
}

static int self_test() {
    const char * path = "/tmp/qwenasr-gguf-info-fixture.gguf";
    std::string error;
    if (!qwenasr_write_metadata_fixture(path, &error)) {
        std::cerr << error << "\n";
        return 1;
    }

    QwenAsrNativeConfig cfg;
    if (!qwenasr_load_gguf_metadata(path, false, &cfg, &error)) {
        std::remove(path);
        std::cerr << error << "\n";
        return 1;
    }
    std::remove(path);
    print_config(cfg);
    std::cout << "status=self-test-ok\n";
    return 0;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        usage(argv[0]);
        return 2;
    }

    bool        allow_metadata_only = false;
    std::string path;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            usage(argv[0]);
            return 0;
        }
        if (arg == "--self-test") {
            return self_test();
        }
        if (arg == "--allow-metadata-only") {
            allow_metadata_only = true;
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

    QwenAsrNativeConfig cfg;
    std::string         error;
    if (!qwenasr_load_gguf_metadata(path.c_str(), !allow_metadata_only, &cfg, &error)) {
        std::cerr << error << "\n";
        return 1;
    }
    print_config(cfg);
    return 0;
}
