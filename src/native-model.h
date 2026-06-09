#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>

struct QwenAsrNativeConfig {
    std::string name;
    std::string arch;

    uint32_t audio_sample_rate = 0;
    uint32_t audio_num_mel_bins = 0;
    uint32_t audio_d_model = 0;
    uint32_t audio_encoder_layers = 0;
    uint32_t audio_encoder_attention_heads = 0;
    uint32_t audio_encoder_ffn_dim = 0;
    uint32_t audio_downsample_hidden_size = 0;
    uint32_t audio_output_dim = 0;

    uint32_t text_vocab_size = 0;
    uint32_t text_hidden_size = 0;
    uint32_t text_intermediate_size = 0;
    uint32_t text_num_hidden_layers = 0;
    uint32_t text_num_attention_heads = 0;
    uint32_t text_num_key_value_heads = 0;
    uint32_t text_head_dim = 0;
    float    text_rope_theta = 0.0f;
    float    text_rms_norm_eps = 0.0f;

    uint32_t token_audio = 0;
    uint32_t token_audio_start = 0;
    uint32_t token_audio_end = 0;

    int64_t n_kv = 0;
    int64_t n_tensors = 0;
    size_t  alignment = 0;
    size_t  data_offset = 0;

    std::vector<std::string> first_tensors;
};

struct QwenAsrTensorSpec {
    std::string name;
    std::vector<int64_t> ne; // GGML order, ne[0] is the innermost dimension.
};

bool qwenasr_load_gguf_metadata(
    const char * path,
    bool require_tensors,
    QwenAsrNativeConfig * out,
    std::string * error);

bool qwenasr_write_metadata_fixture(const char * path, std::string * error);

std::vector<QwenAsrTensorSpec> qwenasr_expected_tensor_specs(const QwenAsrNativeConfig & cfg);

std::vector<std::string> qwenasr_expected_tensor_names(const QwenAsrNativeConfig & cfg);
