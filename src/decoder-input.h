#pragma once

#include "audio-features.h"
#include "gguf-model.h"
#include "tokenizer.h"

#include <cstdint>
#include <string>
#include <vector>

struct QwenAsrDecoderInputOutput {
    int tokens = 0;
    int hidden = 0;
    int audio_tokens = 0;
    std::vector<int32_t> input_ids;
    std::vector<float> values; // [token, hidden]
};

bool qwenasr_decoder_input_forward_ggml(
    const QwenAsrGgufModel & model,
    const QwenAsrTokenizer & tokenizer,
    const QwenAsrFeatures & features,
    int n_threads,
    int n_audio_layers,
    int n_audio_heads,
    int audio_output_dim,
    int audio_token_id,
    const std::string & system_text,
    const std::string & language,
    QwenAsrDecoderInputOutput * out,
    std::string * error);
