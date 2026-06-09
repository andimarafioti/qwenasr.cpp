#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

struct QwenAsrTokenizer {
    std::unordered_map<std::string, int32_t> vocab;
    std::unordered_map<std::string, int32_t> merges;
    std::vector<std::string> id_to_token;
    std::vector<std::pair<std::string, int32_t>> specials;
    std::string byte_encoder[256];
    int32_t eos_token = -1;
};

bool qwenasr_tokenizer_load_gguf(
    const char * path,
    QwenAsrTokenizer * out,
    std::string * error);

std::vector<int32_t> qwenasr_tokenizer_encode(
    const QwenAsrTokenizer & tokenizer,
    const std::string & text,
    bool add_eos = false);

std::string qwenasr_tokenizer_decode(
    const QwenAsrTokenizer & tokenizer,
    const std::vector<int32_t> & ids,
    bool skip_special = true);

std::string qwenasr_build_asr_prompt(
    int audio_tokens,
    const std::string & system_text = "",
    const std::string & language = "");
