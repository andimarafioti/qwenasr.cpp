#include "tokenizer.h"

#include "gguf.h"

#include <algorithm>
#include <cassert>
#include <climits>
#include <cctype>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

static void build_byte_encoder(std::string byte_encoder[256]) {
    int bs[256];
    int cs[256];
    int n = 0;
    int total = 0;
    for (int b = '!'; b <= '~'; ++b) {
        bs[total] = b;
        cs[total] = b;
        ++total;
    }
    for (int b = 0xA1; b <= 0xAC; ++b) {
        bs[total] = b;
        cs[total] = b;
        ++total;
    }
    for (int b = 0xAE; b <= 0xFF; ++b) {
        bs[total] = b;
        cs[total] = b;
        ++total;
    }

    bool used[256] = {};
    for (int i = 0; i < total; ++i) {
        used[bs[i]] = true;
    }
    for (int b = 0; b < 256; ++b) {
        if (!used[b]) {
            bs[total] = b;
            cs[total] = 256 + n;
            ++n;
            ++total;
        }
    }
    assert(total == 256);

    for (int i = 0; i < 256; ++i) {
        const int cp = cs[i];
        char buf[4];
        int len = 0;
        if (cp < 0x80) {
            buf[0] = static_cast<char>(cp);
            len = 1;
        } else if (cp < 0x800) {
            buf[0] = static_cast<char>(0xC0 | (cp >> 6));
            buf[1] = static_cast<char>(0x80 | (cp & 0x3F));
            len = 2;
        } else {
            buf[0] = static_cast<char>(0xE0 | (cp >> 12));
            buf[1] = static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            buf[2] = static_cast<char>(0x80 | (cp & 0x3F));
            len = 3;
        }
        byte_encoder[bs[i]] = std::string(buf, static_cast<size_t>(len));
    }
}

static int utf8_codepoint(const char * s, int * advance) {
    const unsigned char c = static_cast<unsigned char>(s[0]);
    if (c < 0x80) {
        *advance = 1;
        return c;
    }
    if ((c & 0xE0) == 0xC0) {
        *advance = 2;
        return ((c & 0x1F) << 6) | (static_cast<unsigned char>(s[1]) & 0x3F);
    }
    if ((c & 0xF0) == 0xE0) {
        *advance = 3;
        return ((c & 0x0F) << 12) |
            ((static_cast<unsigned char>(s[1]) & 0x3F) << 6) |
            (static_cast<unsigned char>(s[2]) & 0x3F);
    }
    if ((c & 0xF8) == 0xF0) {
        *advance = 4;
        return ((c & 0x07) << 18) |
            ((static_cast<unsigned char>(s[1]) & 0x3F) << 12) |
            ((static_cast<unsigned char>(s[2]) & 0x3F) << 6) |
            (static_cast<unsigned char>(s[3]) & 0x3F);
    }
    *advance = 1;
    return c;
}

static bool is_letter(int cp) {
    if ((cp >= 'A' && cp <= 'Z') || (cp >= 'a' && cp <= 'z')) {
        return true;
    }
    if (cp < 0x80) {
        return false;
    }
    if (cp >= 0xC0 && cp <= 0x024F && cp != 0xD7 && cp != 0xF7) {
        return true;
    }
    if (cp >= 0x0370 && cp <= 0x1FFF) {
        return true;
    }
    if (cp >= 0x2C00 && cp <= 0x2DFF) {
        return true;
    }
    if (cp >= 0x3040 && cp <= 0x9FFF) {
        return true;
    }
    if (cp >= 0xAC00 && cp <= 0xD7AF) {
        return true;
    }
    if (cp >= 0xF900 && cp <= 0xFAFF) {
        return true;
    }
    return cp >= 0x10000;
}

static bool is_digit(int cp) {
    return cp >= '0' && cp <= '9';
}

static bool is_newline(int cp) {
    return cp == '\n' || cp == '\r';
}

static bool is_space_not_newline(int cp) {
    return cp == ' ' || cp == '\t' || cp == 0x0B || cp == 0x0C || cp == 0xA0;
}

static std::vector<std::string> pre_tokenize(const std::string & text) {
    std::vector<std::string> chunks;
    const char * s = text.c_str();
    const int len = static_cast<int>(text.size());
    int i = 0;

    while (i < len) {
        int adv = 0;
        int cp = utf8_codepoint(s + i, &adv);

        if (cp == '\'' && i + adv < len) {
            const char * rest = s + i + adv;
            const int rlen = len - i - adv;
            const char * suffixes[] = { "ll", "re", "ve", "s", "t", "m", "d" };
            bool matched = false;
            for (const char * suffix : suffixes) {
                const int slen = static_cast<int>(std::char_traits<char>::length(suffix));
                if (rlen < slen) {
                    continue;
                }
                bool matches = true;
                for (int k = 0; k < slen; ++k) {
                    if (static_cast<char>(std::tolower(rest[k])) != suffix[k]) {
                        matches = false;
                        break;
                    }
                }
                if (matches) {
                    chunks.push_back(std::string(s + i, static_cast<size_t>(adv + slen)));
                    i += adv + slen;
                    matched = true;
                    break;
                }
            }
            if (matched) {
                continue;
            }
        }

        if (is_space_not_newline(cp) && i + adv < len) {
            int next_adv = 0;
            const int next_cp = utf8_codepoint(s + i + adv, &next_adv);
            if (is_letter(next_cp)) {
                const int start = i;
                i += adv + next_adv;
                while (i < len) {
                    int a2 = 0;
                    const int cp2 = utf8_codepoint(s + i, &a2);
                    if (!is_letter(cp2)) {
                        break;
                    }
                    i += a2;
                }
                chunks.push_back(std::string(s + start, static_cast<size_t>(i - start)));
                continue;
            }
        }

        if (is_letter(cp)) {
            const int start = i;
            i += adv;
            while (i < len) {
                int a2 = 0;
                const int cp2 = utf8_codepoint(s + i, &a2);
                if (!is_letter(cp2)) {
                    break;
                }
                i += a2;
            }
            chunks.push_back(std::string(s + start, static_cast<size_t>(i - start)));
            continue;
        }

        if (is_digit(cp)) {
            chunks.push_back(std::string(s + i, static_cast<size_t>(adv)));
            i += adv;
            continue;
        }

        if (is_newline(cp)) {
            const int start = i;
            i += adv;
            while (i < len) {
                int a2 = 0;
                const int cp2 = utf8_codepoint(s + i, &a2);
                if (!is_newline(cp2)) {
                    break;
                }
                i += a2;
            }
            chunks.push_back(std::string(s + start, static_cast<size_t>(i - start)));
            continue;
        }

        if (is_space_not_newline(cp)) {
            const int start = i;
            i += adv;
            while (i < len) {
                int a2 = 0;
                const int cp2 = utf8_codepoint(s + i, &a2);
                if (!is_space_not_newline(cp2)) {
                    break;
                }
                i += a2;
            }
            chunks.push_back(std::string(s + start, static_cast<size_t>(i - start)));
            continue;
        }

        {
            const int start = i;
            i += adv;
            while (i < len) {
                int a2 = 0;
                const int cp2 = utf8_codepoint(s + i, &a2);
                if (is_letter(cp2) || is_digit(cp2) || is_newline(cp2) || is_space_not_newline(cp2)) {
                    break;
                }
                i += a2;
            }
            chunks.push_back(std::string(s + start, static_cast<size_t>(i - start)));
        }
    }

    return chunks;
}

static void add_special(QwenAsrTokenizer * tok, const std::string & token, int32_t id) {
    if (token.empty() || id < 0) {
        return;
    }
    for (const auto & item : tok->specials) {
        if (item.first == token) {
            return;
        }
    }
    tok->specials.emplace_back(token, id);
    std::sort(tok->specials.begin(), tok->specials.end(), [](const auto & a, const auto & b) {
        if (a.first.size() != b.first.size()) {
            return a.first.size() > b.first.size();
        }
        return a.first < b.first;
    });
}

static bool add_special_from_key(
    const gguf_context * ctx,
    const char * key,
    QwenAsrTokenizer * tok,
    std::string * error) {
    const int64_t kid = gguf_find_key(ctx, key);
    if (kid < 0) {
        if (error) {
            *error = std::string("missing tokenizer special key: ") + key;
        }
        return false;
    }
    const int32_t id = static_cast<int32_t>(gguf_get_val_u32(ctx, kid));
    if (id < 0 || static_cast<size_t>(id) >= tok->id_to_token.size()) {
        if (error) {
            *error = std::string("tokenizer special id out of range for key: ") + key;
        }
        return false;
    }
    add_special(tok, tok->id_to_token[static_cast<size_t>(id)], id);
    return true;
}

bool qwenasr_tokenizer_load_gguf(const char * path, QwenAsrTokenizer * out, std::string * error) {
    if (!path || !out) {
        if (error) {
            *error = "qwenasr_tokenizer_load_gguf: path or out is null";
        }
        return false;
    }

    gguf_init_params params {};
    params.no_alloc = true;
    params.ctx = nullptr;
    gguf_context * ctx = gguf_init_from_file(path, params);
    if (!ctx) {
        if (error) {
            *error = std::string("failed to open GGUF tokenizer: ") + path;
        }
        return false;
    }

    const int64_t tok_key = gguf_find_key(ctx, "tokenizer.ggml.tokens");
    const int64_t merge_key = gguf_find_key(ctx, "tokenizer.ggml.merges");
    const int64_t eos_key = gguf_find_key(ctx, "tokenizer.ggml.eos_token_id");
    if (tok_key < 0 || merge_key < 0 || eos_key < 0) {
        if (error) {
            *error = "GGUF tokenizer metadata is incomplete";
        }
        gguf_free(ctx);
        return false;
    }

    QwenAsrTokenizer tok;
    build_byte_encoder(tok.byte_encoder);
    tok.eos_token = static_cast<int32_t>(gguf_get_val_u32(ctx, eos_key));

    const size_t n_tokens = gguf_get_arr_n(ctx, tok_key);
    tok.id_to_token.resize(n_tokens);
    tok.vocab.reserve(n_tokens);
    for (size_t i = 0; i < n_tokens; ++i) {
        const char * token = gguf_get_arr_str(ctx, tok_key, i);
        tok.id_to_token[i] = token ? token : "";
        tok.vocab[tok.id_to_token[i]] = static_cast<int32_t>(i);
    }

    const size_t n_merges = gguf_get_arr_n(ctx, merge_key);
    tok.merges.reserve(n_merges);
    for (size_t i = 0; i < n_merges; ++i) {
        const char * merge = gguf_get_arr_str(ctx, merge_key, i);
        tok.merges[merge ? merge : ""] = static_cast<int32_t>(i);
    }

    const char * special_keys[] = {
        "qwen3-asr.token.endoftext_token_id",
        "qwen3-asr.token.im_start_token_id",
        "qwen3-asr.token.im_end_token_id",
        "qwen3-asr.token.audio_start_token_id",
        "qwen3-asr.token.audio_end_token_id",
        "qwen3-asr.token.audio_token_id",
        "qwen3-asr.token.asr_text_token_id",
    };
    for (const char * key : special_keys) {
        if (!add_special_from_key(ctx, key, &tok, error)) {
            gguf_free(ctx);
            return false;
        }
    }

    gguf_free(ctx);
    *out = std::move(tok);
    return true;
}

static std::string byte_level_encode(const QwenAsrTokenizer & tok, const std::string & text) {
    std::string out;
    for (unsigned char c : text) {
        out += tok.byte_encoder[c];
    }
    return out;
}

static std::vector<std::string> bpe_merge(
    const std::unordered_map<std::string, int32_t> & ranks,
    const std::vector<std::string> & symbols) {
    if (symbols.size() <= 1) {
        return symbols;
    }

    std::vector<std::string> work = symbols;
    while (work.size() > 1) {
        int32_t best_rank = INT_MAX;
        int best_pos = -1;
        for (int i = 0; i + 1 < static_cast<int>(work.size()); ++i) {
            const std::string key = work[static_cast<size_t>(i)] + " " + work[static_cast<size_t>(i + 1)];
            const auto it = ranks.find(key);
            if (it != ranks.end() && it->second < best_rank) {
                best_rank = it->second;
                best_pos = i;
            }
        }
        if (best_pos < 0) {
            break;
        }
        work[static_cast<size_t>(best_pos)] += work[static_cast<size_t>(best_pos + 1)];
        work.erase(work.begin() + best_pos + 1);
    }
    return work;
}

static void encode_chunk(
    const QwenAsrTokenizer & tok,
    const std::string & chunk,
    std::vector<int32_t> * ids) {
    const std::string encoded = byte_level_encode(tok, chunk);
    std::vector<std::string> symbols;
    const char * s = encoded.c_str();
    const int len = static_cast<int>(encoded.size());
    int i = 0;
    while (i < len) {
        int adv = 0;
        utf8_codepoint(s + i, &adv);
        symbols.emplace_back(s + i, static_cast<size_t>(adv));
        i += adv;
    }

    const std::vector<std::string> merged = bpe_merge(tok.merges, symbols);
    for (const std::string & piece : merged) {
        const auto it = tok.vocab.find(piece);
        if (it != tok.vocab.end()) {
            ids->push_back(it->second);
        }
    }
}

std::vector<int32_t> qwenasr_tokenizer_encode(
    const QwenAsrTokenizer & tokenizer,
    const std::string & text,
    bool add_eos) {
    std::vector<int32_t> ids;

    auto encode_segment = [&](const std::string & segment) {
        const std::vector<std::string> chunks = pre_tokenize(segment);
        for (const std::string & chunk : chunks) {
            encode_chunk(tokenizer, chunk, &ids);
        }
    };

    size_t pos = 0;
    while (pos < text.size()) {
        size_t best_pos = std::string::npos;
        int best_idx = -1;
        for (size_t i = 0; i < tokenizer.specials.size(); ++i) {
            const size_t p = text.find(tokenizer.specials[i].first, pos);
            if (p != std::string::npos && (p < best_pos ||
                    (p == best_pos && best_idx >= 0 &&
                     tokenizer.specials[i].first.size() > tokenizer.specials[static_cast<size_t>(best_idx)].first.size()))) {
                best_pos = p;
                best_idx = static_cast<int>(i);
            }
        }

        if (best_idx < 0) {
            encode_segment(text.substr(pos));
            break;
        }
        if (best_pos > pos) {
            encode_segment(text.substr(pos, best_pos - pos));
        }
        ids.push_back(tokenizer.specials[static_cast<size_t>(best_idx)].second);
        pos = best_pos + tokenizer.specials[static_cast<size_t>(best_idx)].first.size();
    }

    if (add_eos && tokenizer.eos_token >= 0) {
        ids.push_back(tokenizer.eos_token);
    }
    return ids;
}

std::string qwenasr_build_asr_prompt(
    int audio_tokens,
    const std::string & system_text,
    const std::string & language) {
    if (audio_tokens < 0) {
        audio_tokens = 0;
    }
    std::string prompt = "<|im_start|>system\n";
    prompt += system_text;
    prompt += "<|im_end|>\n<|im_start|>user\n<|audio_start|>";
    for (int i = 0; i < audio_tokens; ++i) {
        prompt += "<|audio_pad|>";
    }
    prompt += "<|audio_end|><|im_end|>\n<|im_start|>assistant\n";
    if (!language.empty()) {
        prompt += "language ";
        prompt += language;
        prompt += "<asr_text>";
    }
    return prompt;
}
