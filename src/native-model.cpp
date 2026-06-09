#include "native-model.h"

#include "gguf.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>

static bool require_key(const gguf_context * ctx, const char * key, int64_t * id, std::string * error) {
    *id = gguf_find_key(ctx, key);
    if (*id < 0) {
        if (error) {
            *error = std::string("missing GGUF key: ") + key;
        }
        return false;
    }
    return true;
}

static bool get_u32(const gguf_context * ctx, const char * key, uint32_t * out, std::string * error) {
    int64_t id = -1;
    if (!require_key(ctx, key, &id, error)) {
        return false;
    }
    *out = gguf_get_val_u32(ctx, id);
    return true;
}

static bool get_f32(const gguf_context * ctx, const char * key, float * out, std::string * error) {
    int64_t id = -1;
    if (!require_key(ctx, key, &id, error)) {
        return false;
    }
    *out = gguf_get_val_f32(ctx, id);
    return true;
}

static bool get_str(const gguf_context * ctx, const char * key, std::string * out, std::string * error) {
    int64_t id = -1;
    if (!require_key(ctx, key, &id, error)) {
        return false;
    }
    const char * value = gguf_get_val_str(ctx, id);
    *out = value ? value : "";
    return true;
}

static bool require_tensor(const gguf_context * ctx, const char * name, std::string * error) {
    if (gguf_find_tensor(ctx, name) >= 0) {
        return true;
    }
    if (error) {
        *error = std::string("missing GGUF tensor: ") + name;
    }
    return false;
}

bool qwenasr_load_gguf_metadata(
    const char * path,
    bool require_core_tensors,
    QwenAsrNativeConfig * out,
    std::string * error) {
    if (!path || !out) {
        if (error) {
            *error = "qwenasr_load_gguf_metadata: path or out is null";
        }
        return false;
    }

    gguf_init_params params {};
    params.no_alloc = true;
    params.ctx = nullptr;
    gguf_context * ctx = gguf_init_from_file(path, params);
    if (!ctx) {
        if (error) {
            *error = std::string("failed to open GGUF: ") + path;
        }
        return false;
    }

    QwenAsrNativeConfig cfg;
    cfg.n_kv = gguf_get_n_kv(ctx);
    cfg.n_tensors = gguf_get_n_tensors(ctx);
    cfg.alignment = gguf_get_alignment(ctx);
    cfg.data_offset = gguf_get_data_offset(ctx);

    bool ok = true;
    ok = ok && get_str(ctx, "general.name", &cfg.name, error);
    ok = ok && get_str(ctx, "qwen3-asr.arch", &cfg.arch, error);
    ok = ok && get_u32(ctx, "qwen3-asr.audio.sample_rate", &cfg.audio_sample_rate, error);
    ok = ok && get_u32(ctx, "qwen3-asr.audio.num_mel_bins", &cfg.audio_num_mel_bins, error);
    ok = ok && get_u32(ctx, "qwen3-asr.audio.d_model", &cfg.audio_d_model, error);
    ok = ok && get_u32(ctx, "qwen3-asr.audio.encoder_layers", &cfg.audio_encoder_layers, error);
    ok = ok && get_u32(ctx, "qwen3-asr.audio.encoder_attention_heads", &cfg.audio_encoder_attention_heads, error);
    ok = ok && get_u32(ctx, "qwen3-asr.audio.encoder_ffn_dim", &cfg.audio_encoder_ffn_dim, error);
    ok = ok && get_u32(ctx, "qwen3-asr.audio.downsample_hidden_size", &cfg.audio_downsample_hidden_size, error);
    ok = ok && get_u32(ctx, "qwen3-asr.audio.output_dim", &cfg.audio_output_dim, error);
    ok = ok && get_u32(ctx, "qwen3-asr.text.vocab_size", &cfg.text_vocab_size, error);
    ok = ok && get_u32(ctx, "qwen3-asr.text.hidden_size", &cfg.text_hidden_size, error);
    ok = ok && get_u32(ctx, "qwen3-asr.text.intermediate_size", &cfg.text_intermediate_size, error);
    ok = ok && get_u32(ctx, "qwen3-asr.text.num_hidden_layers", &cfg.text_num_hidden_layers, error);
    ok = ok && get_u32(ctx, "qwen3-asr.text.num_attention_heads", &cfg.text_num_attention_heads, error);
    ok = ok && get_u32(ctx, "qwen3-asr.text.num_key_value_heads", &cfg.text_num_key_value_heads, error);
    ok = ok && get_u32(ctx, "qwen3-asr.text.head_dim", &cfg.text_head_dim, error);
    ok = ok && get_f32(ctx, "qwen3-asr.text.rope_theta", &cfg.text_rope_theta, error);
    ok = ok && get_f32(ctx, "qwen3-asr.text.rms_norm_eps", &cfg.text_rms_norm_eps, error);
    ok = ok && get_u32(ctx, "qwen3-asr.token.audio_token_id", &cfg.token_audio, error);
    ok = ok && get_u32(ctx, "qwen3-asr.token.audio_start_token_id", &cfg.token_audio_start, error);
    ok = ok && get_u32(ctx, "qwen3-asr.token.audio_end_token_id", &cfg.token_audio_end, error);

    if (ok && cfg.arch != "qwen3-asr") {
        if (error) {
            *error = "GGUF arch is not qwen3-asr";
        }
        ok = false;
    }

    if (ok && require_core_tensors) {
        ok = ok && require_tensor(ctx, "text.token_embd.weight", error);
        ok = ok && require_tensor(ctx, "text.output_norm.weight", error);
        ok = ok && require_tensor(ctx, "text.output.weight", error);
        ok = ok && require_tensor(ctx, "audio.conv.0.weight", error);
        ok = ok && require_tensor(ctx, "audio.proj.1.weight", error);
    }

    if (ok) {
        const int64_t limit = std::min<int64_t>(cfg.n_tensors, 12);
        for (int64_t i = 0; i < limit; ++i) {
            const char * name = gguf_get_tensor_name(ctx, i);
            cfg.first_tensors.push_back(name ? name : "");
        }
        *out = cfg;
    }

    gguf_free(ctx);
    return ok;
}

bool qwenasr_write_metadata_fixture(const char * path, std::string * error) {
    if (!path) {
        if (error) {
            *error = "qwenasr_write_metadata_fixture: path is null";
        }
        return false;
    }

    gguf_context * ctx = gguf_init_empty();
    if (!ctx) {
        if (error) {
            *error = "gguf_init_empty failed";
        }
        return false;
    }

    gguf_set_val_str(ctx, "general.name", "Qwen3-ASR metadata fixture");
    gguf_set_val_str(ctx, "qwen3-asr.arch", "qwen3-asr");
    gguf_set_val_u32(ctx, "qwen3-asr.audio.sample_rate", 16000);
    gguf_set_val_u32(ctx, "qwen3-asr.audio.num_mel_bins", 128);
    gguf_set_val_u32(ctx, "qwen3-asr.audio.d_model", 896);
    gguf_set_val_u32(ctx, "qwen3-asr.audio.encoder_layers", 18);
    gguf_set_val_u32(ctx, "qwen3-asr.audio.encoder_attention_heads", 14);
    gguf_set_val_u32(ctx, "qwen3-asr.audio.encoder_ffn_dim", 3584);
    gguf_set_val_u32(ctx, "qwen3-asr.audio.downsample_hidden_size", 480);
    gguf_set_val_u32(ctx, "qwen3-asr.audio.output_dim", 1024);
    gguf_set_val_u32(ctx, "qwen3-asr.text.vocab_size", 151936);
    gguf_set_val_u32(ctx, "qwen3-asr.text.hidden_size", 1024);
    gguf_set_val_u32(ctx, "qwen3-asr.text.intermediate_size", 3072);
    gguf_set_val_u32(ctx, "qwen3-asr.text.num_hidden_layers", 28);
    gguf_set_val_u32(ctx, "qwen3-asr.text.num_attention_heads", 16);
    gguf_set_val_u32(ctx, "qwen3-asr.text.num_key_value_heads", 8);
    gguf_set_val_u32(ctx, "qwen3-asr.text.head_dim", 128);
    gguf_set_val_f32(ctx, "qwen3-asr.text.rope_theta", 1000000.0f);
    gguf_set_val_f32(ctx, "qwen3-asr.text.rms_norm_eps", 0.000001f);
    gguf_set_val_u32(ctx, "qwen3-asr.token.audio_token_id", 151676);
    gguf_set_val_u32(ctx, "qwen3-asr.token.audio_start_token_id", 151669);
    gguf_set_val_u32(ctx, "qwen3-asr.token.audio_end_token_id", 151670);

    const bool ok = gguf_write_to_file(ctx, path, true);
    gguf_free(ctx);
    if (!ok && error) {
        *error = std::string("failed to write GGUF fixture: ") + path;
    }
    return ok;
}
