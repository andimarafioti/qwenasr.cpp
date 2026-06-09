#include "native-model.h"

#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cstdio>
#include <initializer_list>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

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

static std::string shape_string(const int64_t * ne, int n_dims) {
    std::ostringstream ss;
    ss << "[";
    for (int i = 0; i < n_dims; ++i) {
        if (i > 0) {
            ss << ", ";
        }
        ss << ne[i];
    }
    ss << "]";
    return ss.str();
}

static std::string shape_string(const std::vector<int64_t> & ne) {
    return shape_string(ne.data(), static_cast<int>(ne.size()));
}

static bool require_tensor(
    const gguf_context * ctx,
    ggml_context * meta,
    const QwenAsrTensorSpec & spec,
    std::string * error) {
    if (gguf_find_tensor(ctx, spec.name.c_str()) < 0) {
        if (error) {
            *error = std::string("missing GGUF tensor: ") + spec.name;
        }
        return false;
    }

    ggml_tensor * tensor = meta ? ggml_get_tensor(meta, spec.name.c_str()) : nullptr;
    if (!tensor) {
        if (error) {
            *error = std::string("missing GGML tensor metadata: ") + spec.name;
        }
        return false;
    }

    const int n_dims = ggml_n_dims(tensor);
    if (n_dims != static_cast<int>(spec.ne.size())) {
        if (error) {
            *error = "GGUF tensor rank mismatch for " + spec.name +
                ": expected " + shape_string(spec.ne) +
                " got " + shape_string(tensor->ne, n_dims);
        }
        return false;
    }

    for (int i = 0; i < n_dims; ++i) {
        if (tensor->ne[i] != spec.ne[static_cast<size_t>(i)]) {
            if (error) {
                *error = "GGUF tensor shape mismatch for " + spec.name +
                    ": expected " + shape_string(spec.ne) +
                    " got " + shape_string(tensor->ne, n_dims);
            }
            return false;
        }
    }

    return true;
}

static void add_spec(
    std::vector<QwenAsrTensorSpec> & specs,
    std::string name,
    std::initializer_list<int64_t> ne) {
    specs.push_back({ std::move(name), std::vector<int64_t>(ne) });
}

static int64_t audio_conv_freq_bins(uint32_t mel_bins) {
    int64_t bins = mel_bins;
    bins = (bins + 1) / 2;
    bins = (bins + 1) / 2;
    bins = (bins + 1) / 2;
    return bins;
}

bool qwenasr_load_gguf_metadata(
    const char * path,
    bool require_tensors,
    QwenAsrNativeConfig * out,
    std::string * error) {
    if (!path || !out) {
        if (error) {
            *error = "qwenasr_load_gguf_metadata: path or out is null";
        }
        return false;
    }

    ggml_context * meta = nullptr;
    gguf_init_params params {};
    params.no_alloc = true;
    params.ctx = &meta;
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

    if (ok && require_tensors) {
        const std::vector<QwenAsrTensorSpec> expected = qwenasr_expected_tensor_specs(cfg);
        for (const QwenAsrTensorSpec & spec : expected) {
            ok = require_tensor(ctx, meta, spec, error);
            if (!ok) {
                break;
            }
        }
    }

    if (ok) {
        const int64_t limit = std::min<int64_t>(cfg.n_tensors, 12);
        for (int64_t i = 0; i < limit; ++i) {
            const char * name = gguf_get_tensor_name(ctx, i);
            cfg.first_tensors.push_back(name ? name : "");
        }
        *out = cfg;
    }

    if (meta) {
        ggml_free(meta);
    }
    gguf_free(ctx);
    return ok;
}

std::vector<QwenAsrTensorSpec> qwenasr_expected_tensor_specs(const QwenAsrNativeConfig & cfg) {
    std::vector<QwenAsrTensorSpec> specs;
    specs.reserve(
        3 + 13 +
        static_cast<size_t>(cfg.text_num_hidden_layers) * 11 +
        static_cast<size_t>(cfg.audio_encoder_layers) * 16);

    const int64_t text_hidden = cfg.text_hidden_size;
    const int64_t text_q_dim = static_cast<int64_t>(cfg.text_num_attention_heads) * cfg.text_head_dim;
    const int64_t text_kv_dim = static_cast<int64_t>(cfg.text_num_key_value_heads) * cfg.text_head_dim;
    const int64_t text_intermediate = cfg.text_intermediate_size;
    const int64_t text_vocab = cfg.text_vocab_size;
    const int64_t audio_hidden = cfg.audio_d_model;
    const int64_t audio_downsample = cfg.audio_downsample_hidden_size;
    const int64_t audio_ffn = cfg.audio_encoder_ffn_dim;
    const int64_t audio_output = cfg.audio_output_dim;
    const int64_t audio_conv_out = audio_downsample * audio_conv_freq_bins(cfg.audio_num_mel_bins);

    add_spec(specs, "text.token_embd.weight", { text_hidden, text_vocab });
    add_spec(specs, "text.output_norm.weight", { text_hidden });
    add_spec(specs, "text.output.weight", { text_hidden, text_vocab });

    add_spec(specs, "audio.conv.0.weight", { 3, 3, 1, audio_downsample });
    add_spec(specs, "audio.conv.0.bias", { audio_downsample });
    add_spec(specs, "audio.conv.1.weight", { 3, 3, audio_downsample, audio_downsample });
    add_spec(specs, "audio.conv.1.bias", { audio_downsample });
    add_spec(specs, "audio.conv.2.weight", { 3, 3, audio_downsample, audio_downsample });
    add_spec(specs, "audio.conv.2.bias", { audio_downsample });
    add_spec(specs, "audio.conv_out.weight", { audio_conv_out, audio_hidden });
    add_spec(specs, "audio.post_norm.weight", { audio_hidden });
    add_spec(specs, "audio.post_norm.bias", { audio_hidden });
    add_spec(specs, "audio.proj.0.weight", { audio_hidden, audio_hidden });
    add_spec(specs, "audio.proj.0.bias", { audio_hidden });
    add_spec(specs, "audio.proj.1.weight", { audio_hidden, audio_output });
    add_spec(specs, "audio.proj.1.bias", { audio_output });

    for (uint32_t layer = 0; layer < cfg.text_num_hidden_layers; ++layer) {
        const std::string prefix = "text.blk." + std::to_string(layer) + ".";
        add_spec(specs, prefix + "attn_norm.weight", { text_hidden });
        add_spec(specs, prefix + "ffn_norm.weight", { text_hidden });
        add_spec(specs, prefix + "attn_q.weight", { text_hidden, text_q_dim });
        add_spec(specs, prefix + "attn_k.weight", { text_hidden, text_kv_dim });
        add_spec(specs, prefix + "attn_v.weight", { text_hidden, text_kv_dim });
        add_spec(specs, prefix + "attn_output.weight", { text_q_dim, text_hidden });
        add_spec(specs, prefix + "attn_q_norm.weight", { cfg.text_head_dim });
        add_spec(specs, prefix + "attn_k_norm.weight", { cfg.text_head_dim });
        add_spec(specs, prefix + "ffn_gate.weight", { text_hidden, text_intermediate });
        add_spec(specs, prefix + "ffn_up.weight", { text_hidden, text_intermediate });
        add_spec(specs, prefix + "ffn_down.weight", { text_intermediate, text_hidden });
    }

    for (uint32_t layer = 0; layer < cfg.audio_encoder_layers; ++layer) {
        const std::string prefix = "audio.blk." + std::to_string(layer) + ".";
        add_spec(specs, prefix + "attn_norm.weight", { audio_hidden });
        add_spec(specs, prefix + "attn_norm.bias", { audio_hidden });
        add_spec(specs, prefix + "ffn_norm.weight", { audio_hidden });
        add_spec(specs, prefix + "ffn_norm.bias", { audio_hidden });
        add_spec(specs, prefix + "attn_q.weight", { audio_hidden, audio_hidden });
        add_spec(specs, prefix + "attn_q.bias", { audio_hidden });
        add_spec(specs, prefix + "attn_k.weight", { audio_hidden, audio_hidden });
        add_spec(specs, prefix + "attn_k.bias", { audio_hidden });
        add_spec(specs, prefix + "attn_v.weight", { audio_hidden, audio_hidden });
        add_spec(specs, prefix + "attn_v.bias", { audio_hidden });
        add_spec(specs, prefix + "attn_output.weight", { audio_hidden, audio_hidden });
        add_spec(specs, prefix + "attn_output.bias", { audio_hidden });
        add_spec(specs, prefix + "ffn_up.weight", { audio_hidden, audio_ffn });
        add_spec(specs, prefix + "ffn_up.bias", { audio_ffn });
        add_spec(specs, prefix + "ffn_down.weight", { audio_ffn, audio_hidden });
        add_spec(specs, prefix + "ffn_down.bias", { audio_hidden });
    }

    return specs;
}

std::vector<std::string> qwenasr_expected_tensor_names(const QwenAsrNativeConfig & cfg) {
    std::vector<QwenAsrTensorSpec> specs = qwenasr_expected_tensor_specs(cfg);
    std::vector<std::string> names;
    names.reserve(specs.size());
    for (const QwenAsrTensorSpec & spec : specs) {
        names.push_back(spec.name);
    }
    return names;
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
