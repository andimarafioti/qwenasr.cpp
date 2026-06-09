#pragma once

#include "decoder-input.h"
#include "gguf-model.h"

#include <string>
#include <vector>

struct QwenAsrTextLayerOutput {
    int tokens = 0;
    int hidden = 0;
    std::vector<float> values; // [token, hidden]
};

struct QwenAsrTextPrefillOutput {
    int tokens = 0;
    int hidden = 0;
    int vocab = 0;
    std::vector<float> logits; // [vocab], next-token logits from the last prompt position
};

struct QwenAsrTextGenerateOutput {
    int prompt_tokens = 0;
    int total_tokens = 0;
    int hidden = 0;
    int vocab = 0;
    bool stopped = false;
    std::vector<int32_t> generated_ids;
};

struct QwenAsrTextLayerBackend;
struct QwenAsrTextDecoderBackend;

bool qwenasr_text_layer_forward_cpu(
    const QwenAsrGgufModel & model,
    const QwenAsrDecoderInputOutput & input,
    int layer,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int intermediate,
    float rope_theta,
    float rms_norm_eps,
    QwenAsrTextLayerOutput * out,
    std::string * error);

bool qwenasr_text_layer_forward_ggml(
    const QwenAsrGgufModel & model,
    const QwenAsrDecoderInputOutput & input,
    int layer,
    int n_threads,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int intermediate,
    float rope_theta,
    float rms_norm_eps,
    QwenAsrTextLayerOutput * out,
    std::string * error);

bool qwenasr_text_layer_backend_init(
    const QwenAsrGgufModel & model,
    int layer,
    int n_threads,
    int hidden,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int intermediate,
    float rope_theta,
    float rms_norm_eps,
    QwenAsrTextLayerBackend ** out,
    std::string * error);

void qwenasr_text_layer_backend_free(QwenAsrTextLayerBackend * backend);

bool qwenasr_text_layer_backend_forward(
    QwenAsrTextLayerBackend * backend,
    const QwenAsrDecoderInputOutput & input,
    QwenAsrTextLayerOutput * out,
    std::string * error);

bool qwenasr_text_decoder_backend_init(
    const QwenAsrGgufModel & model,
    int n_threads,
    int n_layers,
    int hidden,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int intermediate,
    int vocab,
    float rope_theta,
    float rms_norm_eps,
    QwenAsrTextDecoderBackend ** out,
    std::string * error);

void qwenasr_text_decoder_backend_free(QwenAsrTextDecoderBackend * backend);

bool qwenasr_text_prefill_backend_forward(
    QwenAsrTextDecoderBackend * backend,
    const QwenAsrDecoderInputOutput & input,
    QwenAsrTextPrefillOutput * out,
    std::string * error);

bool qwenasr_text_prefill_forward_cpu(
    const QwenAsrGgufModel & model,
    const QwenAsrDecoderInputOutput & input,
    int n_layers,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int intermediate,
    int vocab,
    float rope_theta,
    float rms_norm_eps,
    QwenAsrTextPrefillOutput * out,
    std::string * error);

bool qwenasr_text_generate_greedy_cpu(
    const QwenAsrGgufModel & model,
    const QwenAsrDecoderInputOutput & input,
    int max_new_tokens,
    int n_layers,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int intermediate,
    int vocab,
    float rope_theta,
    float rms_norm_eps,
    const std::vector<int32_t> & stop_ids,
    QwenAsrTextGenerateOutput * out,
    std::string * error);

bool qwenasr_text_generate_cached_cpu(
    const QwenAsrGgufModel & model,
    const QwenAsrDecoderInputOutput & input,
    int max_new_tokens,
    int n_layers,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int intermediate,
    int vocab,
    float rope_theta,
    float rms_norm_eps,
    const std::vector<int32_t> & stop_ids,
    QwenAsrTextGenerateOutput * out,
    std::string * error);
