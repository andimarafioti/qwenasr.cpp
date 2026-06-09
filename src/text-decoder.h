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
