#pragma once

#include "audio-conv.h"
#include "audio-features.h"
#include "gguf-model.h"

#include <string>
#include <vector>

struct QwenAsrAudioLayerOutput {
    int tokens = 0;
    int hidden = 0;
    std::vector<QwenAsrAudioSegment> attention_segments;
    std::vector<float> values; // [token, hidden]
};

bool qwenasr_audio_layer0_forward_cpu(
    const QwenAsrGgufModel & model,
    const QwenAsrFeatures & features,
    int n_threads,
    int n_heads,
    QwenAsrAudioLayerOutput * out,
    std::string * error);
