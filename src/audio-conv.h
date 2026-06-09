#pragma once

#include "audio-features.h"
#include "gguf-model.h"

#include <string>
#include <vector>

struct QwenAsrConv2dOutput {
    int chunks = 0;
    int channels = 0;
    int freq = 0;
    int frames = 0;
    std::vector<int> chunk_input_lengths;
    std::vector<int> chunk_output_lengths;
    std::vector<float> values; // [chunk, channel, freq, frame]
};

bool qwenasr_audio_conv0_forward(
    const QwenAsrGgufModel & model,
    const QwenAsrFeatures & features,
    QwenAsrConv2dOutput * out,
    std::string * error);
