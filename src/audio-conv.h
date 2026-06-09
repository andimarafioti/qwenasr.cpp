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

struct QwenAsrAudioCnnOutput {
    int chunks = 0;
    int frames = 0;
    int hidden = 0;
    std::vector<int> chunk_input_lengths;
    std::vector<int> chunk_output_lengths;
    std::vector<float> values; // [chunk, frame, hidden]
};

struct QwenAsrAudioPrepOutput {
    int tokens = 0;
    int hidden = 0;
    std::vector<QwenAsrAudioSegment> attention_segments;
    std::vector<float> values; // [token, hidden]
};

bool qwenasr_audio_conv0_forward(
    const QwenAsrGgufModel & model,
    const QwenAsrFeatures & features,
    QwenAsrConv2dOutput * out,
    std::string * error);

bool qwenasr_audio_conv0_forward_ggml(
    const QwenAsrGgufModel & model,
    const QwenAsrFeatures & features,
    int n_threads,
    QwenAsrConv2dOutput * out,
    std::string * error);

bool qwenasr_audio_cnn_forward_ggml(
    const QwenAsrGgufModel & model,
    const QwenAsrFeatures & features,
    int n_threads,
    QwenAsrAudioCnnOutput * out,
    std::string * error);

bool qwenasr_audio_prep_forward_ggml(
    const QwenAsrGgufModel & model,
    const QwenAsrFeatures & features,
    int n_threads,
    QwenAsrAudioPrepOutput * out,
    std::string * error);
