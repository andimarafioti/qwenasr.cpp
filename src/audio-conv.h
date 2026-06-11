#pragma once

#include "audio-features.h"
#include "ggml-backends.h"
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

struct QwenAsrAudioPrepBackend;

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

bool qwenasr_audio_prep_backend_init(
    const QwenAsrGgufModel & model,
    int n_threads,
    int n_mels,
    QwenAsrAudioPrepBackend ** out,
    std::string * error,
    QwenAsrGgmlDevice device = qwenasr_ggml_device_auto());

void qwenasr_audio_prep_backend_free(QwenAsrAudioPrepBackend * backend);

bool qwenasr_audio_prep_backend_forward(
    QwenAsrAudioPrepBackend * backend,
    const QwenAsrFeatures & features,
    QwenAsrAudioPrepOutput * out,
    std::string * error);
