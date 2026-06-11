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

struct QwenAsrAudioEncoderOutput {
    int tokens = 0;
    int hidden = 0;
    std::vector<QwenAsrAudioSegment> attention_segments;
    std::vector<float> values; // [token, hidden]
};

struct QwenAsrAudioEncoderBackend;

bool qwenasr_audio_layer0_forward_cpu(
    const QwenAsrGgufModel & model,
    const QwenAsrFeatures & features,
    int n_threads,
    int n_heads,
    QwenAsrAudioLayerOutput * out,
    std::string * error);

bool qwenasr_audio_encoder_forward_ggml(
    const QwenAsrGgufModel & model,
    const QwenAsrFeatures & features,
    int n_threads,
    int n_layers,
    int n_heads,
    int output_dim,
    QwenAsrAudioEncoderOutput * out,
    std::string * error);

bool qwenasr_audio_layer0_forward_ggml(
    const QwenAsrGgufModel & model,
    const QwenAsrFeatures & features,
    int n_threads,
    int n_heads,
    QwenAsrAudioLayerOutput * out,
    std::string * error);

bool qwenasr_audio_encoder_backend_init(
    const QwenAsrGgufModel & model,
    int n_threads,
    int n_layers,
    int n_heads,
    int hidden,
    int output_dim,
    QwenAsrAudioEncoderBackend ** out,
    std::string * error,
    QwenAsrGgmlDevice device = qwenasr_ggml_device_auto(),
    int n_mels = 0);

void qwenasr_audio_encoder_backend_free(QwenAsrAudioEncoderBackend * backend);

bool qwenasr_audio_encoder_backend_forward(
    QwenAsrAudioEncoderBackend * backend,
    const QwenAsrAudioPrepOutput & prep,
    QwenAsrAudioEncoderOutput * out,
    std::string * error);

bool qwenasr_audio_encoder_backend_forward_features(
    QwenAsrAudioEncoderBackend * backend,
    const QwenAsrFeatures & features,
    QwenAsrAudioEncoderOutput * out,
    std::string * error);
