#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct QwenAsrFeatures {
    int sample_rate = 16000;
    int n_samples = 0;
    int n_mels = 128;
    int n_frames = 0;
    std::vector<float> values; // [n_mels, n_frames], row-major by mel bin
};

struct QwenAsrAudioSegment {
    int offset = 0;
    int length = 0;
};

struct QwenAsrAudioGeometry {
    int feature_len = 0;
    int chunk_window = 100;
    int n_chunks = 0;
    int audio_tokens = 0;
    int max_chunk_input_len = 0;
    int max_chunk_output_len = 0;
    int attention_window = 0;
    std::vector<int> chunk_input_lengths;
    std::vector<int> chunk_output_lengths;
    std::vector<QwenAsrAudioSegment> attention_segments;
};

bool qwenasr_read_wav_16k_mono(
    const char * path,
    std::vector<float> * samples,
    int * sample_rate,
    std::string * error);

bool qwenasr_extract_whisper_features(
    const std::vector<float> & samples,
    QwenAsrFeatures * out,
    std::string * error);

int qwenasr_audio_output_length(int input_frames);

QwenAsrAudioGeometry qwenasr_audio_geometry(
    int feature_len,
    int n_window = 50,
    int n_window_infer = 800);
