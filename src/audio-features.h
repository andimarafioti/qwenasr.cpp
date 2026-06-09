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

bool qwenasr_read_wav_16k_mono(
    const char * path,
    std::vector<float> * samples,
    int * sample_rate,
    std::string * error);

bool qwenasr_extract_whisper_features(
    const std::vector<float> & samples,
    QwenAsrFeatures * out,
    std::string * error);
