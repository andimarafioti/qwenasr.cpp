#include "audio-features.h"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>

static void usage(const char * argv0) {
    std::cerr << "Usage: " << argv0 << " audio.wav [--out features.f32]\n";
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        usage(argv[0]);
        return 2;
    }

    std::string audio;
    std::string out_path;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            usage(argv[0]);
            return 0;
        }
        if (arg == "--out") {
            if (i + 1 >= argc) {
                std::cerr << "--out requires a path\n";
                return 2;
            }
            out_path = argv[++i];
            continue;
        }
        if (!audio.empty()) {
            std::cerr << "only one audio input is supported\n";
            return 2;
        }
        audio = arg;
    }

    if (audio.empty()) {
        std::cerr << "audio input is required\n";
        return 2;
    }

    std::vector<float> samples;
    int sample_rate = 0;
    std::string error;
    if (!qwenasr_read_wav_16k_mono(audio.c_str(), &samples, &sample_rate, &error)) {
        std::cerr << error << "\n";
        return 1;
    }

    QwenAsrFeatures features;
    if (!qwenasr_extract_whisper_features(samples, &features, &error)) {
        std::cerr << error << "\n";
        return 1;
    }

    double sum = 0.0;
    float min_value = std::numeric_limits<float>::infinity();
    float max_value = -std::numeric_limits<float>::infinity();
    for (float value : features.values) {
        sum += value;
        min_value = std::min(min_value, value);
        max_value = std::max(max_value, value);
    }

    if (!out_path.empty()) {
        std::ofstream out(out_path, std::ios::binary);
        if (!out) {
            std::cerr << "failed to open output: " << out_path << "\n";
            return 1;
        }
        out.write(
            reinterpret_cast<const char *>(features.values.data()),
            static_cast<std::streamsize>(features.values.size() * sizeof(float)));
    }

    std::cout << "sample_rate=" << sample_rate << "\n";
    std::cout << "samples=" << features.n_samples << "\n";
    std::cout << "mels=" << features.n_mels << "\n";
    std::cout << "frames=" << features.n_frames << "\n";
    std::cout << "values=" << features.values.size() << "\n";
    const QwenAsrAudioGeometry geometry = qwenasr_audio_geometry(features.n_frames);
    std::cout << "audio_tokens=" << geometry.audio_tokens << "\n";
    std::cout << "feature_chunks=" << geometry.n_chunks << "\n";
    std::cout << "chunk_window=" << geometry.chunk_window << "\n";
    std::cout << "max_chunk_input_len=" << geometry.max_chunk_input_len << "\n";
    std::cout << "max_chunk_output_len=" << geometry.max_chunk_output_len << "\n";
    std::cout << "attention_window=" << geometry.attention_window << "\n";
    std::cout << "attention_segments=" << geometry.attention_segments.size() << "\n";
    std::cout << "min=" << min_value << "\n";
    std::cout << "max=" << max_value << "\n";
    std::cout << "mean=" << (sum / static_cast<double>(features.values.size())) << "\n";
    return 0;
}
