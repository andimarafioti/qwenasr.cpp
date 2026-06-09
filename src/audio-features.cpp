#include "audio-features.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

static uint16_t read_u16_le(std::istream & in) {
    unsigned char b[2] = { 0, 0 };
    in.read(reinterpret_cast<char *>(b), 2);
    return static_cast<uint16_t>(b[0] | (b[1] << 8));
}

static uint32_t read_u32_le(std::istream & in) {
    unsigned char b[4] = { 0, 0, 0, 0 };
    in.read(reinterpret_cast<char *>(b), 4);
    return static_cast<uint32_t>(b[0] | (b[1] << 8) | (b[2] << 16) | (b[3] << 24));
}

static int32_t read_s24_le(const unsigned char * p) {
    uint32_t value = static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
                     (static_cast<uint32_t>(p[2]) << 16);
    if (value & 0x00800000U) {
        value |= 0xff000000U;
    }
    return static_cast<int32_t>(value);
}

static float read_f32_le(const unsigned char * p) {
    uint32_t bits = static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
                    (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24);
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(float));
    return value;
}

bool qwenasr_read_wav_16k_mono(
    const char * path,
    std::vector<float> * samples,
    int * sample_rate,
    std::string * error) {
    if (!path || !samples || !sample_rate) {
        if (error) {
            *error = "qwenasr_read_wav_16k_mono: path, samples, or sample_rate is null";
        }
        return false;
    }

    std::ifstream in(path, std::ios::binary);
    if (!in) {
        if (error) {
            *error = std::string("failed to open WAV: ") + path;
        }
        return false;
    }

    char riff[4] = {};
    char wave[4] = {};
    in.read(riff, 4);
    (void) read_u32_le(in);
    in.read(wave, 4);
    if (std::strncmp(riff, "RIFF", 4) != 0 || std::strncmp(wave, "WAVE", 4) != 0) {
        if (error) {
            *error = "not a RIFF/WAVE file";
        }
        return false;
    }

    uint16_t format = 0;
    uint16_t channels = 0;
    uint32_t sr = 0;
    uint16_t bits_per_sample = 0;
    std::vector<unsigned char> data;

    while (in) {
        char id[4] = {};
        in.read(id, 4);
        if (!in) {
            break;
        }
        uint32_t size = read_u32_le(in);
        std::streampos start = in.tellg();

        if (std::strncmp(id, "fmt ", 4) == 0 && size >= 16) {
            format = read_u16_le(in);
            channels = read_u16_le(in);
            sr = read_u32_le(in);
            (void) read_u32_le(in);
            (void) read_u16_le(in);
            bits_per_sample = read_u16_le(in);
            if (format == 0xfffe && size >= 40) {
                in.seekg(start + static_cast<std::streamoff>(24));
                uint16_t sub_format = read_u16_le(in);
                if (sub_format == 1 || sub_format == 3) {
                    format = sub_format;
                }
            }
        } else if (std::strncmp(id, "data", 4) == 0) {
            data.resize(size);
            in.read(reinterpret_cast<char *>(data.data()), static_cast<std::streamsize>(size));
        }

        in.seekg(start + static_cast<std::streamoff>(size + (size & 1U)));
    }

    if (sr != 16000) {
        if (error) {
            *error = "native feature extractor currently requires 16 kHz WAV input";
        }
        return false;
    }
    if (channels == 0 || data.empty()) {
        if (error) {
            *error = "WAV has no decoded audio data";
        }
        return false;
    }

    const size_t bytes_per_sample = static_cast<size_t>(bits_per_sample / 8);
    if (bytes_per_sample == 0) {
        if (error) {
            *error = "invalid WAV bits_per_sample";
        }
        return false;
    }
    const size_t n_frames = data.size() / (static_cast<size_t>(channels) * bytes_per_sample);
    samples->assign(n_frames, 0.0f);

    for (size_t t = 0; t < n_frames; ++t) {
        double mono = 0.0;
        for (uint16_t ch = 0; ch < channels; ++ch) {
            const unsigned char * p = data.data() + (t * channels + ch) * bytes_per_sample;
            float value = 0.0f;
            if (format == 1 && bits_per_sample == 16) {
                int16_t s = static_cast<int16_t>(static_cast<uint16_t>(p[0] | (p[1] << 8)));
                value = static_cast<float>(s) / 32768.0f;
            } else if (format == 1 && bits_per_sample == 24) {
                value = static_cast<float>(read_s24_le(p)) / 8388608.0f;
            } else if (format == 3 && bits_per_sample == 32) {
                value = read_f32_le(p);
            } else {
                if (error) {
                    *error = "unsupported WAV encoding; expected PCM16, PCM24, or F32";
                }
                return false;
            }
            mono += static_cast<double>(value);
        }
        (*samples)[t] = static_cast<float>(mono / static_cast<double>(channels));
    }

    *sample_rate = static_cast<int>(sr);
    return true;
}

static double hz_to_mel_slaney(double hz) {
    const double min_log_hz = 1000.0;
    const double min_log_mel = 15.0;
    const double logstep = 27.0 / std::log(6.4);
    double mel = 3.0 * hz / 200.0;
    if (hz >= min_log_hz) {
        mel = min_log_mel + std::log(hz / min_log_hz) * logstep;
    }
    return mel;
}

static double mel_to_hz_slaney(double mel) {
    const double min_log_hz = 1000.0;
    const double min_log_mel = 15.0;
    const double logstep = std::log(6.4) / 27.0;
    double hz = 200.0 * mel / 3.0;
    if (mel >= min_log_mel) {
        hz = min_log_hz * std::exp(logstep * (mel - min_log_mel));
    }
    return hz;
}

static std::vector<double> make_hann(int n_fft) {
    std::vector<double> window(static_cast<size_t>(n_fft));
    for (int i = 0; i < n_fft; ++i) {
        window[static_cast<size_t>(i)] = 0.5 - 0.5 * std::cos(2.0 * M_PI * static_cast<double>(i) / n_fft);
    }
    return window;
}

static std::vector<double> make_mel_filters(int n_freq, int n_mels, int sample_rate) {
    const double mel_min = hz_to_mel_slaney(0.0);
    const double mel_max = hz_to_mel_slaney(8000.0);

    std::vector<double> filter_freqs(static_cast<size_t>(n_mels + 2));
    for (int i = 0; i < n_mels + 2; ++i) {
        const double mel = mel_min + (mel_max - mel_min) * static_cast<double>(i) / static_cast<double>(n_mels + 1);
        filter_freqs[static_cast<size_t>(i)] = mel_to_hz_slaney(mel);
    }

    std::vector<double> fft_freqs(static_cast<size_t>(n_freq));
    for (int i = 0; i < n_freq; ++i) {
        fft_freqs[static_cast<size_t>(i)] = static_cast<double>(i) * (sample_rate / 2.0) / static_cast<double>(n_freq - 1);
    }

    std::vector<double> filters(static_cast<size_t>(n_freq * n_mels), 0.0);
    for (int m = 0; m < n_mels; ++m) {
        const double lower = filter_freqs[static_cast<size_t>(m)];
        const double center = filter_freqs[static_cast<size_t>(m + 1)];
        const double upper = filter_freqs[static_cast<size_t>(m + 2)];
        const double enorm = 2.0 / (upper - lower);
        for (int f = 0; f < n_freq; ++f) {
            const double freq = fft_freqs[static_cast<size_t>(f)];
            const double up = (freq - lower) / (center - lower);
            const double down = (upper - freq) / (upper - center);
            double value = std::min(up, down);
            if (value < 0.0) {
                value = 0.0;
            }
            filters[static_cast<size_t>(f * n_mels + m)] = value * enorm;
        }
    }
    return filters;
}

static int reflect_index(int index, int n) {
    if (n <= 1) {
        return 0;
    }
    while (index < 0 || index >= n) {
        if (index < 0) {
            index = -index;
        }
        if (index >= n) {
            index = 2 * n - 2 - index;
        }
    }
    return index;
}

bool qwenasr_extract_whisper_features(
    const std::vector<float> & samples,
    QwenAsrFeatures * out,
    std::string * error) {
    if (!out) {
        if (error) {
            *error = "qwenasr_extract_whisper_features: out is null";
        }
        return false;
    }
    if (samples.empty()) {
        if (error) {
            *error = "cannot extract features from empty audio";
        }
        return false;
    }

    constexpr int sample_rate = 16000;
    constexpr int n_fft = 400;
    constexpr int hop = 160;
    constexpr int n_mels = 128;
    constexpr int n_freq = n_fft / 2 + 1;

    const int n_samples = static_cast<int>(samples.size());
    const int stft_frames = 1 + n_samples / hop;
    const int n_frames = stft_frames - 1;
    if (n_frames <= 0) {
        if (error) {
            *error = "audio is too short for Whisper feature extraction";
        }
        return false;
    }

    const std::vector<double> hann = make_hann(n_fft);
    const std::vector<double> mel_filters = make_mel_filters(n_freq, n_mels, sample_rate);

    std::vector<double> log_spec(static_cast<size_t>(n_mels * n_frames), 0.0);
    double max_log = -std::numeric_limits<double>::infinity();

    for (int frame = 0; frame < n_frames; ++frame) {
        double power[n_freq];
        for (int k = 0; k < n_freq; ++k) {
            double real = 0.0;
            double imag = 0.0;
            for (int n = 0; n < n_fft; ++n) {
                const int sample_index = reflect_index(frame * hop + n - n_fft / 2, n_samples);
                const double x = static_cast<double>(samples[static_cast<size_t>(sample_index)]) *
                                 hann[static_cast<size_t>(n)];
                const double angle = 2.0 * M_PI * static_cast<double>(k * n) / static_cast<double>(n_fft);
                real += x * std::cos(angle);
                imag -= x * std::sin(angle);
            }
            power[k] = real * real + imag * imag;
        }

        for (int mel = 0; mel < n_mels; ++mel) {
            double value = 0.0;
            for (int freq = 0; freq < n_freq; ++freq) {
                value += mel_filters[static_cast<size_t>(freq * n_mels + mel)] * power[freq];
            }
            value = std::max(value, 1.0e-10);
            const double logged = std::log10(value);
            log_spec[static_cast<size_t>(mel * n_frames + frame)] = logged;
            max_log = std::max(max_log, logged);
        }
    }

    out->sample_rate = sample_rate;
    out->n_samples = n_samples;
    out->n_mels = n_mels;
    out->n_frames = n_frames;
    out->values.resize(static_cast<size_t>(n_mels * n_frames));
    const double floor = max_log - 8.0;
    for (size_t i = 0; i < out->values.size(); ++i) {
        const double clamped = std::max(log_spec[i], floor);
        out->values[i] = static_cast<float>((clamped + 4.0) / 4.0);
    }

    return true;
}
