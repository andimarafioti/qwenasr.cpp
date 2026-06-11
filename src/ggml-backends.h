#pragma once

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshadow"
#endif
#include "ggml-backend.h"
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

#include <string>

enum class QwenAsrGgmlDevice {
    Auto,
    Cpu,
    Gpu,
};

struct QwenAsrGgmlBackendSet {
    ggml_backend_t primary = nullptr;
    ggml_backend_t fallback = nullptr;
};

QwenAsrGgmlDevice qwenasr_ggml_device_auto();
QwenAsrGgmlDevice qwenasr_ggml_device_cpu();
QwenAsrGgmlDevice qwenasr_ggml_device_gpu();

bool qwenasr_ggml_device_from_string(
    const std::string & value,
    QwenAsrGgmlDevice * out,
    std::string * error);

const char * qwenasr_ggml_device_name(QwenAsrGgmlDevice device);

bool qwenasr_ggml_flash_attn_enabled();
bool qwenasr_ggml_combined_audio_enabled();

bool qwenasr_ggml_backend_set_init(
    QwenAsrGgmlDevice device,
    int n_threads,
    QwenAsrGgmlBackendSet * out,
    std::string * error);

void qwenasr_ggml_backend_set_free(QwenAsrGgmlBackendSet * backends);

int qwenasr_ggml_backend_set_count(const QwenAsrGgmlBackendSet & backends);

void qwenasr_ggml_backend_set_array(
    const QwenAsrGgmlBackendSet & backends,
    ggml_backend_t out[2]);

const char * qwenasr_ggml_backend_set_primary_name(const QwenAsrGgmlBackendSet & backends);
