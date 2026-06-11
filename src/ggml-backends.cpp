#include "ggml-backends.h"

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshadow"
#endif
#include "ggml-cpu.h"
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

#include <algorithm>
#include <cctype>
#include <cstdlib>

static void set_error(std::string * error, const std::string & message) {
    if (error) {
        *error = message;
    }
}

QwenAsrGgmlDevice qwenasr_ggml_device_auto() {
    return QwenAsrGgmlDevice::Auto;
}

QwenAsrGgmlDevice qwenasr_ggml_device_cpu() {
    return QwenAsrGgmlDevice::Cpu;
}

QwenAsrGgmlDevice qwenasr_ggml_device_gpu() {
    return QwenAsrGgmlDevice::Gpu;
}

static std::string lower_ascii(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

bool qwenasr_ggml_device_from_string(
    const std::string & value,
    QwenAsrGgmlDevice * out,
    std::string * error) {
    if (!out) {
        set_error(error, "qwenasr_ggml_device_from_string: out is null");
        return false;
    }
    const std::string normalized = lower_ascii(value);
    if (normalized == "auto") {
        *out = QwenAsrGgmlDevice::Auto;
        return true;
    }
    if (normalized == "cpu") {
        *out = QwenAsrGgmlDevice::Cpu;
        return true;
    }
    if (normalized == "gpu" || normalized == "cuda") {
        *out = QwenAsrGgmlDevice::Gpu;
        return true;
    }
    set_error(error, "GGML device must be auto, cpu, gpu, or cuda");
    return false;
}

const char * qwenasr_ggml_device_name(QwenAsrGgmlDevice device) {
    switch (device) {
        case QwenAsrGgmlDevice::Auto:
            return "auto";
        case QwenAsrGgmlDevice::Cpu:
            return "cpu";
        case QwenAsrGgmlDevice::Gpu:
            return "gpu";
    }
    return "unknown";
}

bool qwenasr_ggml_flash_attn_enabled() {
    const char * value = std::getenv("QWENASR_GGML_FLASH_ATTN");
    if (!value || value[0] == '\0') {
        return false;
    }
    const std::string normalized = lower_ascii(value);
    return normalized != "0" &&
        normalized != "false" &&
        normalized != "off" &&
        normalized != "no";
}

bool qwenasr_ggml_combined_audio_enabled() {
    const char * value = std::getenv("QWENASR_GGML_COMBINED_AUDIO");
    if (!value || value[0] == '\0') {
        return true;
    }
    const std::string normalized = lower_ascii(value);
    return normalized != "0" &&
        normalized != "false" &&
        normalized != "off" &&
        normalized != "no";
}

static bool backend_is_cpu(ggml_backend_t backend) {
    ggml_backend_dev_t dev = backend ? ggml_backend_get_device(backend) : nullptr;
    return dev && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU;
}

static void maybe_set_cpu_threads(ggml_backend_t backend, int n_threads) {
    if (backend_is_cpu(backend)) {
        ggml_backend_cpu_set_n_threads(backend, n_threads);
    }
}

static ggml_backend_t init_cpu_backend(int n_threads) {
    ggml_backend_t backend = ggml_backend_cpu_init();
    maybe_set_cpu_threads(backend, n_threads);
    return backend;
}

static ggml_backend_t init_gpu_backend() {
    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    if (!backend) {
        backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU, nullptr);
    }
    return backend;
}

bool qwenasr_ggml_backend_set_init(
    QwenAsrGgmlDevice device,
    int n_threads,
    QwenAsrGgmlBackendSet * out,
    std::string * error) {
    if (!out) {
        set_error(error, "qwenasr_ggml_backend_set_init: out is null");
        return false;
    }
    out->primary = nullptr;
    out->fallback = nullptr;
    if (n_threads <= 0) {
        n_threads = 1;
    }

    if (device != QwenAsrGgmlDevice::Cpu) {
        out->primary = init_gpu_backend();
        if (out->primary) {
            out->fallback = init_cpu_backend(n_threads);
            if (!out->fallback) {
                qwenasr_ggml_backend_set_free(out);
                set_error(error, "failed to initialize GGML CPU fallback backend");
                return false;
            }
            return true;
        }
        if (device == QwenAsrGgmlDevice::Gpu) {
            set_error(error, "failed to initialize requested GGML GPU backend; rebuild with GGML_CUDA=ON and verify a CUDA device is visible");
            return false;
        }
    }

    out->primary = init_cpu_backend(n_threads);
    if (!out->primary) {
        set_error(error, "failed to initialize GGML CPU backend");
        return false;
    }
    return true;
}

void qwenasr_ggml_backend_set_free(QwenAsrGgmlBackendSet * backends) {
    if (!backends) {
        return;
    }
    if (backends->fallback) {
        ggml_backend_free(backends->fallback);
        backends->fallback = nullptr;
    }
    if (backends->primary) {
        ggml_backend_free(backends->primary);
        backends->primary = nullptr;
    }
}

int qwenasr_ggml_backend_set_count(const QwenAsrGgmlBackendSet & backends) {
    int count = 0;
    if (backends.primary) {
        ++count;
    }
    if (backends.fallback) {
        ++count;
    }
    return count;
}

void qwenasr_ggml_backend_set_array(
    const QwenAsrGgmlBackendSet & backends,
    ggml_backend_t out[2]) {
    out[0] = backends.primary;
    out[1] = backends.fallback;
}

const char * qwenasr_ggml_backend_set_primary_name(const QwenAsrGgmlBackendSet & backends) {
    return backends.primary ? ggml_backend_name(backends.primary) : "none";
}
