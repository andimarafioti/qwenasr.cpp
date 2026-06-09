#pragma once

#include "ggml.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct gguf_context;

struct QwenAsrGgufTensorView {
    std::string name;
    ggml_type type = GGML_TYPE_COUNT;
    std::vector<int64_t> ne;
    size_t nbytes = 0;
    size_t offset = 0;
    const void * data = nullptr;
};

struct QwenAsrGgufModel {
    gguf_context * gguf = nullptr;
    ggml_context * meta = nullptr;
    const uint8_t * mapping = nullptr;
    size_t file_size = 0;
    size_t data_offset = 0;

#ifdef _WIN32
    std::vector<uint8_t> owned_data;
#else
    int fd = -1;
#endif
};

bool qwenasr_gguf_model_open(
    const char * path,
    QwenAsrGgufModel * model,
    std::string * error);

void qwenasr_gguf_model_close(QwenAsrGgufModel * model);

int64_t qwenasr_gguf_model_tensor_count(const QwenAsrGgufModel & model);

bool qwenasr_gguf_model_tensor_by_index(
    const QwenAsrGgufModel & model,
    int64_t index,
    QwenAsrGgufTensorView * out,
    std::string * error);

bool qwenasr_gguf_model_tensor_by_name(
    const QwenAsrGgufModel & model,
    const char * name,
    QwenAsrGgufTensorView * out,
    std::string * error);
