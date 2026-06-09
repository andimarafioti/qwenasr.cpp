#include "gguf-model.h"

#include "gguf.h"

#include <cerrno>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>

#ifdef _WIN32
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

static void set_error(std::string * error, const std::string & message) {
    if (error) {
        *error = message;
    }
}

#ifdef _WIN32
static bool map_file(const char * path, QwenAsrGgufModel * model, std::string * error) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) {
        set_error(error, std::string("failed to open GGUF file: ") + path);
        return false;
    }
    const std::streamoff end = in.tellg();
    if (end <= 0) {
        set_error(error, std::string("GGUF file is empty: ") + path);
        return false;
    }
    model->file_size = static_cast<size_t>(end);
    model->owned_data.resize(model->file_size);
    in.seekg(0);
    in.read(reinterpret_cast<char *>(model->owned_data.data()), static_cast<std::streamsize>(model->owned_data.size()));
    if (!in) {
        set_error(error, std::string("failed to read GGUF file: ") + path);
        return false;
    }
    model->mapping = model->owned_data.data();
    return true;
}

static void unmap_file(QwenAsrGgufModel * model) {
    model->owned_data.clear();
    model->mapping = nullptr;
    model->file_size = 0;
}
#else
static bool map_file(const char * path, QwenAsrGgufModel * model, std::string * error) {
    model->fd = open(path, O_RDONLY);
    if (model->fd < 0) {
        set_error(error, std::string("failed to open GGUF file: ") + path + ": " + std::strerror(errno));
        return false;
    }

    struct stat st {};
    if (fstat(model->fd, &st) != 0) {
        set_error(error, std::string("failed to stat GGUF file: ") + path + ": " + std::strerror(errno));
        close(model->fd);
        model->fd = -1;
        return false;
    }
    if (st.st_size <= 0) {
        set_error(error, std::string("GGUF file is empty: ") + path);
        close(model->fd);
        model->fd = -1;
        return false;
    }

    model->file_size = static_cast<size_t>(st.st_size);
    void * ptr = mmap(nullptr, model->file_size, PROT_READ, MAP_PRIVATE, model->fd, 0);
    if (ptr == MAP_FAILED) {
        set_error(error, std::string("failed to mmap GGUF file: ") + path + ": " + std::strerror(errno));
        close(model->fd);
        model->fd = -1;
        model->file_size = 0;
        return false;
    }
    model->mapping = static_cast<const uint8_t *>(ptr);
    return true;
}

static void unmap_file(QwenAsrGgufModel * model) {
    if (model->mapping) {
        munmap(const_cast<uint8_t *>(model->mapping), model->file_size);
    }
    if (model->fd >= 0) {
        close(model->fd);
    }
    model->mapping = nullptr;
    model->file_size = 0;
    model->fd = -1;
}
#endif

static bool checked_add(size_t a, size_t b, size_t * out) {
    if (a > std::numeric_limits<size_t>::max() - b) {
        return false;
    }
    *out = a + b;
    return true;
}

static bool tensor_view_by_index(
    const QwenAsrGgufModel & model,
    int64_t index,
    QwenAsrGgufTensorView * out,
    std::string * error) {
    if (!model.gguf || !model.meta || !model.mapping) {
        set_error(error, "GGUF model is not open");
        return false;
    }
    if (index < 0 || index >= gguf_get_n_tensors(model.gguf)) {
        set_error(error, "GGUF tensor index is out of range");
        return false;
    }

    const char * name = gguf_get_tensor_name(model.gguf, index);
    if (!name) {
        set_error(error, "GGUF tensor has no name");
        return false;
    }
    ggml_tensor * tensor = ggml_get_tensor(model.meta, name);
    if (!tensor) {
        set_error(error, std::string("missing tensor metadata: ") + name);
        return false;
    }

    const size_t tensor_offset = gguf_get_tensor_offset(model.gguf, index);
    const size_t nbytes = ggml_nbytes(tensor);
    size_t start = 0;
    size_t end = 0;
    if (!checked_add(model.data_offset, tensor_offset, &start) || !checked_add(start, nbytes, &end)) {
        set_error(error, std::string("tensor offset overflow: ") + name);
        return false;
    }
    if (end > model.file_size) {
        set_error(error, std::string("tensor data extends beyond GGUF file: ") + name);
        return false;
    }

    if (out) {
        out->name = name;
        out->type = tensor->type;
        out->ne.clear();
        const int n_dims = ggml_n_dims(tensor);
        out->ne.reserve(static_cast<size_t>(n_dims));
        for (int i = 0; i < n_dims; ++i) {
            out->ne.push_back(tensor->ne[i]);
        }
        out->nbytes = nbytes;
        out->offset = tensor_offset;
        out->data = model.mapping + start;
    }
    return true;
}

bool qwenasr_gguf_model_open(const char * path, QwenAsrGgufModel * model, std::string * error) {
    if (!path || !model) {
        set_error(error, "qwenasr_gguf_model_open: path or model is null");
        return false;
    }

    qwenasr_gguf_model_close(model);
    if (!map_file(path, model, error)) {
        return false;
    }

    gguf_init_params params {};
    params.no_alloc = true;
    params.ctx = &model->meta;
    model->gguf = gguf_init_from_file(path, params);
    if (!model->gguf) {
        qwenasr_gguf_model_close(model);
        set_error(error, std::string("failed to parse GGUF file: ") + path);
        return false;
    }
    model->data_offset = gguf_get_data_offset(model->gguf);

    const int64_t n_tensors = gguf_get_n_tensors(model->gguf);
    if (model->data_offset > model->file_size && n_tensors > 0) {
        qwenasr_gguf_model_close(model);
        set_error(error, "GGUF data offset is beyond file size");
        return false;
    }

    for (int64_t i = 0; i < n_tensors; ++i) {
        if (!tensor_view_by_index(*model, i, nullptr, error)) {
            qwenasr_gguf_model_close(model);
            return false;
        }
    }
    return true;
}

void qwenasr_gguf_model_close(QwenAsrGgufModel * model) {
    if (!model) {
        return;
    }
    if (model->meta) {
        ggml_free(model->meta);
    }
    if (model->gguf) {
        gguf_free(model->gguf);
    }
    unmap_file(model);
    model->meta = nullptr;
    model->gguf = nullptr;
    model->data_offset = 0;
}

int64_t qwenasr_gguf_model_tensor_count(const QwenAsrGgufModel & model) {
    return model.gguf ? gguf_get_n_tensors(model.gguf) : 0;
}

bool qwenasr_gguf_model_tensor_by_index(
    const QwenAsrGgufModel & model,
    int64_t index,
    QwenAsrGgufTensorView * out,
    std::string * error) {
    return tensor_view_by_index(model, index, out, error);
}

bool qwenasr_gguf_model_tensor_by_name(
    const QwenAsrGgufModel & model,
    const char * name,
    QwenAsrGgufTensorView * out,
    std::string * error) {
    if (!model.gguf || !name) {
        set_error(error, "GGUF model is not open or tensor name is null");
        return false;
    }
    const int64_t index = gguf_find_tensor(model.gguf, name);
    if (index < 0) {
        set_error(error, std::string("missing GGUF tensor: ") + name);
        return false;
    }
    return tensor_view_by_index(model, index, out, error);
}
