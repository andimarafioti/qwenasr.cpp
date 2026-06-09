#include "gguf-model.h"
#include "native-model.h"

#include "gguf.h"

#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>

static void usage(const char * argv0) {
    std::cerr
        << "Usage: " << argv0 << " model.gguf [--allow-metadata-only] [--limit N] [--tensor NAME]\n"
        << "       " << argv0 << " --self-test\n"
        << "\n"
        << "Open a GGUF file through the native mapped-weight loader and inspect tensor data ranges.\n";
}

static std::string shape_string(const std::vector<int64_t> & ne) {
    std::ostringstream ss;
    ss << "[";
    for (size_t i = 0; i < ne.size(); ++i) {
        if (i > 0) {
            ss << ",";
        }
        ss << ne[i];
    }
    ss << "]";
    return ss.str();
}

static bool parse_int(const std::string & text, int * out) {
    char * end = nullptr;
    const long value = std::strtol(text.c_str(), &end, 10);
    if (!end || *end != '\0' || value < 0 || value > 1000000L) {
        return false;
    }
    *out = static_cast<int>(value);
    return true;
}

static void print_tensor(const QwenAsrGgufTensorView & tensor, const std::string & prefix) {
    std::cout << prefix << "name=" << tensor.name << "\n";
    std::cout << prefix << "type=" << ggml_type_name(tensor.type) << "\n";
    std::cout << prefix << "shape=" << shape_string(tensor.ne) << "\n";
    std::cout << prefix << "bytes=" << tensor.nbytes << "\n";
    std::cout << prefix << "offset=" << tensor.offset << "\n";
    if (tensor.type == GGML_TYPE_F32 && tensor.nbytes >= sizeof(float) && tensor.data) {
        float value = 0.0f;
        std::memcpy(&value, tensor.data, sizeof(float));
        std::cout << prefix << "first_f32=" << value << "\n";
    }
}

static bool write_tensor_fixture(const char * path, std::string * error) {
    gguf_context * gguf = gguf_init_empty();
    if (!gguf) {
        if (error) {
            *error = "gguf_init_empty failed";
        }
        return false;
    }

    ggml_init_params params {};
    params.mem_size = 16 * 1024;
    params.mem_buffer = nullptr;
    params.no_alloc = false;
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        gguf_free(gguf);
        if (error) {
            *error = "ggml_init failed";
        }
        return false;
    }

    gguf_set_val_str(gguf, "general.name", "qwenasr tensor fixture");
    ggml_tensor * tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    ggml_set_name(tensor, "fixture.weight");
    float values[4] = { 1.25f, -2.5f, 3.75f, 4.5f };
    std::memcpy(tensor->data, values, sizeof(values));
    gguf_add_tensor(gguf, tensor);
    gguf_set_tensor_data(gguf, "fixture.weight", tensor->data);

    const bool ok = gguf_write_to_file(gguf, path, false);
    ggml_free(ctx);
    gguf_free(gguf);
    if (!ok && error) {
        *error = std::string("failed to write tensor fixture: ") + path;
    }
    return ok;
}

static int self_test() {
    const char * path = "/tmp/qwenasr-weights-fixture.gguf";
    std::string error;
    if (!write_tensor_fixture(path, &error)) {
        std::cerr << error << "\n";
        return 1;
    }

    QwenAsrGgufModel model {};
    if (!qwenasr_gguf_model_open(path, &model, &error)) {
        std::remove(path);
        std::cerr << error << "\n";
        return 1;
    }

    QwenAsrGgufTensorView tensor;
    if (!qwenasr_gguf_model_tensor_by_name(model, "fixture.weight", &tensor, &error)) {
        qwenasr_gguf_model_close(&model);
        std::remove(path);
        std::cerr << error << "\n";
        return 1;
    }

    float values[4] = {};
    std::memcpy(values, tensor.data, sizeof(values));
    const bool ok =
        tensor.type == GGML_TYPE_F32 &&
        tensor.ne == std::vector<int64_t>({ 4 }) &&
        tensor.nbytes == sizeof(values) &&
        values[0] == 1.25f &&
        values[1] == -2.5f &&
        values[2] == 3.75f &&
        values[3] == 4.5f;

    std::cout << "file_size=" << model.file_size << "\n";
    std::cout << "data_offset=" << model.data_offset << "\n";
    std::cout << "tensors=" << qwenasr_gguf_model_tensor_count(model) << "\n";
    print_tensor(tensor, "tensor.0.");
    qwenasr_gguf_model_close(&model);
    std::remove(path);

    if (!ok) {
        std::cerr << "tensor fixture data mismatch\n";
        return 1;
    }
    std::cout << "status=self-test-ok\n";
    return 0;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        usage(argv[0]);
        return 2;
    }

    bool allow_metadata_only = false;
    int limit = 8;
    std::string tensor_name;
    std::string path;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            usage(argv[0]);
            return 0;
        }
        if (arg == "--self-test") {
            return self_test();
        }
        if (arg == "--allow-metadata-only") {
            allow_metadata_only = true;
            continue;
        }
        if (arg == "--limit") {
            if (++i >= argc || !parse_int(argv[i], &limit)) {
                std::cerr << "--limit requires a non-negative integer\n";
                return 2;
            }
            continue;
        }
        if (arg == "--tensor") {
            if (++i >= argc) {
                std::cerr << "--tensor requires a tensor name\n";
                return 2;
            }
            tensor_name = argv[i];
            continue;
        }
        if (!path.empty()) {
            std::cerr << "only one GGUF path is supported\n";
            return 2;
        }
        path = arg;
    }

    if (path.empty()) {
        std::cerr << "GGUF path is required\n";
        return 2;
    }

    QwenAsrNativeConfig cfg;
    std::string error;
    if (!qwenasr_load_gguf_metadata(path.c_str(), !allow_metadata_only, &cfg, &error)) {
        std::cerr << error << "\n";
        return 1;
    }

    QwenAsrGgufModel model {};
    if (!qwenasr_gguf_model_open(path.c_str(), &model, &error)) {
        std::cerr << error << "\n";
        return 1;
    }

    std::cout << "file_size=" << model.file_size << "\n";
    std::cout << "data_offset=" << model.data_offset << "\n";
    std::cout << "tensors=" << qwenasr_gguf_model_tensor_count(model) << "\n";
    std::cout << "expected_tensors=" << qwenasr_expected_tensor_names(cfg).size() << "\n";

    if (!tensor_name.empty()) {
        QwenAsrGgufTensorView tensor;
        if (!qwenasr_gguf_model_tensor_by_name(model, tensor_name.c_str(), &tensor, &error)) {
            qwenasr_gguf_model_close(&model);
            std::cerr << error << "\n";
            return 1;
        }
        print_tensor(tensor, "tensor.");
    } else {
        const int64_t n = std::min<int64_t>(qwenasr_gguf_model_tensor_count(model), limit);
        for (int64_t i = 0; i < n; ++i) {
            QwenAsrGgufTensorView tensor;
            if (!qwenasr_gguf_model_tensor_by_index(model, i, &tensor, &error)) {
                qwenasr_gguf_model_close(&model);
                std::cerr << error << "\n";
                return 1;
            }
            print_tensor(tensor, "tensor." + std::to_string(i) + ".");
        }
    }

    qwenasr_gguf_model_close(&model);
    std::cout << "status=ok\n";
    return 0;
}
