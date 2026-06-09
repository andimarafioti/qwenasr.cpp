#include "qwenasr.h"

#include <Python.h>

#include <cstdlib>
#include <cstring>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

struct qa_context {
    PyObject *  model = nullptr;
    std::string model_id;
    std::string backend;
};

static thread_local std::string g_last_error;
static std::mutex               g_python_init_mutex;

static bool qa_is_set(const char * value) {
    return value && value[0] != '\0';
}

static void qa_set_error(const std::string & message) {
    g_last_error = message;
}

static void qa_set_python_error(const char * prefix) {
    PyObject * type  = nullptr;
    PyObject * value = nullptr;
    PyObject * trace = nullptr;
    PyErr_Fetch(&type, &value, &trace);
    PyErr_NormalizeException(&type, &value, &trace);

    std::string message = prefix ? prefix : "python error";
    if (value) {
        PyObject * s = PyObject_Str(value);
        if (s) {
            const char * utf8 = PyUnicode_AsUTF8(s);
            if (utf8) {
                message += ": ";
                message += utf8;
            }
            Py_DECREF(s);
        }
    }

    Py_XDECREF(type);
    Py_XDECREF(value);
    Py_XDECREF(trace);
    qa_set_error(message);
}

static std::vector<std::string> qa_split_python_path(const char * python_path) {
    std::vector<std::string> out;
    if (!qa_is_set(python_path)) {
        return out;
    }

#ifdef _WIN32
    const char sep = ';';
#else
    const char sep = ':';
#endif

    std::stringstream ss(python_path);
    std::string       item;
    while (std::getline(ss, item, sep)) {
        if (!item.empty()) {
            out.push_back(item);
        }
    }
    return out;
}

static bool qa_add_python_path(const char * python_path) {
    PyObject * sys_path = PySys_GetObject("path");
    if (!sys_path || !PyList_Check(sys_path)) {
        qa_set_error("qa_init: could not access sys.path");
        return false;
    }

    for (const std::string & path : qa_split_python_path(python_path)) {
        PyObject * py_path = PyUnicode_FromString(path.c_str());
        if (!py_path) {
            qa_set_python_error("qa_init: failed to create sys.path entry");
            return false;
        }
        if (PySequence_Contains(sys_path, py_path) == 0) {
            if (PyList_Insert(sys_path, 0, py_path) != 0) {
                Py_DECREF(py_path);
                qa_set_python_error("qa_init: failed to update sys.path");
                return false;
            }
        }
        Py_DECREF(py_path);
    }
    return true;
}

static bool qa_ensure_python(const char * python_path) {
    std::lock_guard<std::mutex> lock(g_python_init_mutex);
    if (!Py_IsInitialized()) {
        Py_Initialize();
        if (!Py_IsInitialized()) {
            qa_set_error("qa_init: Py_Initialize failed");
            return false;
        }
    }
    return qa_add_python_path(python_path);
}

static bool qa_dict_set_object(PyObject * dict, const char * key, PyObject * value) {
    if (!value) {
        qa_set_python_error("qa_init: failed to allocate Python object");
        return false;
    }
    int rc = PyDict_SetItemString(dict, key, value);
    Py_DECREF(value);
    if (rc != 0) {
        qa_set_python_error("qa_init: failed to populate kwargs");
        return false;
    }
    return true;
}

static bool qa_dict_set_string(PyObject * dict, const char * key, const char * value) {
    return qa_dict_set_object(dict, key, PyUnicode_FromString(value));
}

static bool qa_dict_set_int(PyObject * dict, const char * key, int value) {
    return qa_dict_set_object(dict, key, PyLong_FromLong((long) value));
}

static bool qa_dict_set_bool(PyObject * dict, const char * key, bool value) {
    PyObject * py_value = value ? Py_True : Py_False;
    Py_INCREF(py_value);
    return qa_dict_set_object(dict, key, py_value);
}

static std::string qa_get_string_attr(PyObject * object, const char * name) {
    PyObject * attr = PyObject_GetAttrString(object, name);
    if (!attr) {
        PyErr_Clear();
        return "";
    }
    PyObject * s = PyObject_Str(attr);
    Py_DECREF(attr);
    if (!s) {
        PyErr_Clear();
        return "";
    }
    const char * utf8 = PyUnicode_AsUTF8(s);
    std::string out  = utf8 ? utf8 : "";
    Py_DECREF(s);
    return out;
}

static char * qa_strdup(const std::string & value) {
    char * out = static_cast<char *>(std::malloc(value.size() + 1));
    if (!out) {
        return nullptr;
    }
    std::memcpy(out, value.c_str(), value.size() + 1);
    return out;
}

extern "C" {

const char * qa_last_error(void) {
    return g_last_error.c_str();
}

void qa_init_default_params(struct qa_init_params * params) {
    if (!params) {
        return;
    }
    params->abi_version        = QA_ABI_VERSION;
    params->model              = nullptr;
    params->size               = "0.6B";
    params->backend            = "torch";
    params->device             = "auto";
    params->dtype              = "auto";
    params->attn_implementation = "auto";
    params->python_path        = nullptr;
    params->max_new_tokens     = 256;
    params->max_batch_size     = 32;
    params->use_cuda_graph     = true;
    params->cuda_graph_stride  = 128;
    params->local_files_only   = false;
}

void qa_transcribe_default_params(struct qa_transcribe_params * params) {
    if (!params) {
        return;
    }
    params->abi_version = QA_ABI_VERSION;
    params->audio       = nullptr;
    params->language    = nullptr;
    params->context     = "";
}

struct qa_context * qa_init(const struct qa_init_params * params) {
    if (!params) {
        qa_set_error("qa_init: params is NULL");
        return nullptr;
    }
    if (params->abi_version > QA_ABI_VERSION) {
        qa_set_error("qa_init: params ABI is newer than this library");
        return nullptr;
    }
    if (!qa_ensure_python(params->python_path)) {
        return nullptr;
    }

    PyGILState_STATE gil = PyGILState_Ensure();

    PyObject * module = PyImport_ImportModule("qwenasr_cpp");
    if (!module) {
        qa_set_python_error("qa_init: import qwenasr_cpp failed");
        PyGILState_Release(gil);
        return nullptr;
    }

    PyObject * from_pretrained = PyObject_GetAttrString(module, "from_pretrained");
    Py_DECREF(module);
    if (!from_pretrained) {
        qa_set_python_error("qa_init: qwenasr_cpp.from_pretrained missing");
        PyGILState_Release(gil);
        return nullptr;
    }

    PyObject * kwargs = PyDict_New();
    PyObject * args   = PyTuple_New(0);
    if (!kwargs || !args) {
        Py_XDECREF(kwargs);
        Py_XDECREF(args);
        Py_DECREF(from_pretrained);
        qa_set_python_error("qa_init: failed to allocate Python call state");
        PyGILState_Release(gil);
        return nullptr;
    }

    bool ok = true;
    if (qa_is_set(params->model)) {
        ok = ok && qa_dict_set_string(kwargs, "model", params->model);
    } else if (qa_is_set(params->size)) {
        ok = ok && qa_dict_set_string(kwargs, "size", params->size);
    }
    if (qa_is_set(params->backend) && std::strcmp(params->backend, "auto") != 0) {
        ok = ok && qa_dict_set_string(kwargs, "backend", params->backend);
    }
    if (qa_is_set(params->device) && std::strcmp(params->device, "auto") != 0) {
        ok = ok && qa_dict_set_string(kwargs, "device", params->device);
    }
    if (qa_is_set(params->dtype) && std::strcmp(params->dtype, "auto") != 0) {
        ok = ok && qa_dict_set_string(kwargs, "dtype", params->dtype);
    }
    if (qa_is_set(params->attn_implementation) && std::strcmp(params->attn_implementation, "auto") != 0) {
        ok = ok && qa_dict_set_string(kwargs, "attn_implementation", params->attn_implementation);
    }
    ok = ok && qa_dict_set_int(kwargs, "max_new_tokens", params->max_new_tokens);
    ok = ok && qa_dict_set_int(kwargs, "max_batch_size", params->max_batch_size);
    ok = ok && qa_dict_set_bool(kwargs, "use_cuda_graph", params->use_cuda_graph);
    ok = ok && qa_dict_set_int(kwargs, "cuda_graph_stride", params->cuda_graph_stride);
    ok = ok && qa_dict_set_bool(kwargs, "local_files_only", params->local_files_only);

    PyObject * model = nullptr;
    if (ok) {
        model = PyObject_Call(from_pretrained, args, kwargs);
        if (!model) {
            qa_set_python_error("qa_init: model load failed");
        }
    }

    Py_DECREF(args);
    Py_DECREF(kwargs);
    Py_DECREF(from_pretrained);

    if (!model) {
        PyGILState_Release(gil);
        return nullptr;
    }

    qa_context * ctx = new (std::nothrow) qa_context();
    if (!ctx) {
        Py_DECREF(model);
        qa_set_error("qa_init: out of memory");
        PyGILState_Release(gil);
        return nullptr;
    }

    ctx->model    = model;
    ctx->model_id = qa_get_string_attr(model, "model_id");
    ctx->backend  = qa_get_string_attr(model, "backend");

    PyGILState_Release(gil);
    return ctx;
}

void qa_free(struct qa_context * ctx) {
    if (!ctx) {
        return;
    }
    PyGILState_STATE gil = PyGILState_Ensure();
    Py_XDECREF(ctx->model);
    PyGILState_Release(gil);
    delete ctx;
}

const char * qa_model_id(const struct qa_context * ctx) {
    return ctx ? ctx->model_id.c_str() : "";
}

const char * qa_backend(const struct qa_context * ctx) {
    return ctx ? ctx->backend.c_str() : "";
}

enum qa_status qa_transcribe(
    struct qa_context *                 ctx,
    const struct qa_transcribe_params * params,
    struct qa_transcription *           out) {
    if (!ctx || !ctx->model || !params || !params->audio || !out) {
        qa_set_error("qa_transcribe: ctx, audio, or out is NULL");
        return QA_STATUS_INVALID_PARAMS;
    }
    if (params->abi_version > QA_ABI_VERSION) {
        qa_set_error("qa_transcribe: params ABI is newer than this library");
        return QA_STATUS_INVALID_PARAMS;
    }

    out->text     = nullptr;
    out->language = nullptr;

    PyGILState_STATE gil = PyGILState_Ensure();

    PyObject * method = PyObject_GetAttrString(ctx->model, "transcribe");
    if (!method) {
        qa_set_python_error("qa_transcribe: model.transcribe missing");
        PyGILState_Release(gil);
        return QA_STATUS_PYTHON_ERROR;
    }

    PyObject * args   = Py_BuildValue("(s)", params->audio);
    PyObject * kwargs = PyDict_New();
    if (!args || !kwargs) {
        Py_XDECREF(args);
        Py_XDECREF(kwargs);
        Py_DECREF(method);
        qa_set_python_error("qa_transcribe: failed to allocate Python call state");
        PyGILState_Release(gil);
        return QA_STATUS_PYTHON_ERROR;
    }

    bool ok = true;
    ok      = ok && qa_dict_set_bool(kwargs, "return_result", true);
    if (qa_is_set(params->context)) {
        ok = ok && qa_dict_set_string(kwargs, "context", params->context);
    }
    if (qa_is_set(params->language)) {
        ok = ok && qa_dict_set_string(kwargs, "language", params->language);
    }

    PyObject * result = nullptr;
    if (ok) {
        result = PyObject_Call(method, args, kwargs);
        if (!result) {
            qa_set_python_error("qa_transcribe: transcription failed");
        }
    }

    Py_DECREF(args);
    Py_DECREF(kwargs);
    Py_DECREF(method);

    if (!result) {
        PyGILState_Release(gil);
        return QA_STATUS_PYTHON_ERROR;
    }

    std::string text     = qa_get_string_attr(result, "text");
    std::string language = qa_get_string_attr(result, "language");
    Py_DECREF(result);

    out->text     = qa_strdup(text);
    out->language = qa_strdup(language);
    if (!out->text || !out->language) {
        qa_transcription_free(out);
        qa_set_error("qa_transcribe: out of memory");
        PyGILState_Release(gil);
        return QA_STATUS_OOM;
    }

    PyGILState_Release(gil);
    return QA_STATUS_OK;
}

void qa_transcription_free(struct qa_transcription * out) {
    if (!out) {
        return;
    }
    std::free(out->text);
    std::free(out->language);
    out->text     = nullptr;
    out->language = nullptr;
}

enum qa_status qa_synchronize(void) {
    if (!Py_IsInitialized()) {
        return QA_STATUS_OK;
    }

    PyGILState_STATE gil   = PyGILState_Ensure();
    PyObject *       torch = PyImport_ImportModule("torch");
    if (!torch) {
        PyErr_Clear();
        PyGILState_Release(gil);
        return QA_STATUS_OK;
    }
    PyObject * cuda = PyObject_GetAttrString(torch, "cuda");
    Py_DECREF(torch);
    if (!cuda) {
        PyErr_Clear();
        PyGILState_Release(gil);
        return QA_STATUS_OK;
    }
    PyObject * available = PyObject_CallMethod(cuda, "is_available", nullptr);
    int        is_avail  = available ? PyObject_IsTrue(available) : 0;
    Py_XDECREF(available);
    if (is_avail > 0) {
        PyObject * sync_result = PyObject_CallMethod(cuda, "synchronize", nullptr);
        if (!sync_result) {
            Py_DECREF(cuda);
            qa_set_python_error("qa_synchronize: torch.cuda.synchronize failed");
            PyGILState_Release(gil);
            return QA_STATUS_PYTHON_ERROR;
        }
        Py_DECREF(sync_result);
    } else if (is_avail < 0) {
        PyErr_Clear();
    }
    Py_DECREF(cuda);
    PyGILState_Release(gil);
    return QA_STATUS_OK;
}

}
