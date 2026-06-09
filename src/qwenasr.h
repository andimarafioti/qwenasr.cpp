#pragma once

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) || defined(__CYGWIN__)
#    if defined(QA_STATIC)
#        define QA_API
#    elif defined(QA_BUILD)
#        define QA_API __declspec(dllexport)
#    else
#        define QA_API __declspec(dllimport)
#    endif
#elif defined(__GNUC__) || defined(__clang__)
#    define QA_API __attribute__((visibility("default")))
#else
#    define QA_API
#endif

#define QA_ABI_VERSION 1

enum qa_status {
    QA_STATUS_OK             = 0,
    QA_STATUS_INVALID_PARAMS = -1,
    QA_STATUS_PYTHON_ERROR   = -2,
    QA_STATUS_OOM            = -3,
};

struct qa_context;

struct qa_init_params {
    int abi_version;

    const char * model;
    const char * size;
    const char * backend;
    const char * device;
    const char * dtype;
    const char * attn_implementation;
    const char * python_path;

    int  max_new_tokens;
    int  max_batch_size;
    bool use_cuda_graph;
    int  cuda_graph_stride;
    bool local_files_only;
};

struct qa_transcribe_params {
    int abi_version;

    const char * audio;
    const char * language;
    const char * context;
};

struct qa_transcription {
    char * text;
    char * language;
};

QA_API const char * qa_last_error(void);

QA_API void qa_init_default_params(struct qa_init_params * params);
QA_API void qa_transcribe_default_params(struct qa_transcribe_params * params);

QA_API struct qa_context * qa_init(const struct qa_init_params * params);
QA_API void qa_free(struct qa_context * ctx);

QA_API const char * qa_model_id(const struct qa_context * ctx);
QA_API const char * qa_backend(const struct qa_context * ctx);

QA_API enum qa_status qa_transcribe(
    struct qa_context *                  ctx,
    const struct qa_transcribe_params *  params,
    struct qa_transcription *            out);

QA_API void qa_transcription_free(struct qa_transcription * out);

QA_API enum qa_status qa_synchronize(void);

#ifdef __cplusplus
}
#endif
