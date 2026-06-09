#include "decoder-input.h"

#include "audio-encoder.h"

#include <cstring>

static void set_error(std::string * error, const std::string & message) {
    if (error) {
        *error = message;
    }
}

static bool require_f32_tensor(
    const QwenAsrGgufModel & model,
    const char * name,
    QwenAsrGgufTensorView * out,
    std::string * error) {
    if (!qwenasr_gguf_model_tensor_by_name(model, name, out, error)) {
        return false;
    }
    if (out->type != GGML_TYPE_F32) {
        set_error(error, std::string("tensor is not f32: ") + name);
        return false;
    }
    return true;
}

bool qwenasr_decoder_input_forward_ggml(
    const QwenAsrGgufModel & model,
    const QwenAsrTokenizer & tokenizer,
    const QwenAsrFeatures & features,
    int n_threads,
    int n_audio_layers,
    int n_audio_heads,
    int audio_output_dim,
    int audio_token_id,
    const std::string & system_text,
    const std::string & language,
    QwenAsrDecoderInputOutput * out,
    std::string * error) {
    if (!out) {
        set_error(error, "qwenasr_decoder_input_forward_ggml: out is null");
        return false;
    }
    if (audio_output_dim <= 0 || audio_token_id < 0) {
        set_error(error, "invalid decoder input configuration");
        return false;
    }

    QwenAsrAudioEncoderOutput audio;
    if (!qwenasr_audio_encoder_forward_ggml(
            model,
            features,
            n_threads,
            n_audio_layers,
            n_audio_heads,
            audio_output_dim,
            &audio,
            error)) {
        return false;
    }
    if (audio.tokens <= 0 || audio.hidden <= 0 ||
        audio.values.size() != static_cast<size_t>(audio.tokens) * audio.hidden) {
        set_error(error, "invalid native audio encoder output");
        return false;
    }

    const std::string prompt = qwenasr_build_asr_prompt(audio.tokens, system_text, language);
    std::vector<int32_t> ids = qwenasr_tokenizer_encode(tokenizer, prompt, false);
    if (ids.empty()) {
        set_error(error, "decoder prompt produced no tokens");
        return false;
    }

    QwenAsrGgufTensorView embd;
    if (!require_f32_tensor(model, "text.token_embd.weight", &embd, error)) {
        return false;
    }
    if (embd.ne.size() != 2 || embd.ne[0] != audio.hidden || embd.ne[1] <= 0) {
        set_error(error, "text.token_embd.weight tensor shape mismatch");
        return false;
    }
    const int hidden = static_cast<int>(embd.ne[0]);
    const int vocab = static_cast<int>(embd.ne[1]);
    const float * embedding = static_cast<const float *>(embd.data);

    QwenAsrDecoderInputOutput result;
    result.tokens = static_cast<int>(ids.size());
    result.hidden = hidden;
    result.input_ids = std::move(ids);
    result.values.assign(static_cast<size_t>(result.tokens) * hidden, 0.0f);

    int audio_positions = 0;
    for (int token = 0; token < result.tokens; ++token) {
        const int32_t id = result.input_ids[static_cast<size_t>(token)];
        if (id < 0 || id >= vocab) {
            set_error(error, "decoder input token id is out of embedding range");
            return false;
        }
        float * dst = result.values.data() + static_cast<size_t>(token) * hidden;
        std::memcpy(dst, embedding + static_cast<size_t>(id) * hidden, static_cast<size_t>(hidden) * sizeof(float));
        if (id == audio_token_id) {
            if (audio_positions >= audio.tokens) {
                set_error(error, "prompt has more audio placeholders than audio encoder outputs");
                return false;
            }
            const float * src = audio.values.data() + static_cast<size_t>(audio_positions) * hidden;
            std::memcpy(dst, src, static_cast<size_t>(hidden) * sizeof(float));
            ++audio_positions;
        }
    }
    if (audio_positions != audio.tokens) {
        set_error(error, "audio placeholder count does not match audio encoder output count");
        return false;
    }

    result.audio_tokens = audio.tokens;
    *out = std::move(result);
    return true;
}
