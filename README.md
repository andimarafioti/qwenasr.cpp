# qwenasr.cpp

Installable Qwen3-ASR inference for the 0.6B and 1.7B models, with a small API
modeled after `nano-parakeet` and `faster-qwen3-tts`.

```python
from qwenasr_cpp import from_pretrained

model = from_pretrained(size="0.6B")      # Qwen/Qwen3-ASR-0.6B
print(model.transcribe("audio.wav"))
```

The first implementation path is correctness-first and CUDA-friendly:

- `backend="vllm"` uses the Qwen3-ASR vLLM backend for highest throughput.
- `backend="torch"` uses a manual greedy decoder with CUDA graph capture on top
  of the official Torch model.
- `backend="transformers"` uses the official transformers backend.
- `backend="auto"` picks vLLM when it is installed on a CUDA machine, otherwise
  uses the manual Torch path and falls back to transformers only when Torch is absent.
- The Torch backend defaults to eager attention because it is faster for the
  single-token graph loop on the tested GB10 setup; the transformers baseline
  defaults to SDPA.
- CUDA defaults enable TF32 matmul and choose bfloat16 on GPUs that support it.

## Install

```bash
pip install qwenasr-cpp
```

For the fast vLLM backend:

```bash
pip install "qwenasr-cpp[fast]"
```

For checkpoint conversion utilities:

```bash
pip install "qwenasr-cpp[convert]"
```

The upstream Qwen3-ASR package currently pins `transformers==4.57.6`. Use a fresh
environment if you already have a large ML stack installed.

## Python API

```python
from qwenasr_cpp import QwenASR, from_pretrained

asr = from_pretrained(
    size="1.7B",
    backend="auto",
    max_new_tokens=256,
    max_batch_size=32,
)

text = asr.transcribe("speech.wav", language="English")
print(text)

results = asr.transcribe(
    ["a.wav", "b.wav"],
    language=[None, "English"],
    return_result=True,
)
for result in results:
    print(result.language, result.text)
```

`audio` can be a local path, URL, base64 data URL, `(numpy_array, sample_rate)`,
or a `torch.Tensor` sampled at 16 kHz.

## CLI

```bash
qwenasr audio.wav
qwenasr audio.wav --backend torch
qwenasr audio.wav --backend torch --attn-implementation sdpa
qwenasr audio.wav --backend torch --no-cuda-graph
qwenasr audio.wav --backend torch --cuda-graph-stride 64
qwenasr audio1.wav audio2.wav --size 1.7B --backend vllm --json
qwenasr audio.wav --language English --context "Preserve spelling: CUDA, FFmpeg"
qwenasr audio.wav --size 0.6B --local-files-only
```

## C++ CLI

The repo also contains a qwentts.cpp-style C ABI and `qwen-asr` C++ CLI under
`src/` and `tools/`. The current C++ target is a bridge: it embeds Python and
calls the package backend through the C ABI, which gives a buildable C++
harness for benchmarking and for replacing the internals with a native GGML
runtime.

```bash
git submodule update --init --recursive
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

./build/qwen-asr sample.wav \
  --size 0.6B \
  --backend torch \
  --dtype bf16 \
  --language English \
  --python-path "$PWD:/path/to/venv/lib/python3.12/site-packages"
```

The native converter surface is started in `convert.py`. `--dry-run` validates
the full HF -> native tensor-name map for both model sizes, and
`--metadata-only` writes a small GGUF containing the real checkpoint metadata
without the large tensors. Full tensor GGUF writing requires the `convert` extra.

```bash
python convert.py /path/to/Qwen3-ASR-0.6B-snapshot --dry-run
python convert.py /path/to/Qwen3-ASR-1.7B-snapshot --dry-run
python convert.py /path/to/Qwen3-ASR-0.6B-snapshot -o qwen3-asr-0.6b-meta.gguf --metadata-only
python convert.py /path/to/Qwen3-ASR-0.6B-snapshot -o qwen3-asr-0.6b-conv0.gguf --include-tensor-prefix audio.conv.0.
python convert.py /path/to/Qwen3-ASR-0.6B-snapshot -o qwen3-asr-0.6b-audio-cnn.gguf --include-tensor-prefix audio.conv. --include-tensor-prefix audio.conv_out.
python convert.py /path/to/Qwen3-ASR-0.6B-snapshot -o qwen3-asr-0.6b-audio-layer0.gguf --include-tensor-prefix audio.conv. --include-tensor-prefix audio.conv_out. --include-tensor-prefix audio.blk.0.
python convert.py /path/to/Qwen3-ASR-0.6B-snapshot -o qwen3-asr-0.6b-audio-full.gguf --include-tensor-prefix audio.
python convert.py /path/to/Qwen3-ASR-0.6B-snapshot -o qwen3-asr-0.6b-decoder-input.gguf --include-tensor-prefix audio. --include-tensor-prefix text.token_embd.weight
python convert.py /path/to/Qwen3-ASR-0.6B-snapshot -o qwen3-asr-0.6b-text-layer0.gguf --include-tensor-prefix audio. --include-tensor-prefix text.token_embd.weight --include-tensor-prefix text.blk.0.
python convert.py /path/to/Qwen3-ASR-0.6B-snapshot -o qwen3-asr-0.6b-text-full.gguf --include-tensor-prefix audio. --include-tensor-prefix text.
```

The GGML-backed native metadata loader is available as `qwen-asr-gguf-info`:

```bash
./build/qwen-asr-gguf-info --self-test
./build/qwen-asr-gguf-info qwen3-asr-0.6b-meta.gguf --allow-metadata-only
./build/qwen-asr-gguf-info qwen3-asr-0.6b-f32.gguf
```

The native mapped-weight loader is available as `qwen-asr-weights`. It validates
that tensor data ranges fit inside the GGUF file and exposes tensor data pointers
for the future GGML graph loaders:

```bash
./build/qwen-asr-weights --self-test
./build/qwen-asr-weights qwen3-asr-0.6b-meta.gguf --allow-metadata-only
./build/qwen-asr-weights qwen3-asr-0.6b-f32.gguf --tensor text.token_embd.weight
```

The native tokenizer loader is available as `qwen-asr-tokenize`. It reads the
qwentts-style `tokenizer.ggml.*` GGUF metadata, expands the ASR audio prompt,
and can be checked against the HF tokenizer:

```bash
./build/qwen-asr-tokenize qwen3-asr-0.6b-meta.gguf --audio-tokens 143 --language English
python benchmarks/check_tokenizer.py /path/to/Qwen3-ASR-0.6B-snapshot qwen3-asr-0.6b-meta.gguf --audio-tokens 143 --language English
```

The native audio frontend is available as `qwen-asr-features`. It currently
supports 16 kHz WAV input and matches the HF `WhisperFeatureExtractor` path used
by Qwen3-ASR:

```bash
./build/qwen-asr-features sample.wav --out features.f32
python benchmarks/check_audio_features.py sample.wav --local-files-only
```

The first mapped-weight native audio layer is available as
`qwen-asr-audio-conv`. It runs `audio.conv.0.*` from a GGUF through a GGML graph
by default, keeps the scalar backend for comparison, and can be checked against
the original HF checkpoint tensors:

```bash
./build/qwen-asr-audio-conv qwen3-asr-0.6b-conv0.gguf sample.wav --out conv0.f32
./build/qwen-asr-audio-conv qwen3-asr-0.6b-conv0.gguf sample.wav --backend scalar
python benchmarks/check_audio_conv0.py /path/to/Qwen3-ASR-0.6B-snapshot qwen3-asr-0.6b-conv0.gguf sample.wav
```

The native audio CNN frontend is available as `qwen-asr-audio-cnn`. It runs the
three downsampling Conv2D+GELU layers and `conv_out`, producing
`[chunk, frame, hidden]` embeddings before positional embeddings and the audio
transformer:

```bash
./build/qwen-asr-audio-cnn qwen3-asr-0.6b-audio-cnn.gguf sample.wav --out audio-cnn.f32
python benchmarks/check_audio_cnn.py /path/to/Qwen3-ASR-0.6B-snapshot qwen3-asr-0.6b-audio-cnn.gguf sample.wav
```

The pre-transformer audio prep boundary adds sinusoidal positional embeddings,
packs only valid per-chunk frames, and reports the attention segments consumed
by the audio transformer. `--backend sched` keeps the audio CNN weights in a
persistent GGML CPU backend buffer:

```bash
./build/qwen-asr-audio-prep qwen3-asr-0.6b-audio-cnn.gguf sample.wav --out audio-prep.f32
./build/qwen-asr-audio-prep qwen3-asr-0.6b-audio-cnn.gguf sample.wav --backend sched --out audio-prep-sched.f32
python benchmarks/check_audio_prep.py /path/to/Qwen3-ASR-0.6B-snapshot qwen3-asr-0.6b-audio-cnn.gguf sample.wav
python benchmarks/check_audio_prep.py /path/to/Qwen3-ASR-0.6B-snapshot qwen3-asr-0.6b-audio-cnn.gguf sample.wav --native-backend sched
```

The first audio transformer block is available in `qwen-asr-audio-layer`. It
defaults to a native GGML graph and keeps the scalar CPU reference available via
`--backend cpu`. Both paths target the current eager-attention Torch path:

```bash
./build/qwen-asr-audio-layer qwen3-asr-0.6b-audio-layer0.gguf sample.wav --out audio-layer0.f32
./build/qwen-asr-audio-layer qwen3-asr-0.6b-audio-layer0.gguf sample.wav --backend cpu --out audio-layer0-cpu.f32
python benchmarks/check_audio_layer0.py /path/to/Qwen3-ASR-0.6B-snapshot qwen3-asr-0.6b-audio-layer0.gguf sample.wav
```

The full native audio encoder path is available as `qwen-asr-audio-encoder`.
It runs the audio CNN, all audio transformer layers, `ln_post`, and the two
projector layers, producing the 1024-wide audio embeddings consumed by the
Qwen3 text decoder. The default `--backend ggml` path is the original
correctness-first implementation. The `--backend sched` path keeps the audio
CNN, transformer, and projector weights in persistent GGML CPU backend buffers
and runs the hot path through scheduler graphs:

```bash
./build/qwen-asr-audio-encoder qwen3-asr-0.6b-audio-full.gguf sample.wav --out audio-encoder.f32
./build/qwen-asr-audio-encoder qwen3-asr-0.6b-audio-full.gguf sample.wav --backend sched --out audio-encoder-sched.f32
python benchmarks/check_audio_encoder.py /path/to/Qwen3-ASR-0.6B-snapshot qwen3-asr-0.6b-audio-full.gguf sample.wav
python benchmarks/check_audio_encoder.py /path/to/Qwen3-ASR-0.6B-snapshot qwen3-asr-0.6b-audio-full.gguf sample.wav --native-backend sched
```

The decoder input assembly boundary is available as `qwen-asr-decoder-input`.
It looks up prompt token embeddings, replaces `<|audio_pad|>` positions with
the native audio encoder output, and emits the embeddings passed into the Qwen3
text decoder. It also supports `--audio-backend sched` to reuse the scheduled
audio CNN, transformer, and projector backend:

```bash
./build/qwen-asr-decoder-input qwen3-asr-0.6b-decoder-input.gguf sample.wav --language English --out decoder-input.f32
./build/qwen-asr-decoder-input qwen3-asr-0.6b-decoder-input.gguf sample.wav --language English --audio-backend sched --out decoder-input-sched.f32
python benchmarks/check_decoder_input.py /path/to/Qwen3-ASR-0.6B-snapshot qwen3-asr-0.6b-decoder-input.gguf sample.wav --language English
python benchmarks/check_decoder_input.py /path/to/Qwen3-ASR-0.6B-snapshot qwen3-asr-0.6b-decoder-input.gguf sample.wav --language English --native-audio-backend sched
```

The first Qwen3 text decoder block and full prompt prefill logits are available
as `qwen-asr-text-layer`. It builds the decoder input embeddings, then runs
scalar CPU Qwen3 blocks with RMSNorm, Q/K/V projection, per-head Q/K RMSNorm,
RoPE, grouped-query causal attention, output projection, and SwiGLU MLP.
`--prefill` runs every text block plus the output RMSNorm and LM head, returning
the next-token logits from the final prompt position. `--generate N` adds a
minimal greedy loop that decodes generated token ids back to text; `--kv-cache`
keeps scalar per-layer K/V tensors from prompt prefill and runs following tokens
through a one-row cached decode. This is a correctness boundary for the future
native decoder backend path; it is not yet the fast GGML autoregressive decoder:

```bash
./build/qwen-asr-text-layer qwen3-asr-0.6b-text-layer0.gguf sample.wav --language English --out text-layer0.f32
./build/qwen-asr-text-layer qwen3-asr-0.6b-text-layer0.gguf sample.wav --language English --audio-backend sched --out text-layer0-sched.f32
./build/qwen-asr-text-layer qwen3-asr-0.6b-text-full.gguf sample.wav --language English --audio-backend sched --prefill --out next-token-logits.f32
./build/qwen-asr-text-layer qwen3-asr-0.6b-text-full.gguf sample.wav --language English --audio-backend sched --generate 2 --out native-prefix.txt
./build/qwen-asr-text-layer qwen3-asr-0.6b-text-full.gguf sample.wav --language English --audio-backend sched --generate 2 --kv-cache --out native-prefix-cache.txt
python benchmarks/check_text_layer0.py /path/to/Qwen3-ASR-0.6B-snapshot qwen3-asr-0.6b-text-layer0.gguf sample.wav --language English
python benchmarks/check_text_prefill.py /path/to/Qwen3-ASR-0.6B-snapshot qwen3-asr-0.6b-text-full.gguf sample.wav --language English
python benchmarks/check_text_generate.py /path/to/Qwen3-ASR-0.6B-snapshot qwen3-asr-0.6b-text-full.gguf sample.wav --language English --max-new-tokens 2
python benchmarks/check_text_generate.py /path/to/Qwen3-ASR-0.6B-snapshot qwen3-asr-0.6b-text-full.gguf sample.wav --language English --max-new-tokens 2 --native-decode-backend kv-cache
```

## Streaming

Streaming is exposed when the vLLM backend is used:

```python
asr = from_pretrained(size="0.6B", backend="vllm", max_new_tokens=32)
state = asr.init_streaming_state(language="English")

for chunk in chunks_16k_float32:
    state = asr.streaming_transcribe(chunk, state)
    print(state.text)

state = asr.finish_streaming_transcribe(state)
print(state.text)
```

## Benchmark

```bash
python benchmarks/throughput.py sample.wav --size 0.6B --backend auto
python benchmarks/throughput.py sample.wav --size 0.6B --local-files-only
python benchmarks/throughput.py sample.wav --size 1.7B --backend torch --language English
python benchmarks/throughput.py sample.wav --size 0.6B --backend torch --repeat 4
python benchmarks/profile_torch.py sample.wav --size 0.6B --language English
python benchmarks/compare_parakeet.py sample.wav --qwen-size 0.6B
python benchmarks/compare_cpp.py sample.wav --size 0.6B --backend torch --language English
python benchmarks/bench_audio_conv0.py /path/to/Qwen3-ASR-0.6B-snapshot qwen3-asr-0.6b-conv0.gguf sample.wav --torch-device cpu
python benchmarks/bench_audio_cnn.py /path/to/Qwen3-ASR-0.6B-snapshot qwen3-asr-0.6b-audio-cnn.gguf sample.wav --torch-device cpu
python benchmarks/bench_audio_prep.py /path/to/Qwen3-ASR-0.6B-snapshot qwen3-asr-0.6b-audio-cnn.gguf sample.wav --torch-device cpu
python benchmarks/bench_audio_layer0.py /path/to/Qwen3-ASR-0.6B-snapshot qwen3-asr-0.6b-audio-layer0.gguf sample.wav --torch-device cpu
python benchmarks/bench_audio_encoder.py /path/to/Qwen3-ASR-0.6B-snapshot qwen3-asr-0.6b-audio-full.gguf sample.wav --torch-device cpu
python benchmarks/bench_decoder_input.py /path/to/Qwen3-ASR-0.6B-snapshot qwen3-asr-0.6b-decoder-input.gguf sample.wav --language English --torch-device cpu
python benchmarks/bench_text_layer0.py /path/to/Qwen3-ASR-0.6B-snapshot qwen3-asr-0.6b-text-layer0.gguf sample.wav --language English --torch-device cpu
python benchmarks/bench_text_prefill.py /path/to/Qwen3-ASR-0.6B-snapshot qwen3-asr-0.6b-text-full.gguf sample.wav --language English --torch-device cpu
python benchmarks/bench_text_generate.py /path/to/Qwen3-ASR-0.6B-snapshot qwen3-asr-0.6b-text-full.gguf sample.wav --language English --max-new-tokens 2 --cpp-decode-backends kv-cache --torch-device cpu
```

RTF is reported as `audio_duration / wall_time`; values above 1 are faster than
real time.

See `examples/transcribe.py` for the minimal file path flow and
`examples/streaming.py` for a vLLM streaming loop.

Current GB10 smoke numbers on `/tmp/qwen-asr-ref/samples/jfk.wav` with cached
BF16 weights, synchronized CUDA timing, and the default 128-token CUDA graph
bucket:

| Model | Backend | Language | Best Time | RTF |
|---|---|---|---:|---:|
| Qwen3-ASR-0.6B | torch + CUDA graph | auto | 0.329s | 33.5x |
| Qwen3-ASR-0.6B | torch + CUDA graph | English | 0.299s | 36.8x |
| Qwen3-ASR-0.6B | torch + CUDA graph, fp16 | English | 0.294s | 37.4x |
| Qwen3-ASR-0.6B | torch batched fallback, 4 clips | English | 0.391s | 112.4x |
| Qwen3-ASR-0.6B | C++ bridge -> torch + CUDA graph | English | 0.340s | 32.4x |
| Qwen3-ASR-1.7B | torch + CUDA graph | English | 0.644s | 17.1x |
| nano-parakeet | PyTorch | auto | 0.028s | 397.2x |

Qwen3-ASR is still much more autoregressive work than Parakeet TDT on short
clips, so the next major speed target is reducing the per-token decoder cost
further or using a vLLM backend where available.

For the 0.6B English JFK run, `benchmarks/profile_torch.py` reports a
single-chunk stage breakdown of roughly 1 ms audio loading/chunking, 10 ms
processor prep, 25 ms audio/text prefill, and 261 ms CUDA graph decoding for 26
generated tokens. The current bottleneck is therefore the Qwen decoder's
single-token autoregressive loop, not Python tokenization or audio loading.

The native conv0 microbenchmark on the same clip currently shows the first GGML
graph path is correct but not yet competitive: with 8 CPU threads,
`benchmarks/bench_audio_conv0.py` measured roughly 286 ms best for C++ GGML,
170 ms for the scalar C++ loop, and 1.24 ms for Torch CUDA BF16. That makes the
next native speed target clear: avoid per-call graph/setup/output-copy overhead
and move the native audio tower into reusable GGML/backend graphs.

For the larger audio CNN plus `conv_out` slice, `benchmarks/bench_audio_cnn.py`
measured roughly 540 ms best for C++ GGML with 8 CPU threads, 97 ms for Torch
CPU FP32, and 2.43 ms for Torch CUDA BF16. This validates the native tensor
layout through the frontend projection and shows the current native CPU graph is
not yet an end-to-end speed win.

At the packed pre-transformer boundary, `benchmarks/bench_audio_prep.py`
measured roughly 774-874 ms for the original C++ GGML path with 8 CPU threads,
435-462 ms with `--backend sched`, 99 ms for Torch CPU FP32, and 2.54 ms for
Torch CUDA BF16. The output matches PyTorch at `1.53e-5` max absolute error and
produces the JFK attention segments `[(0, 104), (104, 39)]`. A direct
`qwen-asr-audio-prep --backend sched` run reported about 17 ms of one-time
backend initialization and 450 ms hot prep time.

For the first audio encoder block, `benchmarks/bench_audio_layer0.py` measured
roughly 1.09-1.18 s best for the native GGML graph, 1.37 s for the scalar CPU
reference, 123-129 ms for Torch CPU FP32, and 2.73 ms for Torch CUDA BF16. The
GGML output matches the current eager Torch path at `1.52e-5` max absolute
error. The next speed work is making the graph reusable and moving weights onto
real GGML backends instead of rebuilding/copying every call.

For the full audio encoder/projector boundary, `benchmarks/check_audio_encoder.py`
matches the eager Torch reference at `4.09e-6` max absolute error on JFK. The
original per-layer native GGML loop is correctness-first and slow; the
qwentts.cpp-style scheduler path uploads CNN/transformer/projector weights once
and removes the frontend and per-layer context rebuild/copy costs. On JFK with
8 CPU threads, `benchmarks/bench_audio_encoder.py` measured roughly 7.5-8.2 s
for the original C++ GGML path, 0.61-0.62 s for `--backend sched`, 368 ms for
Torch CPU FP32, and 6.88 ms for Torch CUDA BF16. A direct
`qwen-asr-audio-encoder --backend sched` run reported about 244 ms of one-time
backend initialization and 794 ms hot encoder time.

For the decoder input assembly boundary, `benchmarks/check_decoder_input.py`
matches the Torch prompt embedding plus `masked_scatter` path at `4.09e-6` max
absolute error on JFK with English forced output. `benchmarks/bench_decoder_input.py`
measured roughly 7.5-7.8 s for the original C++ GGML audio path with 8 CPU
threads, 0.62-0.64 s with `--audio-backend sched`, 347 ms for Torch CPU FP32,
and 7.23 ms for Torch CUDA BF16. A direct
`qwen-asr-decoder-input --audio-backend sched` run reported about 184 ms of
one-time backend initialization and 766 ms hot decoder-input time.

For the first Qwen3 text decoder block, `benchmarks/check_text_layer0.py`
matches the eager Torch reference at `4.67e-5` max absolute error on JFK with
English forced output. `benchmarks/bench_text_layer0.py` measured roughly
1.23-1.25 s for the scalar C++ text block with 8 CPU threads, 20.4 ms for Torch
CPU FP32 with 8 threads, and 0.99 ms for Torch CUDA BF16. The C++ text block is
therefore currently a validated decoder boundary, not a speed win; the next
native decoder work is moving this layer into reusable GGML/backend graphs and
adding KV-cache single-token decoding.

For the full prompt prefill logits, `benchmarks/check_text_prefill.py` matches
the eager Torch reference at `1.34e-4` max absolute error on JFK and agrees on
the first next-token id (`3036`). `benchmarks/bench_text_prefill.py` measured
roughly 34.5-34.6 s for scalar C++ full text prefill with 8 CPU threads, 491 ms
for Torch CPU FP32 with 8 threads, and 28.1 ms for Torch CUDA BF16. This covers
all decoder layers and the LM head in native C++, but also makes the next
optimization target concrete: reusable GGML/backend text graphs and a KV-cache
decode loop.

For greedy generation, `benchmarks/check_text_generate.py` matches the Torch
reference for the first two JFK English tokens (`3036,773`, decoded as `And so`)
with both the recompute and scalar KV-cache native paths. The original recompute
path took about 69.3 s for two tokens. With `--kv-cache`, the same two-token
run measured about 34.9 s with 8 CPU threads, 984 ms for Torch CPU FP32 with 8
threads, and 82.0 ms for Torch CUDA BF16. This is now an end-to-end native
audio-to-token/text path with cached scalar decode after prompt prefill, but it
still highlights the remaining qwentts.cpp-style work: persistent GGML/backend
text graphs and a fast KV-cache single-token decode loop.

## Implementation Notes

Qwen3-ASR is not a Whisper-style encoder-decoder. It uses a chunked audio
transformer encoder and a Qwen3 decoder:

```text
16 kHz audio -> 128-bin log-mel -> per-chunk 3x Conv2D downsample
  -> windowed AuT encoder -> projector -> Qwen3 autoregressive decoder
```

The current repo intentionally reuses the official Qwen3-ASR model registration
for weight compatibility and prompt correctness. The next performance target is
a lean native/Torch fast path that avoids the official transformers backend's
per-sample audio encoder loop and adds a captured single-stream decoder path.

The native C++ port target mirrors qwentts.cpp's shape: CMake build, C ABI,
CLI tools, GGUF conversion, and a GGML runtime. The bridge currently validates
the C++ surface and comparison harness, and the native path now covers GGUF
metadata/tensor validation, Whisper log-mel features, audio geometry, and Qwen
BPE prompt expansion. It also has a mapped GGUF tensor-data loader, validated
scalar and GGML implementations of the first audio Conv2D layer, and a validated
GGML implementation of the three-layer audio CNN plus `conv_out`, sinusoidal
position embeddings, valid-frame packing, all audio transformer layers, post
normalization, the audio projector, decoder input embedding assembly, and a
qwentts.cpp-style scheduled backend for the audio CNN, transformer, and
projector that is also wired through decoder input assembly. The Qwen3 text
decoder is validated as a scalar CPU path through the first block and full
prompt prefill logits, including final normalization and the LM head, plus a
minimal greedy generation loop with byte-level BPE decode and scalar KV-cache
support for follow-up tokens.
The remaining native work is to move the Qwen3 decoder/KV-cache path into
persistent GGML/backend graphs and extend that backend path through the
remaining standalone frontend tools.
