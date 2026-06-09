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

The native converter surface is started in `convert.py`. Today `--dry-run`
validates the full HF -> native tensor-name map for both model sizes; writing
GGUF requires the optional `gguf` Python module and the native runtime loader is
still under construction.

```bash
python convert.py /path/to/Qwen3-ASR-0.6B-snapshot --dry-run
python convert.py /path/to/Qwen3-ASR-1.7B-snapshot --dry-run
```

The GGML-backed native metadata loader is available as `qwen-asr-gguf-info`:

```bash
./build/qwen-asr-gguf-info --self-test
./build/qwen-asr-gguf-info qwen3-asr-0.6b-f32.gguf
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
CLI tool, future GGUF conversion, and a GGML runtime. The bridge currently
validates the C++ surface and comparison harness; the remaining native work is
to port the Whisper log-mel frontend, ASR audio tower/projector, Qwen tokenizer
prompt expansion, and Qwen3 decoder/KV cache into GGML.
