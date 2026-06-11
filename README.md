# qwenasr.cpp

Native C++/GGML implementation work for Qwen3-ASR 0.6B and 1.7B.

The fast Torch/vLLM CUDA runtime was split into
[`faster-qwen-asr`](https://github.com/andimarafioti/faster-qwen-asr). This
repository now focuses on GGUF conversion, native audio/text graph validation,
and qwentts.cpp-style GGML backend execution.

## faster-qwen-asr benchmark

The split Torch/CUDA runtime in
[`faster-qwen-asr`](https://github.com/andimarafioti/faster-qwen-asr) was
benchmarked on June 10, 2026 on an NVIDIA GB10 with PyTorch 2.11.0+cu130, CUDA
13.0, and driver 580.126.09. The test audio was an 11.0s 16 kHz mono JFK clip,
forced English, `--backend torch --dtype bf16`, one warmup, and five timed
runs. RTF > 1.0 is faster than real time.

| Model | Dynamic decode latency | Dynamic RTF | CUDA graph latency | CUDA graph RTF | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3-ASR-0.6B | 0.3951s | 27.84 | 0.2863s | 38.42 | 1.38x |
| Qwen3-ASR-1.7B | 1.5435s | 7.13 | 0.6296s | 17.47 | 2.45x |

Full run metadata is tracked in
[`bench_results_NVIDIA_GB10.json`](https://github.com/andimarafioti/faster-qwen-asr/blob/main/bench_results_NVIDIA_GB10.json).

## Build

```bash
git submodule update --init --recursive
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

The native GGML build enables CUDA by default through `QWENASR_GGML_CUDA=ON`.
Scheduled GGML tools use `--device auto` by default, which picks a GPU backend
when one is registered and keeps a CPU backend in the scheduler for fallback
ops. Use `--device gpu` to require GPU execution, `--device cpu` to force the
CPU backend, or configure with `-D QWENASR_GGML_CUDA=OFF` for a CPU-only build.
The scheduled audio encoder tries a combined audio-CNN-plus-encoder graph first
so full chunks can avoid a GPU-to-CPU-to-GPU intermediate copy. Tool output
prints `audio_path=combined` or `audio_path=two-stage`; set
`QWENASR_GGML_COMBINED_AUDIO=0` to force the older two-stage path. The
experimental `QWENASR_GGML_FLASH_ATTN=1` path is available for attention
testing but is off by default.

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
by the audio transformer. The default `--backend sched --device auto` path
keeps the audio CNN weights in a persistent GGML backend buffer, prefers a GPU
backend when available, and keeps the original per-call `--backend ggml` path
for correctness comparisons:

```bash
./build/qwen-asr-audio-prep qwen3-asr-0.6b-audio-cnn.gguf sample.wav --out audio-prep.f32
./build/qwen-asr-audio-prep qwen3-asr-0.6b-audio-cnn.gguf sample.wav --backend ggml --out audio-prep-ggml.f32
./build/qwen-asr-audio-prep qwen3-asr-0.6b-audio-cnn.gguf sample.wav --device gpu --out audio-prep-gpu.f32
python benchmarks/check_audio_prep.py /path/to/Qwen3-ASR-0.6B-snapshot qwen3-asr-0.6b-audio-cnn.gguf sample.wav --native-backend ggml
python benchmarks/check_audio_prep.py /path/to/Qwen3-ASR-0.6B-snapshot qwen3-asr-0.6b-audio-cnn.gguf sample.wav --native-backend sched --native-device gpu --atol 2e-3
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
Qwen3 text decoder. The default `--backend sched --device auto` path keeps the
audio CNN, transformer, and projector weights in persistent GGML backend
buffers and runs the hot path through scheduler graphs on GPU when available.
The original correctness-first implementation remains available as
`--backend ggml`:

```bash
./build/qwen-asr-audio-encoder qwen3-asr-0.6b-audio-full.gguf sample.wav --out audio-encoder.f32
./build/qwen-asr-audio-encoder qwen3-asr-0.6b-audio-full.gguf sample.wav --backend ggml --out audio-encoder-ggml.f32
./build/qwen-asr-audio-encoder qwen3-asr-0.6b-audio-full.gguf sample.wav --device gpu --out audio-encoder-gpu.f32
python benchmarks/check_audio_encoder.py /path/to/Qwen3-ASR-0.6B-snapshot qwen3-asr-0.6b-audio-full.gguf sample.wav --native-backend ggml
python benchmarks/check_audio_encoder.py /path/to/Qwen3-ASR-0.6B-snapshot qwen3-asr-0.6b-audio-full.gguf sample.wav --native-backend sched --native-device gpu
```

The decoder input assembly boundary is available as `qwen-asr-decoder-input`.
It looks up prompt token embeddings, replaces `<|audio_pad|>` positions with
the native audio encoder output, and emits the embeddings passed into the Qwen3
text decoder. Its default `--audio-backend sched --device auto` path reuses the
scheduled audio CNN, transformer, and projector backend, while
`--audio-backend ggml` keeps the original per-call audio path available:

```bash
./build/qwen-asr-decoder-input qwen3-asr-0.6b-decoder-input.gguf sample.wav --language English --out decoder-input.f32
./build/qwen-asr-decoder-input qwen3-asr-0.6b-decoder-input.gguf sample.wav --language English --audio-backend ggml --out decoder-input-ggml.f32
./build/qwen-asr-decoder-input qwen3-asr-0.6b-decoder-input.gguf sample.wav --language English --device gpu --out decoder-input-gpu.f32
python benchmarks/check_decoder_input.py /path/to/Qwen3-ASR-0.6B-snapshot qwen3-asr-0.6b-decoder-input.gguf sample.wav --language English --native-audio-backend ggml
python benchmarks/check_decoder_input.py /path/to/Qwen3-ASR-0.6B-snapshot qwen3-asr-0.6b-decoder-input.gguf sample.wav --language English --native-audio-backend sched --native-device gpu
```

The first Qwen3 text decoder block and full prompt prefill logits are available
as `qwen-asr-text-layer`. It builds the decoder input embeddings, then runs
Qwen3 blocks with RMSNorm, Q/K/V projection, per-head Q/K RMSNorm, RoPE,
grouped-query causal attention, output projection, and SwiGLU MLP. Layer mode
defaults to a qwentts.cpp-style `--backend sched --device auto` graph that
uploads layer weights into a GGML backend buffer once and prefers GPU execution
when available. `--backend scalar` and the correctness-first per-call
`--backend ggml` graph remain available for comparisons. `--prefill` runs every
text block plus the output RMSNorm and LM head, returning the next-token logits
from the final prompt position. `--generate N` decodes generated token ids back
to text; the scheduled backend uses KV cache automatically, storing K/V tensors
in backend buffers and appending through GGML graph-side cache writes:

```bash
./build/qwen-asr-text-layer qwen3-asr-0.6b-text-layer0.gguf sample.wav --language English --out text-layer0.f32
./build/qwen-asr-text-layer qwen3-asr-0.6b-text-layer0.gguf sample.wav --language English --backend scalar --out text-layer0-scalar.f32
./build/qwen-asr-text-layer qwen3-asr-0.6b-text-layer0.gguf sample.wav --language English --backend ggml --out text-layer0-ggml.f32
./build/qwen-asr-text-layer qwen3-asr-0.6b-text-layer0.gguf sample.wav --language English --device gpu --out text-layer0-gpu.f32
./build/qwen-asr-text-layer qwen3-asr-0.6b-text-full.gguf sample.wav --language English --prefill --out next-token-logits.f32
./build/qwen-asr-text-layer qwen3-asr-0.6b-text-full.gguf sample.wav --language English --prefill --device gpu --out next-token-logits-gpu.f32
./build/qwen-asr-text-layer qwen3-asr-0.6b-text-full.gguf sample.wav --language English --generate 2 --out native-prefix.txt
./build/qwen-asr-text-layer qwen3-asr-0.6b-text-full.gguf sample.wav --language English --generate 2 --device gpu --out native-prefix-gpu.txt
python benchmarks/check_text_layer0.py /path/to/Qwen3-ASR-0.6B-snapshot qwen3-asr-0.6b-text-layer0.gguf sample.wav --language English --native-backend scalar
python benchmarks/check_text_layer0.py /path/to/Qwen3-ASR-0.6B-snapshot qwen3-asr-0.6b-text-layer0.gguf sample.wav --language English --native-backend ggml
python benchmarks/check_text_layer0.py /path/to/Qwen3-ASR-0.6B-snapshot qwen3-asr-0.6b-text-layer0.gguf sample.wav --language English --native-backend sched --native-device gpu --atol 2e-2
python benchmarks/check_text_prefill.py /path/to/Qwen3-ASR-0.6B-snapshot qwen3-asr-0.6b-text-full.gguf sample.wav --language English --native-backend scalar
python benchmarks/check_text_prefill.py /path/to/Qwen3-ASR-0.6B-snapshot qwen3-asr-0.6b-text-full.gguf sample.wav --language English --native-backend sched --native-device gpu --atol 3e-2
python benchmarks/check_text_generate.py /path/to/Qwen3-ASR-0.6B-snapshot qwen3-asr-0.6b-text-full.gguf sample.wav --language English --max-new-tokens 2 --native-backend scalar
python benchmarks/check_text_generate.py /path/to/Qwen3-ASR-0.6B-snapshot qwen3-asr-0.6b-text-full.gguf sample.wav --language English --max-new-tokens 4 --native-backend scalar --native-decode-backend kv-cache
python benchmarks/check_text_generate.py /path/to/Qwen3-ASR-0.6B-snapshot qwen3-asr-0.6b-text-full.gguf sample.wav --language English --max-new-tokens 4 --native-backend sched --native-decode-backend kv-cache --native-device gpu
```

## Benchmark

```bash
python benchmarks/bench_audio_conv0.py /path/to/Qwen3-ASR-0.6B-snapshot qwen3-asr-0.6b-conv0.gguf sample.wav --torch-device cpu
python benchmarks/bench_audio_cnn.py /path/to/Qwen3-ASR-0.6B-snapshot qwen3-asr-0.6b-audio-cnn.gguf sample.wav --torch-device cpu
python benchmarks/bench_audio_prep.py /path/to/Qwen3-ASR-0.6B-snapshot qwen3-asr-0.6b-audio-cnn.gguf sample.wav --cpp-devices gpu cpu --torch-device cpu
python benchmarks/bench_audio_layer0.py /path/to/Qwen3-ASR-0.6B-snapshot qwen3-asr-0.6b-audio-layer0.gguf sample.wav --torch-device cpu
python benchmarks/bench_audio_encoder.py /path/to/Qwen3-ASR-0.6B-snapshot qwen3-asr-0.6b-audio-full.gguf sample.wav --cpp-devices gpu cpu --torch-device cpu
python benchmarks/bench_decoder_input.py /path/to/Qwen3-ASR-0.6B-snapshot qwen3-asr-0.6b-decoder-input.gguf sample.wav --language English --cpp-devices gpu cpu --torch-device cpu
python benchmarks/bench_text_layer0.py /path/to/Qwen3-ASR-0.6B-snapshot qwen3-asr-0.6b-text-layer0.gguf sample.wav --language English --cpp-backends scalar ggml sched --cpp-devices gpu cpu --torch-device cpu
python benchmarks/bench_text_prefill.py /path/to/Qwen3-ASR-0.6B-snapshot qwen3-asr-0.6b-text-full.gguf sample.wav --language English --cpp-backends scalar sched --cpp-devices gpu cpu --torch-device cpu
python benchmarks/bench_text_generate.py /path/to/Qwen3-ASR-0.6B-snapshot qwen3-asr-0.6b-text-full.gguf sample.wav --language English --max-new-tokens 4 --cpp-backends scalar sched --cpp-devices gpu cpu --cpp-decode-backends kv-cache --torch-device cpu
```

These scripts use Torch only as a reference implementation and timing baseline;
the native implementation under test is in C++/GGML.

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
435-462 ms with `--backend sched --device cpu`, 99 ms for Torch CPU FP32, and
2.54 ms for Torch CUDA BF16. The output matches PyTorch at `1.53e-5` max
absolute error and produces the JFK attention segments `[(0, 104), (104, 39)]`.
With GGML CUDA enabled, `--backend sched --device gpu` measured about 185 ms
hot prep time with expected CUDA numeric drift under `2e-3`.

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
for the original C++ GGML path, 0.61-0.62 s for `--backend sched --device cpu`,
368 ms for Torch CPU FP32, and 6.88 ms for Torch CUDA BF16. With GGML CUDA
enabled, a direct `qwen-asr-audio-encoder --backend sched --device gpu` run
reported about 1.18 s of one-time backend/weight initialization and 178 ms hot
encoder time, compared with 628 ms hot encoder time for `--device cpu` in the
same CUDA-enabled build.

For the decoder input assembly boundary, `benchmarks/check_decoder_input.py`
matches the Torch prompt embedding plus `masked_scatter` path at `4.09e-6` max
absolute error on JFK with English forced output. `benchmarks/bench_decoder_input.py`
measured roughly 7.5-7.8 s for the original C++ GGML audio path with 8 CPU
threads, 0.62-0.64 s with `--audio-backend sched --device cpu`, 347 ms for
Torch CPU FP32, and 7.23 ms for Torch CUDA BF16. With GGML CUDA enabled, a
direct `qwen-asr-decoder-input --audio-backend sched --device gpu` run reported
about 196 ms hot decoder-input time.

For the first Qwen3 text decoder block, `benchmarks/check_text_layer0.py`
matches the eager Torch reference at `4.67e-5` max absolute error for the scalar
path and `4.69e-5` for the GGML graph on JFK with English forced output.
`benchmarks/bench_text_layer0.py` measured roughly 1.24 s best for the scalar
C++ text block, 310 ms best for the per-call GGML graph, 21.1 ms best for
`--backend sched --device cpu`, and 2.55 ms best for
`--backend sched --device gpu` after a 63 ms one-time GPU text weight upload.
Torch CPU FP32 with 8 threads measured 16.5 ms in the same run. The scheduled
GPU layer is now substantially faster than the CPU paths for this prompt-sized
block.

For the full prompt prefill logits, `benchmarks/check_text_prefill.py` matches
the eager Torch reference at `1.34e-4` max absolute error for the scalar path and
`1.33e-4` for the scheduled GGML backend on JFK, and both agree on the first
next-token id (`3036`). The GPU scheduled path matches the same top token with
larger CUDA numeric drift (`max_abs=0.0194` on JFK), so GPU validation uses
`--atol 2e-2` for layer checks and `--atol 3e-2` for full prefill while still
enforcing top-token agreement.
`benchmarks/bench_text_prefill.py` measured roughly 34.9 s for scalar C++ full
text prefill with 8 CPU threads, 481 ms for `--backend sched --device cpu`,
22.5 ms for `--backend sched --device gpu` after a 1.87 s one-time GPU text
weight upload, and 478 ms for Torch CPU FP32 with 8 threads.

For greedy generation, `benchmarks/check_text_generate.py` matches the Torch
reference for the first four JFK English tokens (`3036,773,11,847`, decoded as
`And so, my`) with both the scalar and scheduled KV-cache native paths. The
original recompute path took about 69.3 s for just two tokens. With scalar
`--kv-cache`, a four-token run measured about 35.8 s with 8 CPU threads. With
`--backend sched --kv-cache --device cpu`, the same four-token run measured
579 ms after a 768 ms one-time text weight/KV-cache initialization. With
`--backend sched --kv-cache --device gpu`, it measured 67.4 ms after a 1.69 s
one-time GPU text weight/KV-cache initialization, compared with 2.00 s for
Torch CPU FP32 with 8 threads. This is now an end-to-end qwentts.cpp-style
native audio-to-token/text path with scheduled GGML prompt prefill and
single-token KV-cache decode that primarily targets GPU when CUDA is enabled.

## Implementation Notes

Qwen3-ASR is not a Whisper-style encoder-decoder. It uses a chunked audio
transformer encoder and a Qwen3 decoder:

```text
16 kHz audio -> 128-bin log-mel -> per-chunk 3x Conv2D downsample
  -> windowed AuT encoder -> projector -> Qwen3 autoregressive decoder
```

The native C++ port target mirrors qwentts.cpp's shape: CMake build, CLI tools,
GGUF conversion, and a GGML runtime. The native path covers GGUF
metadata/tensor validation, Whisper log-mel features, audio geometry, Qwen BPE
prompt expansion, mapped GGUF tensor-data loading, scalar and GGML
implementations of the audio frontend, all audio transformer layers, decoder
input embedding assembly, and qwentts.cpp-style scheduled backends that can
target GGML CUDA with CPU scheduler fallback. The Qwen3 text decoder is
validated as a scalar CPU path through the first block and full prompt prefill
logits, including final normalization and the LM head, plus a minimal greedy
generation loop with byte-level BPE decode. The decoder also has qwentts.cpp-style
scheduled GGML/backend graphs for one-layer checks, full prompt prefill logits,
and KV-cache follow-up token decoding, including GPU-first backend selection
through `--device auto|cpu|gpu`.
The remaining native work is to reuse the scheduled text backend across a
longer-lived transcription API and extend that backend path through the
remaining standalone frontend tools.
