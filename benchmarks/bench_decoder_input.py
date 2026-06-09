#!/usr/bin/env python3
"""Benchmark native decoder input assembly against a Torch eager reference."""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from bench_audio_encoder import _audio_layer_forward
from bench_audio_layer0 import _torch_dtype, _sync
from check_audio_cnn import TENSORS as FRONTEND_TENSORS
from check_audio_conv0 import (
    ROOT,
    _build_chunks,
    _dump_native_features,
    _load_checkpoint_tensor,
    _parse_kv,
    _run,
)
from check_audio_encoder import PROJECTOR_TENSORS, _layer_tensors
from check_decoder_input import AUDIO_TOKEN_ID, TEXT_EMBED_NAME, _hf_ids
from check_audio_prep import _audio_output_length


def _summarize(values: list[float]) -> tuple[float, float]:
    return min(values), statistics.mean(values)


def _bench_cpp(
    decoder_bin: Path,
    gguf: Path,
    audio: Path,
    threads: int,
    repeat: int,
    system: str,
    language: str,
) -> list[float]:
    times = []
    for _ in range(repeat):
        cmd = [
            str(decoder_bin),
            str(gguf),
            str(audio),
            "--threads",
            str(threads),
        ]
        if system:
            cmd += ["--system", system]
        if language:
            cmd += ["--language", language]
        meta = _parse_kv(_run(cmd))
        times.append(float(meta["decoder_input_ms"]))
    return times


def _bench_torch(
    chunks,
    chunk_input_lengths: list[int],
    ids: list[int],
    frontend: dict[str, torch.Tensor],
    layers: list[dict[str, torch.Tensor]],
    projector: dict[str, torch.Tensor],
    embeddings: torch.Tensor,
    heads: int,
    device: torch.device,
    dtype: torch.dtype,
    warmup: int,
    repeat: int,
) -> list[float]:
    local_chunks = chunks.to(device=device, dtype=dtype)
    local_frontend = {name: tensor.to(device=device, dtype=dtype) for name, tensor in frontend.items()}
    local_layers = [
        {name: tensor.to(device=device, dtype=dtype) for name, tensor in layer.items()}
        for layer in layers
    ]
    local_projector = {name: tensor.to(device=device, dtype=dtype) for name, tensor in projector.items()}
    ids_tensor = torch.tensor(ids, dtype=torch.long, device=device)
    audio_positions = torch.nonzero(ids_tensor == AUDIO_TOKEN_ID, as_tuple=False).flatten()
    embeddings = embeddings.to(device=device, dtype=dtype)

    def forward():
        from bench_audio_layer0 import _make_positional

        x = F.gelu(F.conv2d(local_chunks, local_frontend["conv0_weight"], local_frontend["conv0_bias"], stride=2, padding=1))
        x = F.gelu(F.conv2d(x, local_frontend["conv1_weight"], local_frontend["conv1_bias"], stride=2, padding=1))
        x = F.gelu(F.conv2d(x, local_frontend["conv2_weight"], local_frontend["conv2_bias"], stride=2, padding=1))
        bsz, channels, freq, frames = x.shape
        x = x.permute(0, 3, 1, 2).contiguous().view(bsz, frames, channels * freq)
        x = F.linear(x, local_frontend["conv_out_weight"])
        x = x + _make_positional(frames, x.shape[-1], device, dtype).unsqueeze(0)
        rows = []
        for chunk, chunk_len in enumerate(chunk_input_lengths):
            rows.append(x[chunk, : _audio_output_length(chunk_len), :])
        hidden_states = torch.cat(rows, dim=0)
        for weights in local_layers:
            hidden_states = _audio_layer_forward(hidden_states, weights, heads)
        x = F.layer_norm(
            hidden_states,
            (hidden_states.shape[-1],),
            local_projector["post_norm_weight"],
            local_projector["post_norm_bias"],
            eps=1e-5,
        )
        x = F.linear(x, local_projector["proj0_weight"], local_projector["proj0_bias"])
        x = F.gelu(x)
        audio_features = F.linear(x, local_projector["proj1_weight"], local_projector["proj1_bias"])

        out = embeddings.index_select(0, ids_tensor).clone()
        out[audio_positions, :] = audio_features
        return out

    with torch.no_grad():
        for _ in range(warmup):
            forward()
        _sync(device)

        times = []
        for _ in range(repeat):
            _sync(device)
            start = time.perf_counter()
            forward()
            _sync(device)
            times.append((time.perf_counter() - start) * 1000.0)
    return times


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark Qwen3-ASR decoder input assembly")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("gguf", type=Path)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--cpp-bin", default=str(ROOT / "build" / "qwen-asr-decoder-input"))
    parser.add_argument("--features-bin", default=str(ROOT / "build" / "qwen-asr-features"))
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--layers", type=int, default=18)
    parser.add_argument("--heads", type=int, default=14)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--system", default="")
    parser.add_argument("--language", default="")
    parser.add_argument("--torch-device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--torch-dtype", choices=("fp32", "bf16", "fp16"), default="fp32")
    parser.add_argument("--torch-threads", type=int, default=None)
    args = parser.parse_args()

    if args.torch_threads is not None:
        torch.set_num_threads(args.torch_threads)

    decoder_bin = Path(args.cpp_bin)
    features_bin = Path(args.features_bin)
    if not decoder_bin.is_file():
        raise SystemExit(f"C++ decoder-input binary not found: {decoder_bin}")
    if not features_bin.is_file():
        raise SystemExit(f"C++ feature binary not found: {features_bin}")

    features, feature_meta = _dump_native_features(features_bin, args.audio)
    chunks = torch.from_numpy(_build_chunks(features, feature_meta))
    chunk_window = int(feature_meta["chunk_window"])
    chunk_input_lengths = [
        min(chunk_window, features.shape[1] - chunk * chunk_window)
        for chunk in range(int(feature_meta["feature_chunks"]))
    ]
    audio_tokens = sum(_audio_output_length(length) for length in chunk_input_lengths)
    ids = _hf_ids(args.checkpoint, audio_tokens, args.system, args.language)

    frontend = {
        name: _load_checkpoint_tensor(args.checkpoint, tensor)
        for name, tensor in FRONTEND_TENSORS.items()
    }
    layers = [
        {
            name: _load_checkpoint_tensor(args.checkpoint, tensor)
            for name, tensor in _layer_tensors(layer).items()
        }
        for layer in range(args.layers)
    ]
    projector = {
        name: _load_checkpoint_tensor(args.checkpoint, tensor)
        for name, tensor in PROJECTOR_TENSORS.items()
    }
    embeddings = _load_checkpoint_tensor(args.checkpoint, TEXT_EMBED_NAME)

    cpp_times = _bench_cpp(
        decoder_bin,
        args.gguf,
        args.audio,
        args.threads,
        args.repeat,
        args.system,
        args.language,
    )
    torch_times = _bench_torch(
        chunks,
        chunk_input_lengths,
        ids,
        frontend,
        layers,
        projector,
        embeddings,
        args.heads,
        torch.device(args.torch_device),
        _torch_dtype(args.torch_dtype),
        args.warmup,
        args.repeat,
    )

    cpp_best, cpp_mean = _summarize(cpp_times)
    torch_best, torch_mean = _summarize(torch_times)
    print(f"shape=({len(ids)}, {embeddings.shape[1]})")
    print(f"audio_tokens={audio_tokens}")
    print(f"threads={args.threads}")
    print(f"layers={args.layers}")
    print(f"heads={args.heads}")
    print(f"torch_device={args.torch_device}")
    print(f"torch_dtype={args.torch_dtype}")
    print(f"cpp_ggml_best_ms={cpp_best:.3f}")
    print(f"cpp_ggml_mean_ms={cpp_mean:.3f}")
    print(f"torch_best_ms={torch_best:.3f}")
    print(f"torch_mean_ms={torch_mean:.3f}")
    print("status=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
