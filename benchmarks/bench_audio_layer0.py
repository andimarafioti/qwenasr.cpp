#!/usr/bin/env python3
"""Benchmark native audio encoder layer 0 against a Torch eager reference."""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from check_audio_cnn import TENSORS as FRONTEND_TENSORS
from check_audio_conv0 import (
    ROOT,
    _build_chunks,
    _dump_native_features,
    _load_checkpoint_tensor,
    _parse_kv,
    _run,
)
from check_audio_layer0 import LAYER0_TENSORS
from check_audio_prep import _audio_output_length


def _summarize(values: list[float]) -> tuple[float, float]:
    return min(values), statistics.mean(values)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _torch_dtype(name: str) -> torch.dtype:
    if name == "fp32":
        return torch.float32
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    raise ValueError(f"unsupported dtype: {name}")


def _bench_cpp(layer_bin: Path, gguf: Path, audio: Path, threads: int, repeat: int) -> list[float]:
    times = []
    for _ in range(repeat):
        meta = _parse_kv(
            _run(
                [
                    str(layer_bin),
                    str(gguf),
                    str(audio),
                    "--threads",
                    str(threads),
                ]
            )
        )
        times.append(float(meta["layer_ms"]))
    return times


def _make_positional(frames: int, hidden: int, device: torch.device, dtype: torch.dtype):
    half = hidden // 2
    log_timescale_increment = np.log(10000.0) / (half - 1)
    inv_timescales = torch.exp(
        -log_timescale_increment * torch.arange(half, device=device).float()
    )
    scaled_time = torch.arange(frames, device=device).float()[:, None] * inv_timescales[None, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1).to(dtype=dtype)


def _bench_torch(
    chunks,
    chunk_input_lengths: list[int],
    frontend: dict[str, torch.Tensor],
    weights: dict[str, torch.Tensor],
    heads: int,
    device: torch.device,
    dtype: torch.dtype,
    warmup: int,
    repeat: int,
) -> list[float]:
    chunks = chunks.to(device=device, dtype=dtype)
    frontend = {name: tensor.to(device=device, dtype=dtype) for name, tensor in frontend.items()}
    weights = {name: tensor.to(device=device, dtype=dtype) for name, tensor in weights.items()}

    def forward():
        x = F.gelu(F.conv2d(chunks, frontend["conv0_weight"], frontend["conv0_bias"], stride=2, padding=1))
        x = F.gelu(F.conv2d(x, frontend["conv1_weight"], frontend["conv1_bias"], stride=2, padding=1))
        x = F.gelu(F.conv2d(x, frontend["conv2_weight"], frontend["conv2_bias"], stride=2, padding=1))
        bsz, channels, freq, frames = x.shape
        x = x.permute(0, 3, 1, 2).contiguous().view(bsz, frames, channels * freq)
        x = F.linear(x, frontend["conv_out_weight"])
        x = x + _make_positional(frames, x.shape[-1], device, dtype).unsqueeze(0)
        rows = []
        for chunk, chunk_len in enumerate(chunk_input_lengths):
            rows.append(x[chunk, : _audio_output_length(chunk_len), :])
        hidden_states = torch.cat(rows, dim=0)

        residual = hidden_states
        x = F.layer_norm(
            hidden_states,
            (hidden_states.shape[-1],),
            weights["attn_norm_weight"],
            weights["attn_norm_bias"],
            eps=1e-5,
        )
        q = F.linear(x, weights["q_weight"], weights["q_bias"])
        k = F.linear(x, weights["k_weight"], weights["k_bias"])
        v = F.linear(x, weights["v_weight"], weights["v_bias"])
        seq_len, hidden = q.shape
        head_dim = hidden // heads
        q = q.reshape(seq_len, heads, head_dim).transpose(0, 1).unsqueeze(0)
        k = k.reshape(seq_len, heads, head_dim).transpose(0, 1).unsqueeze(0)
        v = v.reshape(seq_len, heads, head_dim).transpose(0, 1).unsqueeze(0)
        attn = torch.matmul(q, k.transpose(2, 3)) * (head_dim**-0.5)
        attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
        x = torch.matmul(attn, v).transpose(1, 2).reshape(seq_len, hidden).contiguous()
        x = F.linear(x, weights["out_weight"], weights["out_bias"])
        hidden_states = residual + x

        residual = hidden_states
        x = F.layer_norm(
            hidden_states,
            (hidden_states.shape[-1],),
            weights["ffn_norm_weight"],
            weights["ffn_norm_bias"],
            eps=1e-5,
        )
        x = F.linear(x, weights["up_weight"], weights["up_bias"])
        x = F.gelu(x)
        x = F.linear(x, weights["down_weight"], weights["down_bias"])
        return residual + x

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
    parser = argparse.ArgumentParser(description="Benchmark Qwen3-ASR audio layer0 implementations")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("gguf", type=Path)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--cpp-bin", default=str(ROOT / "build" / "qwen-asr-audio-layer"))
    parser.add_argument("--features-bin", default=str(ROOT / "build" / "qwen-asr-features"))
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--heads", type=int, default=14)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--torch-device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--torch-dtype", choices=("fp32", "bf16", "fp16"), default="fp32")
    parser.add_argument("--torch-threads", type=int, default=None)
    args = parser.parse_args()

    if args.torch_threads is not None:
        torch.set_num_threads(args.torch_threads)

    layer_bin = Path(args.cpp_bin)
    features_bin = Path(args.features_bin)
    if not layer_bin.is_file():
        raise SystemExit(f"C++ layer binary not found: {layer_bin}")
    if not features_bin.is_file():
        raise SystemExit(f"C++ feature binary not found: {features_bin}")

    features, feature_meta = _dump_native_features(features_bin, args.audio)
    chunks = torch.from_numpy(_build_chunks(features, feature_meta))
    chunk_window = int(feature_meta["chunk_window"])
    chunk_input_lengths = [
        min(chunk_window, features.shape[1] - chunk * chunk_window)
        for chunk in range(int(feature_meta["feature_chunks"]))
    ]
    frontend = {
        name: _load_checkpoint_tensor(args.checkpoint, tensor)
        for name, tensor in FRONTEND_TENSORS.items()
    }
    weights = {
        name: _load_checkpoint_tensor(args.checkpoint, tensor)
        for name, tensor in LAYER0_TENSORS.items()
    }

    cpp_times = _bench_cpp(layer_bin, args.gguf, args.audio, args.threads, args.repeat)
    torch_times = _bench_torch(
        chunks,
        chunk_input_lengths,
        frontend,
        weights,
        args.heads,
        torch.device(args.torch_device),
        _torch_dtype(args.torch_dtype),
        args.warmup,
        args.repeat,
    )

    cpp_best, cpp_mean = _summarize(cpp_times)
    torch_best, torch_mean = _summarize(torch_times)
    tokens = sum(_audio_output_length(length) for length in chunk_input_lengths)

    print(f"shape=({tokens}, {weights['attn_norm_weight'].shape[0]})")
    print(f"threads={args.threads}")
    print(f"heads={args.heads}")
    print(f"torch_device={args.torch_device}")
    print(f"torch_dtype={args.torch_dtype}")
    print(f"cpp_cpu_best_ms={cpp_best:.3f}")
    print(f"cpp_cpu_mean_ms={cpp_mean:.3f}")
    print(f"torch_best_ms={torch_best:.3f}")
    print(f"torch_mean_ms={torch_mean:.3f}")
    print("status=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
