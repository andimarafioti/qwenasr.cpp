#!/usr/bin/env python3
"""Benchmark native audio Conv2D layer 0 against a Torch reference."""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from check_audio_conv0 import (
    BIAS_NAME,
    ROOT,
    WEIGHT_NAME,
    _build_chunks,
    _dump_native_features,
    _load_checkpoint_tensor,
    _parse_kv,
    _run,
)


def _summarize(values: list[float]) -> tuple[float, float]:
    return min(values), statistics.mean(values)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _bench_cpp(conv_bin: Path, gguf: Path, audio: Path, backend: str, threads: int, repeat: int) -> list[float]:
    times = []
    for _ in range(repeat):
        meta = _parse_kv(
            _run(
                [
                    str(conv_bin),
                    str(gguf),
                    str(audio),
                    "--backend",
                    backend,
                    "--threads",
                    str(threads),
                ]
            )
        )
        times.append(float(meta["conv_ms"]))
    return times


def _torch_dtype(name: str) -> torch.dtype:
    if name == "fp32":
        return torch.float32
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    raise ValueError(f"unsupported dtype: {name}")


def _bench_torch(
    chunks,
    weight,
    bias,
    device: torch.device,
    dtype: torch.dtype,
    warmup: int,
    repeat: int,
) -> list[float]:
    chunks = chunks.to(device=device, dtype=dtype)
    weight = weight.to(device=device, dtype=dtype)
    bias = bias.to(device=device, dtype=dtype)

    with torch.no_grad():
        for _ in range(warmup):
            F.gelu(F.conv2d(chunks, weight, bias, stride=2, padding=1))
        _sync(device)

        times = []
        for _ in range(repeat):
            _sync(device)
            start = time.perf_counter()
            F.gelu(F.conv2d(chunks, weight, bias, stride=2, padding=1))
            _sync(device)
            times.append((time.perf_counter() - start) * 1000.0)
    return times


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark Qwen3-ASR audio conv0 implementations")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("gguf", type=Path)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--cpp-bin", default=str(ROOT / "build" / "qwen-asr-audio-conv"))
    parser.add_argument("--features-bin", default=str(ROOT / "build" / "qwen-asr-features"))
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--torch-device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--torch-dtype", choices=("fp32", "bf16", "fp16"), default="fp32")
    parser.add_argument("--torch-threads", type=int, default=None)
    args = parser.parse_args()

    if args.torch_threads is not None:
        torch.set_num_threads(args.torch_threads)

    conv_bin = Path(args.cpp_bin)
    features_bin = Path(args.features_bin)
    if not conv_bin.is_file():
        raise SystemExit(f"C++ conv binary not found: {conv_bin}")
    if not features_bin.is_file():
        raise SystemExit(f"C++ feature binary not found: {features_bin}")

    features, feature_meta = _dump_native_features(features_bin, args.audio)
    chunks = torch.from_numpy(_build_chunks(features, feature_meta))
    weight = _load_checkpoint_tensor(args.checkpoint, WEIGHT_NAME)
    bias = _load_checkpoint_tensor(args.checkpoint, BIAS_NAME)

    cpp_ggml = _bench_cpp(conv_bin, args.gguf, args.audio, "ggml", args.threads, args.repeat)
    cpp_scalar = _bench_cpp(conv_bin, args.gguf, args.audio, "scalar", args.threads, args.repeat)
    torch_times = _bench_torch(
        chunks,
        weight,
        bias,
        torch.device(args.torch_device),
        _torch_dtype(args.torch_dtype),
        args.warmup,
        args.repeat,
    )

    ggml_best, ggml_mean = _summarize(cpp_ggml)
    scalar_best, scalar_mean = _summarize(cpp_scalar)
    torch_best, torch_mean = _summarize(torch_times)

    print(f"shape=({feature_meta['feature_chunks']}, 480, 64, 50)")
    print(f"threads={args.threads}")
    print(f"torch_device={args.torch_device}")
    print(f"torch_dtype={args.torch_dtype}")
    print(f"cpp_ggml_best_ms={ggml_best:.3f}")
    print(f"cpp_ggml_mean_ms={ggml_mean:.3f}")
    print(f"cpp_scalar_best_ms={scalar_best:.3f}")
    print(f"cpp_scalar_mean_ms={scalar_mean:.3f}")
    print(f"torch_best_ms={torch_best:.3f}")
    print(f"torch_mean_ms={torch_mean:.3f}")
    print("status=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
