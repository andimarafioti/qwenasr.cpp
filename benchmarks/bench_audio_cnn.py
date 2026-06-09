#!/usr/bin/env python3
"""Benchmark native audio CNN plus conv_out against a Torch reference."""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from check_audio_cnn import TENSORS
from check_audio_conv0 import (
    ROOT,
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


def _torch_dtype(name: str) -> torch.dtype:
    if name == "fp32":
        return torch.float32
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    raise ValueError(f"unsupported dtype: {name}")


def _bench_cpp(cnn_bin: Path, gguf: Path, audio: Path, threads: int, repeat: int) -> list[float]:
    times = []
    for _ in range(repeat):
        meta = _parse_kv(
            _run(
                [
                    str(cnn_bin),
                    str(gguf),
                    str(audio),
                    "--threads",
                    str(threads),
                ]
            )
        )
        times.append(float(meta["cnn_ms"]))
    return times


def _bench_torch(
    chunks,
    weights: dict[str, torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
    warmup: int,
    repeat: int,
) -> list[float]:
    chunks = chunks.to(device=device, dtype=dtype)
    weights = {name: tensor.to(device=device, dtype=dtype) for name, tensor in weights.items()}

    def forward():
        x = F.gelu(F.conv2d(chunks, weights["conv0_weight"], weights["conv0_bias"], stride=2, padding=1))
        x = F.gelu(F.conv2d(x, weights["conv1_weight"], weights["conv1_bias"], stride=2, padding=1))
        x = F.gelu(F.conv2d(x, weights["conv2_weight"], weights["conv2_bias"], stride=2, padding=1))
        bsz, channels, freq, frames = x.shape
        x = x.permute(0, 3, 1, 2).contiguous().view(bsz, frames, channels * freq)
        return F.linear(x, weights["conv_out_weight"])

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
    parser = argparse.ArgumentParser(description="Benchmark Qwen3-ASR audio CNN implementations")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("gguf", type=Path)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--cpp-bin", default=str(ROOT / "build" / "qwen-asr-audio-cnn"))
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

    cnn_bin = Path(args.cpp_bin)
    features_bin = Path(args.features_bin)
    if not cnn_bin.is_file():
        raise SystemExit(f"C++ CNN binary not found: {cnn_bin}")
    if not features_bin.is_file():
        raise SystemExit(f"C++ feature binary not found: {features_bin}")

    features, feature_meta = _dump_native_features(features_bin, args.audio)
    chunks = torch.from_numpy(_build_chunks(features, feature_meta))
    weights = {name: _load_checkpoint_tensor(args.checkpoint, tensor) for name, tensor in TENSORS.items()}

    cpp_times = _bench_cpp(cnn_bin, args.gguf, args.audio, args.threads, args.repeat)
    torch_times = _bench_torch(
        chunks,
        weights,
        torch.device(args.torch_device),
        _torch_dtype(args.torch_dtype),
        args.warmup,
        args.repeat,
    )

    cpp_best, cpp_mean = _summarize(cpp_times)
    torch_best, torch_mean = _summarize(torch_times)

    print(
        f"shape=({feature_meta['feature_chunks']}, "
        f"{feature_meta['max_chunk_output_len']}, {weights['conv_out_weight'].shape[0]})"
    )
    print(f"threads={args.threads}")
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
