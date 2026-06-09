#!/usr/bin/env python3
"""Benchmark native full Qwen3-ASR text prefill logits against Torch eager."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from bench_audio_layer0 import _sync, _torch_dtype
from check_audio_conv0 import ROOT, _load_checkpoint_tensor, _parse_kv, _run
from check_text_layer0 import _rms_norm, _text_layer_reference, _text_layer_tensors
from check_text_prefill import OUTPUT_NORM_NAME, OUTPUT_WEIGHT_NAME, _decoder_input_reference


def _summarize(values: list[float]) -> tuple[float, float]:
    return min(values), statistics.mean(values)


def _bench_cpp(
    text_bin: Path,
    gguf: Path,
    audio: Path,
    threads: int,
    repeat: int,
    text_backend: str,
    audio_backend: str,
    system: str,
    language: str,
) -> tuple[list[float], list[float], list[float]]:
    prefill_times = []
    decoder_times = []
    text_init_times = []
    for _ in range(repeat):
        cmd = [
            str(text_bin),
            str(gguf),
            str(audio),
            "--threads",
            str(threads),
            "--backend",
            text_backend,
            "--audio-backend",
            audio_backend,
            "--prefill",
        ]
        if system:
            cmd += ["--system", system]
        if language:
            cmd += ["--language", language]
        meta = _parse_kv(_run(cmd))
        prefill_times.append(float(meta["prefill_ms"]))
        decoder_times.append(float(meta["decoder_input_ms"]))
        text_init_times.append(float(meta.get("text_init_ms", 0.0)))
    return prefill_times, decoder_times, text_init_times


def _load_text_weights(checkpoint: Path, layers: int):
    layer_weights = [
        {
            name: _load_checkpoint_tensor(checkpoint, tensor)
            for name, tensor in _text_layer_tensors(layer).items()
        }
        for layer in range(layers)
    ]
    output_norm = _load_checkpoint_tensor(checkpoint, OUTPUT_NORM_NAME)
    output_weight = _load_checkpoint_tensor(checkpoint, OUTPUT_WEIGHT_NAME)
    return layer_weights, output_norm, output_weight


def _bench_torch(
    x: torch.Tensor,
    layer_weights: list[dict[str, torch.Tensor]],
    output_norm: torch.Tensor,
    output_weight: torch.Tensor,
    *,
    heads: int,
    kv_heads: int,
    head_dim: int,
    rope_theta: float,
    eps: float,
    device: torch.device,
    dtype: torch.dtype,
    warmup: int,
    repeat: int,
) -> list[float]:
    x = x.to(device=device, dtype=dtype)
    layer_weights = [
        {name: tensor.to(device=device, dtype=dtype) for name, tensor in weights.items()}
        for weights in layer_weights
    ]
    output_norm = output_norm.to(device=device, dtype=dtype)
    output_weight = output_weight.to(device=device, dtype=dtype)

    def forward():
        y = x
        for weights in layer_weights:
            y = _text_layer_reference(
                y,
                weights,
                heads=heads,
                kv_heads=kv_heads,
                head_dim=head_dim,
                rope_theta=rope_theta,
                eps=eps,
            )
        y = _rms_norm(y[-1, :], output_norm, eps)
        return F.linear(y.unsqueeze(0), output_weight).squeeze(0)

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
    parser = argparse.ArgumentParser(description="Benchmark Qwen3-ASR full text prefill logits")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("gguf", type=Path)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--cpp-bin", default=str(ROOT / "build" / "qwen-asr-text-layer"))
    parser.add_argument("--features-bin", default=str(ROOT / "build" / "qwen-asr-features"))
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--system", default="")
    parser.add_argument("--language", default="")
    parser.add_argument("--cpp-backends", nargs="+", choices=("scalar", "sched"), default=["scalar"])
    parser.add_argument("--cpp-audio-backends", nargs="+", choices=("ggml", "sched"), default=["sched"])
    parser.add_argument("--torch-device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--torch-dtype", choices=("fp32", "bf16", "fp16"), default="fp32")
    parser.add_argument("--torch-threads", type=int, default=None)
    args = parser.parse_args()

    if args.torch_threads is not None:
        torch.set_num_threads(args.torch_threads)

    text_bin = Path(args.cpp_bin)
    features_bin = Path(args.features_bin)
    if not text_bin.is_file():
        raise SystemExit(f"C++ text-layer binary not found: {text_bin}")
    if not features_bin.is_file():
        raise SystemExit(f"C++ feature binary not found: {features_bin}")

    cfg = json.loads((args.checkpoint / "config.json").read_text())
    text_cfg = cfg.get("thinker_config", cfg)["text_config"]
    layers = int(text_cfg["num_hidden_layers"])
    heads = int(text_cfg["num_attention_heads"])
    kv_heads = int(text_cfg["num_key_value_heads"])
    head_dim = int(text_cfg["head_dim"])
    rope_theta = float(text_cfg["rope_theta"])
    eps = float(text_cfg["rms_norm_eps"])

    x, hf_ids = _decoder_input_reference(args.checkpoint, features_bin, args.audio, args.system, args.language)
    layer_weights, output_norm, output_weight = _load_text_weights(args.checkpoint, layers)

    cpp_results = {}
    for text_backend in args.cpp_backends:
        for audio_backend in args.cpp_audio_backends:
            cpp_results[(text_backend, audio_backend)] = _bench_cpp(
                text_bin,
                args.gguf,
                args.audio,
                args.threads,
                args.repeat,
                text_backend,
                audio_backend,
                args.system,
                args.language,
            )
    torch_times = _bench_torch(
        x,
        layer_weights,
        output_norm,
        output_weight,
        heads=heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        rope_theta=rope_theta,
        eps=eps,
        device=torch.device(args.torch_device),
        dtype=_torch_dtype(args.torch_dtype),
        warmup=args.warmup,
        repeat=args.repeat,
    )

    torch_best, torch_mean = _summarize(torch_times)
    print(f"shape=({len(hf_ids)}, {output_weight.shape[0]})")
    print(f"threads={args.threads}")
    print(f"layers={layers}")
    print(f"heads={heads}")
    print(f"kv_heads={kv_heads}")
    print(f"head_dim={head_dim}")
    print(f"torch_device={args.torch_device}")
    print(f"torch_dtype={args.torch_dtype}")
    for (text_backend, audio_backend), (prefill_times, decoder_times, text_init_times) in cpp_results.items():
        cpp_best, cpp_mean = _summarize(prefill_times)
        decoder_best, decoder_mean = _summarize(decoder_times)
        text_init_best, text_init_mean = _summarize(text_init_times)
        prefix = f"cpp_{text_backend}_{audio_backend}"
        print(f"{prefix}_text_init_best_ms={text_init_best:.3f}")
        print(f"{prefix}_text_init_mean_ms={text_init_mean:.3f}")
        print(f"{prefix}_prefill_best_ms={cpp_best:.3f}")
        print(f"{prefix}_prefill_mean_ms={cpp_mean:.3f}")
        print(f"{prefix}_decoder_input_best_ms={decoder_best:.3f}")
        print(f"{prefix}_decoder_input_mean_ms={decoder_mean:.3f}")
    print(f"torch_prefill_best_ms={torch_best:.3f}")
    print(f"torch_prefill_mean_ms={torch_mean:.3f}")
    print("status=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
