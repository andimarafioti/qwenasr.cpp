#!/usr/bin/env python3
"""Benchmark native recompute greedy Qwen3-ASR generation against Torch eager."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from bench_audio_layer0 import _sync, _torch_dtype
from check_audio_conv0 import ROOT, _parse_kv, _run
from check_text_generate import _load_text_stack, _prefill_logits_loaded
from check_text_prefill import _decoder_input_reference


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
    device: str,
    decode_backend: str,
    max_new_tokens: int,
    system: str,
    language: str,
) -> tuple[list[float], list[float], list[float]]:
    generate_times = []
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
            "--device",
            device,
            "--generate",
            str(max_new_tokens),
        ]
        if decode_backend == "kv-cache":
            cmd.append("--kv-cache")
        if system:
            cmd += ["--system", system]
        if language:
            cmd += ["--language", language]
        meta = _parse_kv(_run(cmd))
        generate_times.append(float(meta["generate_ms"]))
        decoder_times.append(float(meta["decoder_input_ms"]))
        text_init_times.append(float(meta.get("text_init_ms", 0.0)))
    return generate_times, decoder_times, text_init_times


def _move_weights(layer_weights, output_norm, output_weight, embeddings, device: torch.device, dtype: torch.dtype):
    layer_weights = [
        {name: tensor.to(device=device, dtype=dtype) for name, tensor in weights.items()}
        for weights in layer_weights
    ]
    return (
        layer_weights,
        output_norm.to(device=device, dtype=dtype),
        output_weight.to(device=device, dtype=dtype),
        embeddings.to(device=device, dtype=dtype),
    )


def _bench_torch(
    x,
    layer_weights,
    output_norm,
    output_weight,
    embeddings,
    text_cfg: dict,
    max_new_tokens: int,
    device: torch.device,
    dtype: torch.dtype,
    warmup: int,
    repeat: int,
) -> list[float]:
    x = x.to(device=device, dtype=dtype)
    layer_weights, output_norm, output_weight, embeddings = _move_weights(
        layer_weights,
        output_norm,
        output_weight,
        embeddings,
        device,
        dtype,
    )

    def forward():
        y = x
        generated = []
        for _ in range(max_new_tokens):
            logits = _prefill_logits_loaded(y, layer_weights, output_norm, output_weight, text_cfg)
            next_id = int(logits.argmax().item())
            generated.append(next_id)
            if next_id in {151643, 151645}:
                break
            y = torch.cat([y, embeddings[next_id].view(1, -1)], dim=0)
        return generated

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
    parser = argparse.ArgumentParser(description="Benchmark Qwen3-ASR greedy text generation")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("gguf", type=Path)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--cpp-bin", default=str(ROOT / "build" / "qwen-asr-text-layer"))
    parser.add_argument("--features-bin", default=str(ROOT / "build" / "qwen-asr-features"))
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--system", default="")
    parser.add_argument("--language", default="")
    parser.add_argument("--cpp-backends", nargs="+", choices=("scalar", "sched"), default=["scalar"])
    parser.add_argument("--cpp-audio-backends", nargs="+", choices=("ggml", "sched"), default=["sched"])
    parser.add_argument("--cpp-devices", nargs="+", choices=("auto", "cpu", "gpu", "cuda"), default=["auto"])
    parser.add_argument("--cpp-decode-backends", nargs="+", choices=("recompute", "kv-cache"), default=["kv-cache"])
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

    x, hf_ids = _decoder_input_reference(args.checkpoint, features_bin, args.audio, args.system, args.language)
    layer_weights, output_norm, output_weight, embeddings = _load_text_stack(args.checkpoint, layers)

    cpp_results = {}
    for text_backend in args.cpp_backends:
        for audio_backend in args.cpp_audio_backends:
            for device in args.cpp_devices:
                for decode_backend in args.cpp_decode_backends:
                    if text_backend == "sched" and decode_backend != "kv-cache":
                        continue
                    cpp_results[(text_backend, audio_backend, device, decode_backend)] = _bench_cpp(
                        text_bin,
                        args.gguf,
                        args.audio,
                        args.threads,
                        args.repeat,
                        text_backend,
                        audio_backend,
                        device,
                        decode_backend,
                        args.max_new_tokens,
                        args.system,
                        args.language,
                    )
    torch_times = _bench_torch(
        x,
        layer_weights,
        output_norm,
        output_weight,
        embeddings,
        text_cfg,
        args.max_new_tokens,
        torch.device(args.torch_device),
        _torch_dtype(args.torch_dtype),
        args.warmup,
        args.repeat,
    )

    torch_best, torch_mean = _summarize(torch_times)
    print(f"prompt_tokens={len(hf_ids)}")
    print(f"requested_tokens={args.max_new_tokens}")
    print(f"threads={args.threads}")
    print(f"layers={layers}")
    print(f"torch_device={args.torch_device}")
    print(f"torch_dtype={args.torch_dtype}")
    for (text_backend, audio_backend, device, decode_backend), (generate_times, decoder_times, text_init_times) in cpp_results.items():
        prefix = f"cpp_{text_backend}_{audio_backend}_{device}_{decode_backend.replace('-', '_')}"
        cpp_best, cpp_mean = _summarize(generate_times)
        decoder_best, decoder_mean = _summarize(decoder_times)
        text_init_best, text_init_mean = _summarize(text_init_times)
        print(f"{prefix}_text_init_best_ms={text_init_best:.3f}")
        print(f"{prefix}_text_init_mean_ms={text_init_mean:.3f}")
        print(f"{prefix}_generate_best_ms={cpp_best:.3f}")
        print(f"{prefix}_generate_mean_ms={cpp_mean:.3f}")
        print(f"{prefix}_decoder_input_best_ms={decoder_best:.3f}")
        print(f"{prefix}_decoder_input_mean_ms={decoder_mean:.3f}")
    print(f"torch_generate_best_ms={torch_best:.3f}")
    print(f"torch_generate_mean_ms={torch_mean:.3f}")
    print("status=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
