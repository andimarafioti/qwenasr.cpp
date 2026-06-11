#!/usr/bin/env python3
"""Benchmark native Qwen3-ASR text decoder layer 0 against a Torch eager reference."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from bench_audio_layer0 import _sync, _torch_dtype
from check_audio_conv0 import ROOT, _dump_native_features, _load_checkpoint_tensor, _parse_kv, _run
from check_decoder_input import AUDIO_TOKEN_ID, TEXT_EMBED_NAME, _hf_ids, _torch_audio_encoder
from check_text_layer0 import _text_layer_reference, _text_layer_tensors


def _summarize(values: list[float]) -> tuple[float, float]:
    return min(values), statistics.mean(values)


def _bench_cpp(
    text_bin: Path,
    gguf: Path,
    audio: Path,
    threads: int,
    layer: int,
    repeat: int,
    text_backend: str,
    audio_backend: str,
    device: str,
    system: str,
    language: str,
) -> tuple[list[float], list[float], list[float]]:
    text_times = []
    decoder_times = []
    text_init_times = []
    for _ in range(repeat):
        cmd = [
            str(text_bin),
            str(gguf),
            str(audio),
            "--threads",
            str(threads),
            "--layer",
            str(layer),
            "--backend",
            text_backend,
            "--audio-backend",
            audio_backend,
            "--device",
            device,
        ]
        if system:
            cmd += ["--system", system]
        if language:
            cmd += ["--language", language]
        meta = _parse_kv(_run(cmd))
        text_times.append(float(meta["text_layer_ms"]))
        decoder_times.append(float(meta["decoder_input_ms"]))
        text_init_times.append(float(meta.get("text_init_ms", 0.0)))
    return text_times, decoder_times, text_init_times


def _bench_torch(
    x: torch.Tensor,
    weights: dict[str, torch.Tensor],
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
    weights = {name: tensor.to(device=device, dtype=dtype) for name, tensor in weights.items()}

    def forward():
        return _text_layer_reference(
            x,
            weights,
            heads=heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            rope_theta=rope_theta,
            eps=eps,
        )

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


def _decoder_input_reference(
    checkpoint: Path,
    features_bin: Path,
    audio: Path,
    system: str,
    language: str,
    audio_layers: int,
    audio_heads: int,
) -> tuple[torch.Tensor, int, list[int]]:
    features, feature_meta = _dump_native_features(features_bin, audio)
    audio_features = _torch_audio_encoder(checkpoint, features, feature_meta, audio_layers, audio_heads)
    hf_ids = _hf_ids(checkpoint, audio_features.shape[0], system, language)
    embeddings = _load_checkpoint_tensor(checkpoint, TEXT_EMBED_NAME)
    x = embeddings[hf_ids].clone()
    ids_tensor = torch.tensor(hf_ids, dtype=torch.long)
    audio_positions = torch.nonzero(ids_tensor == AUDIO_TOKEN_ID, as_tuple=False).flatten()
    if audio_positions.numel() != audio_features.shape[0]:
        raise SystemExit(
            f"audio placeholder mismatch: prompt={audio_positions.numel()} audio={audio_features.shape[0]}"
        )
    x[audio_positions, :] = audio_features
    return x, audio_features.shape[0], hf_ids


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark Qwen3-ASR text decoder layer 0 implementations")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("gguf", type=Path)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--cpp-bin", default=str(ROOT / "build" / "qwen-asr-text-layer"))
    parser.add_argument("--features-bin", default=str(ROOT / "build" / "qwen-asr-features"))
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--audio-layers", type=int, default=18)
    parser.add_argument("--audio-heads", type=int, default=14)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--system", default="")
    parser.add_argument("--language", default="")
    parser.add_argument("--cpp-backends", nargs="+", choices=("scalar", "ggml", "sched"), default=["scalar"])
    parser.add_argument("--cpp-audio-backends", nargs="+", choices=("ggml", "sched"), default=["sched"])
    parser.add_argument("--cpp-devices", nargs="+", choices=("auto", "cpu", "gpu", "cuda"), default=["auto"])
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
    heads = int(text_cfg["num_attention_heads"])
    kv_heads = int(text_cfg["num_key_value_heads"])
    head_dim = int(text_cfg["head_dim"])
    rope_theta = float(text_cfg["rope_theta"])
    eps = float(text_cfg["rms_norm_eps"])

    x, audio_tokens, ids = _decoder_input_reference(
        args.checkpoint,
        features_bin,
        args.audio,
        args.system,
        args.language,
        args.audio_layers,
        args.audio_heads,
    )
    weights = {
        name: _load_checkpoint_tensor(args.checkpoint, tensor)
        for name, tensor in _text_layer_tensors(args.layer).items()
    }

    cpp_results = {}
    for text_backend in args.cpp_backends:
        for audio_backend in args.cpp_audio_backends:
            for device in args.cpp_devices:
                cpp_results[(text_backend, audio_backend, device)] = _bench_cpp(
                    text_bin,
                    args.gguf,
                    args.audio,
                    args.threads,
                    args.layer,
                    args.repeat,
                    text_backend,
                    audio_backend,
                    device,
                    args.system,
                    args.language,
                )
    torch_times = _bench_torch(
        x,
        weights,
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
    print(f"shape=({len(ids)}, {x.shape[1]})")
    print(f"audio_tokens={audio_tokens}")
    print(f"threads={args.threads}")
    print(f"layer={args.layer}")
    print(f"heads={heads}")
    print(f"kv_heads={kv_heads}")
    print(f"head_dim={head_dim}")
    print(f"torch_device={args.torch_device}")
    print(f"torch_dtype={args.torch_dtype}")
    for (text_backend, audio_backend, device), (text_times, decoder_times, text_init_times) in cpp_results.items():
        cpp_best, cpp_mean = _summarize(text_times)
        decoder_best, decoder_mean = _summarize(decoder_times)
        text_init_best, text_init_mean = _summarize(text_init_times)
        prefix = f"cpp_{text_backend}_{audio_backend}_{device}"
        print(f"{prefix}_text_init_best_ms={text_init_best:.3f}")
        print(f"{prefix}_text_init_mean_ms={text_init_mean:.3f}")
        print(f"{prefix}_text_layer_best_ms={cpp_best:.3f}")
        print(f"{prefix}_text_layer_mean_ms={cpp_mean:.3f}")
        print(f"{prefix}_decoder_input_best_ms={decoder_best:.3f}")
        print(f"{prefix}_decoder_input_mean_ms={decoder_mean:.3f}")
    print(f"torch_text_layer_best_ms={torch_best:.3f}")
    print(f"torch_text_layer_mean_ms={torch_mean:.3f}")
    print("status=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
