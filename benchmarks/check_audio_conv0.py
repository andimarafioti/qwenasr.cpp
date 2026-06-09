#!/usr/bin/env python3
"""Compare native C++ audio Conv2D layer 0 against the HF checkpoint weights."""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WEIGHT_NAME = "thinker.audio_tower.conv2d1.weight"
BIAS_NAME = "thinker.audio_tower.conv2d1.bias"


def _parse_kv(text: str) -> dict[str, str]:
    out = {}
    for line in text.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def _run(args: list[str]) -> str:
    proc = subprocess.run(
        args,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )
    return proc.stdout


def _load_checkpoint_tensor(checkpoint: Path, name: str):
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise SystemExit("safetensors is required to read checkpoint weights") from exc

    files = sorted(checkpoint.glob("*.safetensors"))
    if not files:
        raise SystemExit(f"no safetensors files found in {checkpoint}")

    for file in files:
        with safe_open(file, framework="pt", device="cpu") as handle:
            if name in handle.keys():
                return handle.get_tensor(name).float()
    raise SystemExit(f"checkpoint tensor not found: {name}")


def _dump_native_features(features_bin: Path, audio: Path) -> tuple[np.ndarray, dict[str, str]]:
    with tempfile.NamedTemporaryFile(suffix=".f32", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        meta = _parse_kv(_run([str(features_bin), str(audio), "--out", str(tmp_path)]))
        features = np.fromfile(tmp_path, dtype=np.float32).reshape(
            int(meta["mels"]),
            int(meta["frames"]),
        )
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
    return features, meta


def _dump_native_conv(conv_bin: Path, gguf: Path, audio: Path) -> tuple[np.ndarray, dict[str, str]]:
    with tempfile.NamedTemporaryFile(suffix=".f32", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        meta = _parse_kv(_run([str(conv_bin), str(gguf), str(audio), "--out", str(tmp_path)]))
        conv = np.fromfile(tmp_path, dtype=np.float32).reshape(
            int(meta["chunks"]),
            int(meta["channels"]),
            int(meta["freq"]),
            int(meta["frames"]),
        )
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
    return conv, meta


def _build_chunks(features: np.ndarray, meta: dict[str, str]) -> np.ndarray:
    chunk_window = int(meta["chunk_window"])
    n_chunks = int(meta["feature_chunks"])
    max_len = int(meta["max_chunk_input_len"])

    chunks = np.zeros((n_chunks, 1, features.shape[0], max_len), dtype=np.float32)
    for chunk in range(n_chunks):
        start = chunk * chunk_window
        stop = min(start + chunk_window, features.shape[1])
        if start < stop:
            chunks[chunk, 0, :, : stop - start] = features[:, start:stop]
    return chunks


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate native Qwen3-ASR audio conv0")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("gguf", type=Path)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--cpp-bin", default=str(ROOT / "build" / "qwen-asr-audio-conv"))
    parser.add_argument("--features-bin", default=str(ROOT / "build" / "qwen-asr-features"))
    parser.add_argument("--atol", type=float, default=2e-5)
    args = parser.parse_args()

    conv_bin = Path(args.cpp_bin)
    features_bin = Path(args.features_bin)
    if not conv_bin.is_file():
        raise SystemExit(f"C++ conv binary not found: {conv_bin}")
    if not features_bin.is_file():
        raise SystemExit(f"C++ feature binary not found: {features_bin}")

    features, feature_meta = _dump_native_features(features_bin, args.audio)
    native, conv_meta = _dump_native_conv(conv_bin, args.gguf, args.audio)
    ref_chunks = _build_chunks(features, feature_meta)

    try:
        import torch
        import torch.nn.functional as F
    except ImportError as exc:
        raise SystemExit("torch is required for the PyTorch reference conv") from exc

    weight = _load_checkpoint_tensor(args.checkpoint, WEIGHT_NAME)
    bias = _load_checkpoint_tensor(args.checkpoint, BIAS_NAME)
    with torch.no_grad():
        ref = F.gelu(
            F.conv2d(torch.from_numpy(ref_chunks), weight, bias, stride=2, padding=1)
        ).numpy()

    if native.shape != ref.shape:
        raise SystemExit(f"shape mismatch: native={native.shape} torch={ref.shape}")

    diff = np.abs(native - ref)
    max_abs = float(diff.max())
    print(f"shape={native.shape}")
    print(f"feature_frames={feature_meta['frames']}")
    print(f"chunks={conv_meta['chunks']}")
    print(f"max_abs={max_abs:.8f}")
    print(f"mean_abs={float(diff.mean()):.8f}")
    print(f"native_mean={float(native.mean()):.8f}")
    print(f"torch_mean={float(ref.mean()):.8f}")
    if max_abs > args.atol:
        raise SystemExit(f"conv0 mismatch: max_abs {max_abs:.8f} > atol {args.atol}")
    print("status=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
