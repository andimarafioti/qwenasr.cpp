#!/usr/bin/env python3
"""Compare native C++ audio CNN plus conv_out against HF checkpoint weights."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import numpy as np

from check_audio_conv0 import (
    ROOT,
    _build_chunks,
    _dump_native_features,
    _load_checkpoint_tensor,
    _parse_kv,
    _run,
)


TENSORS = {
    "conv0_weight": "thinker.audio_tower.conv2d1.weight",
    "conv0_bias": "thinker.audio_tower.conv2d1.bias",
    "conv1_weight": "thinker.audio_tower.conv2d2.weight",
    "conv1_bias": "thinker.audio_tower.conv2d2.bias",
    "conv2_weight": "thinker.audio_tower.conv2d3.weight",
    "conv2_bias": "thinker.audio_tower.conv2d3.bias",
    "conv_out_weight": "thinker.audio_tower.conv_out.weight",
}


def _dump_native_cnn(
    cnn_bin: Path,
    gguf: Path,
    audio: Path,
    n_threads: int,
) -> tuple[np.ndarray, dict[str, str]]:
    with tempfile.NamedTemporaryFile(suffix=".f32", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        meta = _parse_kv(
            _run(
                [
                    str(cnn_bin),
                    str(gguf),
                    str(audio),
                    "--threads",
                    str(n_threads),
                    "--out",
                    str(tmp_path),
                ]
            )
        )
        cnn = np.fromfile(tmp_path, dtype=np.float32).reshape(
            int(meta["chunks"]),
            int(meta["frames"]),
            int(meta["hidden"]),
        )
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
    return cnn, meta


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate native Qwen3-ASR audio CNN")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("gguf", type=Path)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--cpp-bin", default=str(ROOT / "build" / "qwen-asr-audio-cnn"))
    parser.add_argument("--features-bin", default=str(ROOT / "build" / "qwen-asr-features"))
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--atol", type=float, default=5e-4)
    args = parser.parse_args()

    cnn_bin = Path(args.cpp_bin)
    features_bin = Path(args.features_bin)
    if not cnn_bin.is_file():
        raise SystemExit(f"C++ CNN binary not found: {cnn_bin}")
    if not features_bin.is_file():
        raise SystemExit(f"C++ feature binary not found: {features_bin}")

    features, feature_meta = _dump_native_features(features_bin, args.audio)
    native, cnn_meta = _dump_native_cnn(cnn_bin, args.gguf, args.audio, args.threads)
    ref_chunks = _build_chunks(features, feature_meta)

    try:
        import torch
        import torch.nn.functional as F
    except ImportError as exc:
        raise SystemExit("torch is required for the PyTorch reference CNN") from exc

    weights = {name: _load_checkpoint_tensor(args.checkpoint, tensor) for name, tensor in TENSORS.items()}
    with torch.no_grad():
        x = torch.from_numpy(ref_chunks)
        x = F.gelu(F.conv2d(x, weights["conv0_weight"], weights["conv0_bias"], stride=2, padding=1))
        x = F.gelu(F.conv2d(x, weights["conv1_weight"], weights["conv1_bias"], stride=2, padding=1))
        x = F.gelu(F.conv2d(x, weights["conv2_weight"], weights["conv2_bias"], stride=2, padding=1))
        bsz, channels, freq, frames = x.shape
        x = x.permute(0, 3, 1, 2).contiguous().view(bsz, frames, channels * freq)
        ref = F.linear(x, weights["conv_out_weight"]).numpy()

    if native.shape != ref.shape:
        raise SystemExit(f"shape mismatch: native={native.shape} torch={ref.shape}")

    diff = np.abs(native - ref)
    max_abs = float(diff.max())
    print(f"shape={native.shape}")
    print(f"native_backend={cnn_meta['backend']}")
    print(f"native_cnn_ms={float(cnn_meta['cnn_ms']):.3f}")
    print(f"feature_frames={feature_meta['frames']}")
    print(f"chunks={cnn_meta['chunks']}")
    print(f"max_abs={max_abs:.8f}")
    print(f"mean_abs={float(diff.mean()):.8f}")
    print(f"native_mean={float(native.mean()):.8f}")
    print(f"torch_mean={float(ref.mean()):.8f}")
    if max_abs > args.atol:
        raise SystemExit(f"audio CNN mismatch: max_abs {max_abs:.8f} > atol {args.atol}")
    print("status=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
