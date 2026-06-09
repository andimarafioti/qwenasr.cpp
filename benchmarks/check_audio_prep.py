#!/usr/bin/env python3
"""Compare native packed audio pre-transformer states against a Torch reference."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import numpy as np

from check_audio_cnn import TENSORS
from check_audio_conv0 import (
    ROOT,
    _build_chunks,
    _dump_native_features,
    _load_checkpoint_tensor,
    _parse_kv,
    _run,
)


def _dump_native_prep(
    prep_bin: Path,
    gguf: Path,
    audio: Path,
    n_threads: int,
    backend: str,
) -> tuple[np.ndarray, dict[str, str]]:
    with tempfile.NamedTemporaryFile(suffix=".f32", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        meta = _parse_kv(
            _run(
                [
                    str(prep_bin),
                    str(gguf),
                    str(audio),
                    "--threads",
                    str(n_threads),
                    "--backend",
                    backend,
                    "--out",
                    str(tmp_path),
                ]
            )
        )
        prep = np.fromfile(tmp_path, dtype=np.float32).reshape(
            int(meta["tokens"]),
            int(meta["hidden"]),
        )
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
    return prep, meta


def _expected_segments(total_tokens: int, chunk_output_len: int, window_multiplier: int = 8):
    window = chunk_output_len * window_multiplier
    segments = []
    for offset in range(0, total_tokens, window):
        segments.append((offset, min(window, total_tokens - offset)))
    return segments


def _audio_output_length(input_frames: int) -> int:
    feat_lengths = 0 if input_frames % 100 == 0 else ((input_frames % 100 - 1) // 2 + 1)
    tail = 0 if feat_lengths == 0 else ((((feat_lengths - 1) // 2 + 1 - 1) // 2) + 1)
    return tail + (input_frames // 100) * 13


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate native Qwen3-ASR audio prep")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("gguf", type=Path)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--cpp-bin", default=str(ROOT / "build" / "qwen-asr-audio-prep"))
    parser.add_argument("--features-bin", default=str(ROOT / "build" / "qwen-asr-features"))
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--native-backend", choices=("ggml", "sched"), default="ggml")
    parser.add_argument("--atol", type=float, default=5e-4)
    args = parser.parse_args()

    prep_bin = Path(args.cpp_bin)
    features_bin = Path(args.features_bin)
    if not prep_bin.is_file():
        raise SystemExit(f"C++ prep binary not found: {prep_bin}")
    if not features_bin.is_file():
        raise SystemExit(f"C++ feature binary not found: {features_bin}")

    features, feature_meta = _dump_native_features(features_bin, args.audio)
    native, prep_meta = _dump_native_prep(
        prep_bin,
        args.gguf,
        args.audio,
        args.threads,
        args.native_backend,
    )
    ref_chunks = _build_chunks(features, feature_meta)

    try:
        import torch
        import torch.nn.functional as F
    except ImportError as exc:
        raise SystemExit("torch is required for the PyTorch reference audio prep") from exc

    weights = {name: _load_checkpoint_tensor(args.checkpoint, tensor) for name, tensor in TENSORS.items()}
    with torch.no_grad():
        x = torch.from_numpy(ref_chunks)
        x = F.gelu(F.conv2d(x, weights["conv0_weight"], weights["conv0_bias"], stride=2, padding=1))
        x = F.gelu(F.conv2d(x, weights["conv1_weight"], weights["conv1_bias"], stride=2, padding=1))
        x = F.gelu(F.conv2d(x, weights["conv2_weight"], weights["conv2_bias"], stride=2, padding=1))
        bsz, channels, freq, frames = x.shape
        x = x.permute(0, 3, 1, 2).contiguous().view(bsz, frames, channels * freq)
        x = F.linear(x, weights["conv_out_weight"])

        hidden = x.shape[-1]
        half = hidden // 2
        log_timescale_increment = np.log(10000.0) / (half - 1)
        inv_timescales = torch.exp(-log_timescale_increment * torch.arange(half).float())
        scaled_time = torch.arange(frames).float()[:, None] * inv_timescales[None, :]
        positional = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)
        x = x + positional.unsqueeze(0)

        ref_rows = []
        for chunk in range(ref_chunks.shape[0]):
            start = chunk * int(feature_meta["chunk_window"])
            chunk_len = min(int(feature_meta["chunk_window"]), features.shape[1] - start)
            if chunk_len <= 0:
                continue
            valid = _audio_output_length(chunk_len)
            ref_rows.append(x[chunk, :valid, :])
        ref = torch.cat(ref_rows, dim=0).numpy()

    if native.shape != ref.shape:
        raise SystemExit(f"shape mismatch: native={native.shape} torch={ref.shape}")

    expected_segments = _expected_segments(
        native.shape[0],
        int(feature_meta["max_chunk_output_len"]),
    )
    native_segments = [
        (int(prep_meta[f"segment.{i}.offset"]), int(prep_meta[f"segment.{i}.length"]))
        for i in range(int(prep_meta["attention_segments"]))
    ]
    if native_segments != expected_segments:
        raise SystemExit(f"segment mismatch: native={native_segments} expected={expected_segments}")

    diff = np.abs(native - ref)
    max_abs = float(diff.max())
    print(f"shape={native.shape}")
    print(f"native_backend={prep_meta['backend']}")
    print(f"native_prep_ms={float(prep_meta['prep_ms']):.3f}")
    print(f"attention_segments={native_segments}")
    print(f"max_abs={max_abs:.8f}")
    print(f"mean_abs={float(diff.mean()):.8f}")
    print(f"native_mean={float(native.mean()):.8f}")
    print(f"torch_mean={float(ref.mean()):.8f}")
    if max_abs > args.atol:
        raise SystemExit(f"audio prep mismatch: max_abs {max_abs:.8f} > atol {args.atol}")
    print("status=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
