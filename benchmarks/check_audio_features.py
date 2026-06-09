#!/usr/bin/env python3
"""Compare native C++ Whisper features against the HF Qwen3-ASR processor."""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _parse_kv(text: str) -> dict[str, str]:
    out = {}
    for line in text.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate native Qwen3-ASR audio features")
    parser.add_argument("audio")
    parser.add_argument("--cpp-bin", default=str(ROOT / "build" / "qwen-asr-features"))
    parser.add_argument("--model", default="Qwen/Qwen3-ASR-0.6B")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--atol", type=float, default=2e-4)
    args = parser.parse_args()

    cpp_bin = Path(args.cpp_bin)
    if not cpp_bin.is_file():
        raise SystemExit(f"C++ feature binary not found: {cpp_bin}")

    with tempfile.NamedTemporaryFile(suffix=".f32", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        proc = subprocess.run(
            [str(cpp_bin), args.audio, "--out", str(tmp_path)],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True,
        )
        meta = _parse_kv(proc.stdout)
        native = np.fromfile(tmp_path, dtype=np.float32).reshape(int(meta["mels"]), int(meta["frames"]))
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    from qwen_asr.inference.utils import normalize_audio_input
    from qwen_asr import Qwen3ASRModel

    model = Qwen3ASRModel.from_pretrained(args.model, local_files_only=args.local_files_only)
    wav = normalize_audio_input(args.audio)
    features = model.processor.feature_extractor(
        [wav],
        sampling_rate=16000,
        padding=True,
        truncation=False,
        return_attention_mask=True,
        return_tensors="np",
    )
    ref = features["input_features"][0].astype(np.float32)

    if native.shape != ref.shape:
        raise SystemExit(f"shape mismatch: native={native.shape} hf={ref.shape}")

    diff = np.abs(native - ref)
    print(f"shape={native.shape}")
    print(f"max_abs={float(diff.max()):.8f}")
    print(f"mean_abs={float(diff.mean()):.8f}")
    print(f"native_mean={float(native.mean()):.8f}")
    print(f"hf_mean={float(ref.mean()):.8f}")
    if float(diff.max()) > args.atol:
        raise SystemExit(f"feature mismatch: max_abs {float(diff.max()):.8f} > atol {args.atol}")
    print("status=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
