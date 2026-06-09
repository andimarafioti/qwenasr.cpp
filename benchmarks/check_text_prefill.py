#!/usr/bin/env python3
"""Compare native full Qwen3-ASR text prefill logits against a Torch reference."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from check_audio_conv0 import ROOT, _dump_native_features, _load_checkpoint_tensor, _parse_kv, _run
from check_decoder_input import AUDIO_TOKEN_ID, TEXT_EMBED_NAME, _hf_ids, _torch_audio_encoder
from check_text_layer0 import _rms_norm, _text_layer_reference, _text_layer_tensors


OUTPUT_NORM_NAME = "thinker.model.norm.weight"
OUTPUT_WEIGHT_NAME = "thinker.lm_head.weight"


def _dump_native_prefill(
    text_bin: Path,
    gguf: Path,
    audio: Path,
    n_threads: int,
    native_backend: str,
    audio_backend: str,
    system: str,
    language: str,
) -> tuple[np.ndarray, dict[str, str]]:
    with tempfile.NamedTemporaryFile(suffix=".f32", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        cmd = [
            str(text_bin),
            str(gguf),
            str(audio),
            "--threads",
            str(n_threads),
            "--backend",
            native_backend,
            "--audio-backend",
            audio_backend,
            "--prefill",
            "--out",
            str(tmp_path),
        ]
        if system:
            cmd += ["--system", system]
        if language:
            cmd += ["--language", language]
        meta = _parse_kv(_run(cmd))
        values = np.fromfile(tmp_path, dtype=np.float32)
        if values.shape != (int(meta["vocab"]),):
            raise SystemExit(f"logit shape mismatch: file={values.shape} meta_vocab={meta['vocab']}")
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
    return values, meta


def _decoder_input_reference(checkpoint: Path, features_bin: Path, audio: Path, system: str, language: str):
    import torch

    features, feature_meta = _dump_native_features(features_bin, audio)
    audio_features = _torch_audio_encoder(checkpoint, features, feature_meta, 18, 14)
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
    return x, hf_ids


def _text_prefill_reference(checkpoint: Path, x, text_cfg: dict):
    import torch
    import torch.nn.functional as F

    layers = int(text_cfg["num_hidden_layers"])
    heads = int(text_cfg["num_attention_heads"])
    kv_heads = int(text_cfg["num_key_value_heads"])
    head_dim = int(text_cfg["head_dim"])
    rope_theta = float(text_cfg["rope_theta"])
    eps = float(text_cfg["rms_norm_eps"])
    with torch.no_grad():
        for layer in range(layers):
            weights = {
                name: _load_checkpoint_tensor(checkpoint, tensor)
                for name, tensor in _text_layer_tensors(layer).items()
            }
            x = _text_layer_reference(
                x,
                weights,
                heads=heads,
                kv_heads=kv_heads,
                head_dim=head_dim,
                rope_theta=rope_theta,
                eps=eps,
            )
        x = _rms_norm(x[-1, :], _load_checkpoint_tensor(checkpoint, OUTPUT_NORM_NAME), eps)
        return F.linear(x.unsqueeze(0), _load_checkpoint_tensor(checkpoint, OUTPUT_WEIGHT_NAME)).squeeze(0)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate native Qwen3-ASR full text prefill logits")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("gguf", type=Path)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--cpp-bin", default=str(ROOT / "build" / "qwen-asr-text-layer"))
    parser.add_argument("--features-bin", default=str(ROOT / "build" / "qwen-asr-features"))
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--native-backend", choices=("scalar", "sched"), default="scalar")
    parser.add_argument("--audio-backend", choices=("ggml", "sched"), default="sched")
    parser.add_argument("--system", default="")
    parser.add_argument("--language", default="")
    parser.add_argument("--atol", type=float, default=2e-3)
    args = parser.parse_args()

    text_bin = Path(args.cpp_bin)
    features_bin = Path(args.features_bin)
    if not text_bin.is_file():
        raise SystemExit(f"C++ text-layer binary not found: {text_bin}")
    if not features_bin.is_file():
        raise SystemExit(f"C++ feature binary not found: {features_bin}")

    native, meta = _dump_native_prefill(
        text_bin,
        args.gguf,
        args.audio,
        args.threads,
        args.native_backend,
        args.audio_backend,
        args.system,
        args.language,
    )

    x, hf_ids = _decoder_input_reference(args.checkpoint, features_bin, args.audio, args.system, args.language)
    cfg = json.loads((args.checkpoint / "config.json").read_text())
    text_cfg = cfg.get("thinker_config", cfg)["text_config"]
    ref = _text_prefill_reference(args.checkpoint, x, text_cfg).numpy()

    if native.shape != ref.shape:
        raise SystemExit(f"shape mismatch: native={native.shape} torch={ref.shape}")
    diff = np.abs(native - ref)
    max_abs = float(diff.max())
    native_top = int(native.argmax())
    torch_top = int(ref.argmax())
    print(f"shape={native.shape}")
    print(f"tokens={len(hf_ids)}")
    print(f"layers={int(text_cfg['num_hidden_layers'])}")
    print(f"native_backend={meta['backend']}")
    print(f"native_audio_backend={meta['audio_backend']}")
    print(f"native_text_init_ms={float(meta.get('text_init_ms', 0.0)):.3f}")
    print(f"native_decoder_input_ms={float(meta['decoder_input_ms']):.3f}")
    print(f"native_prefill_ms={float(meta['prefill_ms']):.3f}")
    print(f"native_top_id={native_top}")
    print(f"torch_top_id={torch_top}")
    print(f"native_top_logit={float(native[native_top]):.8f}")
    print(f"torch_top_logit={float(ref[torch_top]):.8f}")
    print(f"max_abs={max_abs:.8f}")
    print(f"mean_abs={float(diff.mean()):.8f}")
    print(f"native_mean={float(native.mean()):.8f}")
    print(f"torch_mean={float(ref.mean()):.8f}")
    if native_top != torch_top:
        raise SystemExit(f"top-token mismatch: native={native_top} torch={torch_top}")
    if max_abs > args.atol:
        raise SystemExit(f"text prefill mismatch: max_abs {max_abs:.8f} > atol {args.atol}")
    print("status=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
