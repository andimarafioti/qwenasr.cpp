#!/usr/bin/env python3
"""Compare native full audio encoder/projector against a Torch eager reference."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import numpy as np

from check_audio_cnn import TENSORS as FRONTEND_TENSORS
from check_audio_conv0 import ROOT, _dump_native_features, _load_checkpoint_tensor, _parse_kv, _run
from check_audio_layer0 import LAYER0_TENSORS, _make_prep_reference


PROJECTOR_TENSORS = {
    "post_norm_weight": "thinker.audio_tower.ln_post.weight",
    "post_norm_bias": "thinker.audio_tower.ln_post.bias",
    "proj0_weight": "thinker.audio_tower.proj1.weight",
    "proj0_bias": "thinker.audio_tower.proj1.bias",
    "proj1_weight": "thinker.audio_tower.proj2.weight",
    "proj1_bias": "thinker.audio_tower.proj2.bias",
}


def _layer_tensors(layer: int) -> dict[str, str]:
    return {
        name: tensor.replace(".layers.0.", f".layers.{layer}.")
        for name, tensor in LAYER0_TENSORS.items()
    }


def _dump_native_encoder(
    encoder_bin: Path,
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
                    str(encoder_bin),
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
        encoder = np.fromfile(tmp_path, dtype=np.float32).reshape(
            int(meta["tokens"]),
            int(meta["hidden"]),
        )
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
    return encoder, meta


def _audio_layer_reference(hidden_states, weights, heads: int):
    import torch
    import torch.nn.functional as F

    with torch.no_grad():
        residual = hidden_states
        x = F.layer_norm(
            hidden_states,
            (hidden_states.shape[-1],),
            weights["attn_norm_weight"],
            weights["attn_norm_bias"],
            eps=1e-5,
        )
        q = F.linear(x, weights["q_weight"], weights["q_bias"])
        k = F.linear(x, weights["k_weight"], weights["k_bias"])
        v = F.linear(x, weights["v_weight"], weights["v_bias"])
        seq_len, hidden = q.shape
        head_dim = hidden // heads
        q = q.reshape(seq_len, heads, head_dim).transpose(0, 1).unsqueeze(0)
        k = k.reshape(seq_len, heads, head_dim).transpose(0, 1).unsqueeze(0)
        v = v.reshape(seq_len, heads, head_dim).transpose(0, 1).unsqueeze(0)
        attn = torch.matmul(q, k.transpose(2, 3)) * (head_dim**-0.5)
        attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
        x = torch.matmul(attn, v).transpose(1, 2).reshape(seq_len, hidden).contiguous()
        x = F.linear(x, weights["out_weight"], weights["out_bias"])
        hidden_states = residual + x

        residual = hidden_states
        x = F.layer_norm(
            hidden_states,
            (hidden_states.shape[-1],),
            weights["ffn_norm_weight"],
            weights["ffn_norm_bias"],
            eps=1e-5,
        )
        x = F.linear(x, weights["up_weight"], weights["up_bias"])
        x = F.gelu(x)
        x = F.linear(x, weights["down_weight"], weights["down_bias"])
        return residual + x


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate native Qwen3-ASR full audio encoder")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("gguf", type=Path)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--cpp-bin", default=str(ROOT / "build" / "qwen-asr-audio-encoder"))
    parser.add_argument("--features-bin", default=str(ROOT / "build" / "qwen-asr-features"))
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--layers", type=int, default=18)
    parser.add_argument("--heads", type=int, default=14)
    parser.add_argument("--native-backend", choices=("ggml", "sched"), default="ggml")
    parser.add_argument("--atol", type=float, default=5e-3)
    args = parser.parse_args()

    encoder_bin = Path(args.cpp_bin)
    features_bin = Path(args.features_bin)
    if not encoder_bin.is_file():
        raise SystemExit(f"C++ encoder binary not found: {encoder_bin}")
    if not features_bin.is_file():
        raise SystemExit(f"C++ feature binary not found: {features_bin}")

    features, feature_meta = _dump_native_features(features_bin, args.audio)
    native, encoder_meta = _dump_native_encoder(
        encoder_bin,
        args.gguf,
        args.audio,
        args.threads,
        args.native_backend,
    )

    try:
        import torch
        import torch.nn.functional as F
    except ImportError as exc:
        raise SystemExit("torch is required for the PyTorch reference audio encoder") from exc

    frontend = {
        name: _load_checkpoint_tensor(args.checkpoint, tensor)
        for name, tensor in FRONTEND_TENSORS.items()
    }
    hidden_states = _make_prep_reference(features, feature_meta, frontend)

    for layer in range(args.layers):
        weights = {
            name: _load_checkpoint_tensor(args.checkpoint, tensor)
            for name, tensor in _layer_tensors(layer).items()
        }
        hidden_states = _audio_layer_reference(hidden_states, weights, args.heads)

    projector = {
        name: _load_checkpoint_tensor(args.checkpoint, tensor)
        for name, tensor in PROJECTOR_TENSORS.items()
    }
    with torch.no_grad():
        x = F.layer_norm(
            hidden_states,
            (hidden_states.shape[-1],),
            projector["post_norm_weight"],
            projector["post_norm_bias"],
            eps=1e-5,
        )
        x = F.linear(x, projector["proj0_weight"], projector["proj0_bias"])
        x = F.gelu(x)
        ref = F.linear(x, projector["proj1_weight"], projector["proj1_bias"]).numpy()

    if native.shape != ref.shape:
        raise SystemExit(f"shape mismatch: native={native.shape} torch={ref.shape}")

    diff = np.abs(native - ref)
    max_abs = float(diff.max())
    print(f"shape={native.shape}")
    print(f"native_backend={encoder_meta['backend']}")
    print(f"native_encoder_ms={float(encoder_meta['encoder_ms']):.3f}")
    print(f"max_abs={max_abs:.8f}")
    print(f"mean_abs={float(diff.mean()):.8f}")
    print(f"native_mean={float(native.mean()):.8f}")
    print(f"torch_mean={float(ref.mean()):.8f}")
    if max_abs > args.atol:
        raise SystemExit(f"audio encoder mismatch: max_abs {max_abs:.8f} > atol {args.atol}")
    print("status=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
