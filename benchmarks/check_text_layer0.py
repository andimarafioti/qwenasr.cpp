#!/usr/bin/env python3
"""Compare native Qwen3-ASR text decoder layer 0 against a Torch reference."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from check_audio_conv0 import ROOT, _dump_native_features, _load_checkpoint_tensor, _parse_kv, _run
from check_decoder_input import TEXT_EMBED_NAME, _hf_ids, _torch_audio_encoder


def _dump_native_text_layer(
    text_bin: Path,
    gguf: Path,
    audio: Path,
    n_threads: int,
    layer: int,
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
            "--layer",
            str(layer),
            "--backend",
            native_backend,
            "--audio-backend",
            audio_backend,
            "--out",
            str(tmp_path),
        ]
        if system:
            cmd += ["--system", system]
        if language:
            cmd += ["--language", language]
        meta = _parse_kv(_run(cmd))
        values = np.fromfile(tmp_path, dtype=np.float32).reshape(
            int(meta["tokens"]),
            int(meta["hidden"]),
        )
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
    return values, meta


def _text_layer_tensors(layer: int) -> dict[str, str]:
    prefix = f"thinker.model.layers.{layer}."
    return {
        "attn_norm_weight": prefix + "input_layernorm.weight",
        "ffn_norm_weight": prefix + "post_attention_layernorm.weight",
        "q_weight": prefix + "self_attn.q_proj.weight",
        "k_weight": prefix + "self_attn.k_proj.weight",
        "v_weight": prefix + "self_attn.v_proj.weight",
        "out_weight": prefix + "self_attn.o_proj.weight",
        "q_norm_weight": prefix + "self_attn.q_norm.weight",
        "k_norm_weight": prefix + "self_attn.k_norm.weight",
        "gate_weight": prefix + "mlp.gate_proj.weight",
        "up_weight": prefix + "mlp.up_proj.weight",
        "down_weight": prefix + "mlp.down_proj.weight",
    }


def _rms_norm(x, weight, eps: float):
    import torch

    variance = x.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
    return (x * torch.rsqrt(variance + eps).to(dtype=x.dtype)) * weight


def _rotate_half(x):
    import torch

    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _apply_rope(q, k, rope_theta: float):
    import torch

    head_dim = q.shape[-1]
    positions = torch.arange(q.shape[0], dtype=torch.float32, device=q.device)
    inv_freq = 1.0 / (
        rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=q.device) / head_dim)
    )
    freqs = torch.outer(positions, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().to(dtype=q.dtype).unsqueeze(1)
    sin = emb.sin().to(dtype=q.dtype).unsqueeze(1)
    return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)


def _text_layer_reference(x, weights, *, heads: int, kv_heads: int, head_dim: int, rope_theta: float, eps: float):
    import torch
    import torch.nn.functional as F

    with torch.no_grad():
        residual = x
        y = _rms_norm(x, weights["attn_norm_weight"], eps)
        q = F.linear(y, weights["q_weight"])
        k = F.linear(y, weights["k_weight"])
        v = F.linear(y, weights["v_weight"])
        q = q.view(q.shape[0], heads, head_dim)
        k = k.view(k.shape[0], kv_heads, head_dim)
        v = v.view(v.shape[0], kv_heads, head_dim)
        q = _rms_norm(q, weights["q_norm_weight"], eps)
        k = _rms_norm(k, weights["k_norm_weight"], eps)
        q, k = _apply_rope(q, k, rope_theta)

        repeat = heads // kv_heads
        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(repeat, dim=1)
        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        scores = torch.matmul(q, k.transpose(1, 2)) * (head_dim**-0.5)
        mask = torch.triu(torch.ones(scores.shape[-2:], dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask.to(scores.device).unsqueeze(0), float("-inf"))
        probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        y = torch.matmul(probs, v).transpose(0, 1).reshape(x.shape[0], heads * head_dim).contiguous()
        y = F.linear(y, weights["out_weight"])
        x = residual + y

        residual = x
        y = _rms_norm(x, weights["ffn_norm_weight"], eps)
        y = F.silu(F.linear(y, weights["gate_weight"])) * F.linear(y, weights["up_weight"])
        y = F.linear(y, weights["down_weight"])
        return residual + y


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate native Qwen3-ASR text decoder layer 0")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("gguf", type=Path)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--cpp-bin", default=str(ROOT / "build" / "qwen-asr-text-layer"))
    parser.add_argument("--features-bin", default=str(ROOT / "build" / "qwen-asr-features"))
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--native-backend", choices=("scalar", "ggml", "sched"), default="scalar")
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

    native, meta = _dump_native_text_layer(
        text_bin,
        args.gguf,
        args.audio,
        args.threads,
        args.layer,
        args.native_backend,
        args.audio_backend,
        args.system,
        args.language,
    )

    features, feature_meta = _dump_native_features(features_bin, args.audio)
    audio_features = _torch_audio_encoder(args.checkpoint, features, feature_meta, 18, 14)
    hf_ids = _hf_ids(args.checkpoint, audio_features.shape[0], args.system, args.language)
    embeddings = _load_checkpoint_tensor(args.checkpoint, TEXT_EMBED_NAME)
    ref = embeddings[hf_ids].clone()
    import torch

    ids_tensor = torch.tensor(hf_ids, dtype=torch.long)
    audio_positions = torch.nonzero(ids_tensor == 151676, as_tuple=False).flatten()
    ref[audio_positions, :] = audio_features

    cfg = json.loads((args.checkpoint / "config.json").read_text())
    text_cfg = cfg.get("thinker_config", cfg)["text_config"]
    weights = {
        name: _load_checkpoint_tensor(args.checkpoint, tensor)
        for name, tensor in _text_layer_tensors(args.layer).items()
    }
    ref = _text_layer_reference(
        ref,
        weights,
        heads=int(text_cfg["num_attention_heads"]),
        kv_heads=int(text_cfg["num_key_value_heads"]),
        head_dim=int(text_cfg["head_dim"]),
        rope_theta=float(text_cfg["rope_theta"]),
        eps=float(text_cfg["rms_norm_eps"]),
    ).numpy()

    if native.shape != ref.shape:
        raise SystemExit(f"shape mismatch: native={native.shape} torch={ref.shape}")
    diff = np.abs(native - ref)
    max_abs = float(diff.max())
    print(f"shape={native.shape}")
    print(f"native_backend={meta['backend']}")
    print(f"native_audio_backend={meta['audio_backend']}")
    print(f"native_text_init_ms={float(meta.get('text_init_ms', 0.0)):.3f}")
    print(f"native_decoder_input_ms={float(meta['decoder_input_ms']):.3f}")
    print(f"native_text_layer_ms={float(meta['text_layer_ms']):.3f}")
    print(f"max_abs={max_abs:.8f}")
    print(f"mean_abs={float(diff.mean()):.8f}")
    print(f"native_mean={float(native.mean()):.8f}")
    print(f"torch_mean={float(ref.mean()):.8f}")
    if max_abs > args.atol:
        raise SystemExit(f"text layer mismatch: max_abs {max_abs:.8f} > atol {args.atol}")
    print("status=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
