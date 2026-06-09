#!/usr/bin/env python3
"""Compare native C++ audio encoder layer 0 against a Torch eager reference."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import numpy as np

from check_audio_cnn import TENSORS as FRONTEND_TENSORS
from check_audio_conv0 import (
    ROOT,
    _build_chunks,
    _dump_native_features,
    _load_checkpoint_tensor,
    _parse_kv,
    _run,
)
from check_audio_prep import _audio_output_length


LAYER0_TENSORS = {
    "attn_norm_weight": "thinker.audio_tower.layers.0.self_attn_layer_norm.weight",
    "attn_norm_bias": "thinker.audio_tower.layers.0.self_attn_layer_norm.bias",
    "ffn_norm_weight": "thinker.audio_tower.layers.0.final_layer_norm.weight",
    "ffn_norm_bias": "thinker.audio_tower.layers.0.final_layer_norm.bias",
    "q_weight": "thinker.audio_tower.layers.0.self_attn.q_proj.weight",
    "q_bias": "thinker.audio_tower.layers.0.self_attn.q_proj.bias",
    "k_weight": "thinker.audio_tower.layers.0.self_attn.k_proj.weight",
    "k_bias": "thinker.audio_tower.layers.0.self_attn.k_proj.bias",
    "v_weight": "thinker.audio_tower.layers.0.self_attn.v_proj.weight",
    "v_bias": "thinker.audio_tower.layers.0.self_attn.v_proj.bias",
    "out_weight": "thinker.audio_tower.layers.0.self_attn.out_proj.weight",
    "out_bias": "thinker.audio_tower.layers.0.self_attn.out_proj.bias",
    "up_weight": "thinker.audio_tower.layers.0.fc1.weight",
    "up_bias": "thinker.audio_tower.layers.0.fc1.bias",
    "down_weight": "thinker.audio_tower.layers.0.fc2.weight",
    "down_bias": "thinker.audio_tower.layers.0.fc2.bias",
}


def _dump_native_layer(
    layer_bin: Path,
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
                    str(layer_bin),
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
        layer = np.fromfile(tmp_path, dtype=np.float32).reshape(
            int(meta["tokens"]),
            int(meta["hidden"]),
        )
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
    return layer, meta


def _make_prep_reference(features, feature_meta, weights):
    import torch
    import torch.nn.functional as F

    ref_chunks = _build_chunks(features, feature_meta)
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

        rows = []
        for chunk in range(ref_chunks.shape[0]):
            start = chunk * int(feature_meta["chunk_window"])
            chunk_len = min(int(feature_meta["chunk_window"]), features.shape[1] - start)
            if chunk_len > 0:
                rows.append(x[chunk, : _audio_output_length(chunk_len), :])
        return torch.cat(rows, dim=0)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate native Qwen3-ASR audio layer 0")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("gguf", type=Path)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--cpp-bin", default=str(ROOT / "build" / "qwen-asr-audio-layer"))
    parser.add_argument("--features-bin", default=str(ROOT / "build" / "qwen-asr-features"))
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--backend", choices=("ggml", "cpu"), default="ggml")
    parser.add_argument("--heads", type=int, default=14)
    parser.add_argument("--atol", type=float, default=3e-3)
    args = parser.parse_args()

    layer_bin = Path(args.cpp_bin)
    features_bin = Path(args.features_bin)
    if not layer_bin.is_file():
        raise SystemExit(f"C++ layer binary not found: {layer_bin}")
    if not features_bin.is_file():
        raise SystemExit(f"C++ feature binary not found: {features_bin}")

    features, feature_meta = _dump_native_features(features_bin, args.audio)
    native, layer_meta = _dump_native_layer(layer_bin, args.gguf, args.audio, args.threads, args.backend)

    try:
        import torch
        import torch.nn.functional as F
    except ImportError as exc:
        raise SystemExit("torch is required for the PyTorch reference audio layer") from exc

    frontend = {
        name: _load_checkpoint_tensor(args.checkpoint, tensor)
        for name, tensor in FRONTEND_TENSORS.items()
    }
    weights = {
        name: _load_checkpoint_tensor(args.checkpoint, tensor)
        for name, tensor in LAYER0_TENSORS.items()
    }

    hidden_states = _make_prep_reference(features, feature_meta, frontend)
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
        head_dim = hidden // args.heads
        q = q.reshape(seq_len, args.heads, head_dim).transpose(0, 1).unsqueeze(0)
        k = k.reshape(seq_len, args.heads, head_dim).transpose(0, 1).unsqueeze(0)
        v = v.reshape(seq_len, args.heads, head_dim).transpose(0, 1).unsqueeze(0)
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
        ref = (residual + x).numpy()

    if native.shape != ref.shape:
        raise SystemExit(f"shape mismatch: native={native.shape} torch={ref.shape}")

    diff = np.abs(native - ref)
    max_abs = float(diff.max())
    print(f"shape={native.shape}")
    print(f"native_backend={layer_meta['backend']}")
    print(f"native_layer_ms={float(layer_meta['layer_ms']):.3f}")
    print(f"max_abs={max_abs:.8f}")
    print(f"mean_abs={float(diff.mean()):.8f}")
    print(f"native_mean={float(native.mean()):.8f}")
    print(f"torch_mean={float(ref.mean()):.8f}")
    if max_abs > args.atol:
        raise SystemExit(f"audio layer0 mismatch: max_abs {max_abs:.8f} > atol {args.atol}")
    print("status=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
