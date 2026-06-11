#!/usr/bin/env python3
"""Compare native decoder input embeddings against Torch prompt/audio merge."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import numpy as np

from check_audio_cnn import TENSORS as FRONTEND_TENSORS
from check_audio_conv0 import ROOT, _dump_native_features, _load_checkpoint_tensor, _parse_kv, _run
from check_audio_encoder import PROJECTOR_TENSORS, _audio_layer_reference, _layer_tensors
from check_audio_layer0 import _make_prep_reference


TEXT_EMBED_NAME = "thinker.model.embed_tokens.weight"
AUDIO_TOKEN_ID = 151676


def _prompt(audio_tokens: int, system: str, language: str) -> str:
    text = (
        "<|im_start|>system\n"
        + system
        + "<|im_end|>\n<|im_start|>user\n<|audio_start|>"
        + "<|audio_pad|>" * audio_tokens
        + "<|audio_end|><|im_end|>\n<|im_start|>assistant\n"
    )
    if language:
        text += f"language {language}<asr_text>"
    return text


def _hf_ids(checkpoint: Path, audio_tokens: int, system: str, language: str) -> list[int]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint,
        local_files_only=True,
        trust_remote_code=True,
        fix_mistral_regex=True,
    )
    return tokenizer(_prompt(audio_tokens, system, language), add_special_tokens=False).input_ids


def _dump_native_decoder_input(
    decoder_bin: Path,
    gguf: Path,
    audio: Path,
    n_threads: int,
    audio_backend: str,
    device: str,
    system: str,
    language: str,
) -> tuple[np.ndarray, dict[str, str], list[int]]:
    with tempfile.NamedTemporaryFile(suffix=".f32", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        cmd = [
            str(decoder_bin),
            str(gguf),
            str(audio),
            "--threads",
            str(n_threads),
            "--audio-backend",
            audio_backend,
            "--device",
            device,
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
        ids = [int(item) for item in meta["ids"].split(",") if item]
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
    return values, meta, ids


def _torch_audio_encoder(checkpoint: Path, features, feature_meta, layers: int, heads: int):
    import torch
    import torch.nn.functional as F

    frontend = {
        name: _load_checkpoint_tensor(checkpoint, tensor)
        for name, tensor in FRONTEND_TENSORS.items()
    }
    hidden_states = _make_prep_reference(features, feature_meta, frontend)

    for layer in range(layers):
        weights = {
            name: _load_checkpoint_tensor(checkpoint, tensor)
            for name, tensor in _layer_tensors(layer).items()
        }
        hidden_states = _audio_layer_reference(hidden_states, weights, heads)

    projector = {
        name: _load_checkpoint_tensor(checkpoint, tensor)
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
        return F.linear(x, projector["proj1_weight"], projector["proj1_bias"])


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate native Qwen3-ASR decoder input embeddings")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("gguf", type=Path)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--cpp-bin", default=str(ROOT / "build" / "qwen-asr-decoder-input"))
    parser.add_argument("--features-bin", default=str(ROOT / "build" / "qwen-asr-features"))
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--layers", type=int, default=18)
    parser.add_argument("--heads", type=int, default=14)
    parser.add_argument("--system", default="")
    parser.add_argument("--language", default="")
    parser.add_argument("--native-audio-backend", choices=("ggml", "sched"), default="ggml")
    parser.add_argument("--native-device", choices=("auto", "cpu", "gpu", "cuda"), default="auto")
    parser.add_argument("--atol", type=float, default=5e-3)
    args = parser.parse_args()

    decoder_bin = Path(args.cpp_bin)
    features_bin = Path(args.features_bin)
    if not decoder_bin.is_file():
        raise SystemExit(f"C++ decoder-input binary not found: {decoder_bin}")
    if not features_bin.is_file():
        raise SystemExit(f"C++ feature binary not found: {features_bin}")

    features, feature_meta = _dump_native_features(features_bin, args.audio)
    native, meta, native_ids = _dump_native_decoder_input(
        decoder_bin,
        args.gguf,
        args.audio,
        args.threads,
        args.native_audio_backend,
        args.native_device,
        args.system,
        args.language,
    )
    audio_tokens = int(meta["audio_tokens"])
    hf_ids = _hf_ids(args.checkpoint, audio_tokens, args.system, args.language)
    if native_ids != hf_ids:
        raise SystemExit(f"prompt id mismatch: native={native_ids} hf={hf_ids}")

    audio_features = _torch_audio_encoder(args.checkpoint, features, feature_meta, args.layers, args.heads)
    embeddings = _load_checkpoint_tensor(args.checkpoint, TEXT_EMBED_NAME)
    ids_np = np.asarray(hf_ids, dtype=np.int64)
    ref = embeddings[hf_ids].numpy()
    audio_positions = np.flatnonzero(ids_np == AUDIO_TOKEN_ID)
    if len(audio_positions) != audio_features.shape[0]:
        raise SystemExit(
            f"audio placeholder mismatch: prompt={len(audio_positions)} audio={audio_features.shape[0]}"
        )
    ref[audio_positions, :] = audio_features.numpy()

    if native.shape != ref.shape:
        raise SystemExit(f"shape mismatch: native={native.shape} torch={ref.shape}")
    diff = np.abs(native - ref)
    max_abs = float(diff.max())
    print(f"shape={native.shape}")
    print(f"audio_tokens={audio_tokens}")
    print(f"native_backend={meta['backend']}")
    print(f"native_audio_backend={meta['audio_backend']}")
    print(f"native_device={meta.get('device', args.native_device)}")
    print(f"native_decoder_input_ms={float(meta['decoder_input_ms']):.3f}")
    print(f"max_abs={max_abs:.8f}")
    print(f"mean_abs={float(diff.mean()):.8f}")
    print(f"native_mean={float(native.mean()):.8f}")
    print(f"torch_mean={float(ref.mean()):.8f}")
    if max_abs > args.atol:
        raise SystemExit(f"decoder input mismatch: max_abs {max_abs:.8f} > atol {args.atol}")
    print("status=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
