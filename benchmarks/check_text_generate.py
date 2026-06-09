#!/usr/bin/env python3
"""Compare native greedy Qwen3-ASR text generation against a Torch reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from check_audio_conv0 import ROOT, _load_checkpoint_tensor, _parse_kv, _run
from check_decoder_input import TEXT_EMBED_NAME
from check_text_layer0 import _rms_norm, _text_layer_reference, _text_layer_tensors
from check_text_prefill import OUTPUT_NORM_NAME, OUTPUT_WEIGHT_NAME, _decoder_input_reference


def _dump_native_generate(
    text_bin: Path,
    gguf: Path,
    audio: Path,
    n_threads: int,
    native_backend: str,
    audio_backend: str,
    decode_backend: str,
    max_new_tokens: int,
    system: str,
    language: str,
) -> tuple[list[int], str, dict[str, str]]:
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
        "--generate",
        str(max_new_tokens),
    ]
    if decode_backend == "kv-cache":
        cmd.append("--kv-cache")
    if system:
        cmd += ["--system", system]
    if language:
        cmd += ["--language", language]
    meta = _parse_kv(_run(cmd))
    ids = [int(item) for item in meta["generated_ids"].split(",") if item]
    text = json.loads(meta["text_json"])
    return ids, text, meta


def _load_text_stack(checkpoint: Path, layers: int):
    layer_weights = [
        {
            name: _load_checkpoint_tensor(checkpoint, tensor)
            for name, tensor in _text_layer_tensors(layer).items()
        }
        for layer in range(layers)
    ]
    output_norm = _load_checkpoint_tensor(checkpoint, OUTPUT_NORM_NAME)
    output_weight = _load_checkpoint_tensor(checkpoint, OUTPUT_WEIGHT_NAME)
    embeddings = _load_checkpoint_tensor(checkpoint, TEXT_EMBED_NAME)
    return layer_weights, output_norm, output_weight, embeddings


def _prefill_logits_loaded(x, layer_weights, output_norm, output_weight, text_cfg: dict):
    import torch
    import torch.nn.functional as F

    heads = int(text_cfg["num_attention_heads"])
    kv_heads = int(text_cfg["num_key_value_heads"])
    head_dim = int(text_cfg["head_dim"])
    rope_theta = float(text_cfg["rope_theta"])
    eps = float(text_cfg["rms_norm_eps"])
    with torch.no_grad():
        y = x
        for weights in layer_weights:
            y = _text_layer_reference(
                y,
                weights,
                heads=heads,
                kv_heads=kv_heads,
                head_dim=head_dim,
                rope_theta=rope_theta,
                eps=eps,
            )
        y = _rms_norm(y[-1, :], output_norm, eps)
        return F.linear(y.unsqueeze(0), output_weight).squeeze(0)


def _torch_generate(checkpoint: Path, x, text_cfg: dict, max_new_tokens: int, stop_ids: set[int]) -> tuple[list[int], str]:
    import torch
    from transformers import AutoTokenizer

    layers = int(text_cfg["num_hidden_layers"])
    layer_weights, output_norm, output_weight, embeddings = _load_text_stack(checkpoint, layers)
    generated: list[int] = []
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = _prefill_logits_loaded(x, layer_weights, output_norm, output_weight, text_cfg)
            next_id = int(logits.argmax().item())
            generated.append(next_id)
            if next_id in stop_ids:
                break
            x = torch.cat([x, embeddings[next_id].view(1, -1)], dim=0)

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint,
        local_files_only=True,
        trust_remote_code=True,
        fix_mistral_regex=True,
    )
    return generated, tokenizer.decode(generated, skip_special_tokens=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate native Qwen3-ASR greedy text generation")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("gguf", type=Path)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--cpp-bin", default=str(ROOT / "build" / "qwen-asr-text-layer"))
    parser.add_argument("--features-bin", default=str(ROOT / "build" / "qwen-asr-features"))
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--native-backend", choices=("scalar", "sched"), default="scalar")
    parser.add_argument("--audio-backend", choices=("ggml", "sched"), default="sched")
    parser.add_argument("--native-decode-backend", choices=("recompute", "kv-cache"), default="recompute")
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--system", default="")
    parser.add_argument("--language", default="")
    args = parser.parse_args()

    text_bin = Path(args.cpp_bin)
    features_bin = Path(args.features_bin)
    if not text_bin.is_file():
        raise SystemExit(f"C++ text-layer binary not found: {text_bin}")
    if not features_bin.is_file():
        raise SystemExit(f"C++ feature binary not found: {features_bin}")
    if args.native_backend == "sched" and args.native_decode_backend != "kv-cache":
        raise SystemExit("--native-backend sched requires --native-decode-backend kv-cache")

    native_ids, native_text, meta = _dump_native_generate(
        text_bin,
        args.gguf,
        args.audio,
        args.threads,
        args.native_backend,
        args.audio_backend,
        args.native_decode_backend,
        args.max_new_tokens,
        args.system,
        args.language,
    )

    x, _ = _decoder_input_reference(args.checkpoint, features_bin, args.audio, args.system, args.language)
    cfg = json.loads((args.checkpoint / "config.json").read_text())
    text_cfg = cfg.get("thinker_config", cfg)["text_config"]
    torch_ids, torch_text = _torch_generate(
        args.checkpoint,
        x,
        text_cfg,
        args.max_new_tokens,
        {151643, 151645},
    )

    print(f"native_ids={','.join(str(item) for item in native_ids)}")
    print(f"torch_ids={','.join(str(item) for item in torch_ids)}")
    print(f"native_text_json={json.dumps(native_text)}")
    print(f"torch_text_json={json.dumps(torch_text)}")
    print(f"native_decoder_input_ms={float(meta['decoder_input_ms']):.3f}")
    print(f"native_text_init_ms={float(meta.get('text_init_ms', 0.0)):.3f}")
    print(f"native_generate_ms={float(meta['generate_ms']):.3f}")
    print(f"native_backend={meta['backend']}")
    print(f"native_decode_backend={meta['decode_backend']}")
    print(f"native_stopped={meta['stopped']}")
    if native_ids != torch_ids:
        raise SystemExit(f"generated id mismatch: native={native_ids} torch={torch_ids}")
    if native_text != torch_text:
        raise SystemExit(f"generated text mismatch: native={native_text!r} torch={torch_text!r}")
    print("status=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
