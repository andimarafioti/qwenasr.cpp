#!/usr/bin/env python3
"""Qwen3-ASR Hugging Face checkpoint -> native tensor map / GGUF.

The native GGML runtime is still under construction. This converter already
locks down the weight naming contract that runtime will load. `--dry-run`
requires only `safetensors` and validates that every checkpoint tensor maps to a
qwenasr.cpp name. Writing GGUF additionally requires the `gguf` Python module.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

from safetensors import safe_open


ARCH = "qwen3-asr"


def _load_config(checkpoint: Path) -> dict:
    path = checkpoint / "config.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing config.json in {checkpoint}")
    return json.loads(path.read_text())


def _iter_safetensors(checkpoint: Path) -> list[Path]:
    files = sorted(checkpoint.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no safetensors files found in {checkpoint}")
    return files


def rename_tensor(name: str) -> str:
    if name == "thinker.model.embed_tokens.weight":
        return "text.token_embd.weight"
    if name == "thinker.model.norm.weight":
        return "text.output_norm.weight"
    if name == "thinker.lm_head.weight":
        return "text.output.weight"

    if name.startswith("thinker.model.layers."):
        return _rename_text_layer(name)
    if name.startswith("thinker.audio_tower.layers."):
        return _rename_audio_layer(name)
    if name.startswith("thinker.audio_tower."):
        return _rename_audio_global(name)

    raise ValueError(f"unhandled tensor name: {name}")


def _rename_text_layer(name: str) -> str:
    match = re.fullmatch(r"thinker\.model\.layers\.(\d+)\.(.+)", name)
    if not match:
        raise ValueError(f"bad text layer tensor name: {name}")
    layer, rest = match.groups()
    mapping = {
        "input_layernorm.weight": "attn_norm.weight",
        "post_attention_layernorm.weight": "ffn_norm.weight",
        "self_attn.q_proj.weight": "attn_q.weight",
        "self_attn.k_proj.weight": "attn_k.weight",
        "self_attn.v_proj.weight": "attn_v.weight",
        "self_attn.o_proj.weight": "attn_output.weight",
        "self_attn.q_norm.weight": "attn_q_norm.weight",
        "self_attn.k_norm.weight": "attn_k_norm.weight",
        "mlp.gate_proj.weight": "ffn_gate.weight",
        "mlp.up_proj.weight": "ffn_up.weight",
        "mlp.down_proj.weight": "ffn_down.weight",
    }
    if rest not in mapping:
        raise ValueError(f"unhandled text layer tensor: {name}")
    return f"text.blk.{layer}.{mapping[rest]}"


def _rename_audio_global(name: str) -> str:
    mapping = {
        "thinker.audio_tower.conv2d1.weight": "audio.conv.0.weight",
        "thinker.audio_tower.conv2d1.bias": "audio.conv.0.bias",
        "thinker.audio_tower.conv2d2.weight": "audio.conv.1.weight",
        "thinker.audio_tower.conv2d2.bias": "audio.conv.1.bias",
        "thinker.audio_tower.conv2d3.weight": "audio.conv.2.weight",
        "thinker.audio_tower.conv2d3.bias": "audio.conv.2.bias",
        "thinker.audio_tower.conv_out.weight": "audio.conv_out.weight",
        "thinker.audio_tower.ln_post.weight": "audio.post_norm.weight",
        "thinker.audio_tower.ln_post.bias": "audio.post_norm.bias",
        "thinker.audio_tower.proj1.weight": "audio.proj.0.weight",
        "thinker.audio_tower.proj1.bias": "audio.proj.0.bias",
        "thinker.audio_tower.proj2.weight": "audio.proj.1.weight",
        "thinker.audio_tower.proj2.bias": "audio.proj.1.bias",
    }
    if name not in mapping:
        raise ValueError(f"unhandled audio tensor: {name}")
    return mapping[name]


def _rename_audio_layer(name: str) -> str:
    match = re.fullmatch(r"thinker\.audio_tower\.layers\.(\d+)\.(.+)", name)
    if not match:
        raise ValueError(f"bad audio layer tensor name: {name}")
    layer, rest = match.groups()
    mapping = {
        "self_attn_layer_norm.weight": "attn_norm.weight",
        "self_attn_layer_norm.bias": "attn_norm.bias",
        "final_layer_norm.weight": "ffn_norm.weight",
        "final_layer_norm.bias": "ffn_norm.bias",
        "self_attn.q_proj.weight": "attn_q.weight",
        "self_attn.q_proj.bias": "attn_q.bias",
        "self_attn.k_proj.weight": "attn_k.weight",
        "self_attn.k_proj.bias": "attn_k.bias",
        "self_attn.v_proj.weight": "attn_v.weight",
        "self_attn.v_proj.bias": "attn_v.bias",
        "self_attn.out_proj.weight": "attn_output.weight",
        "self_attn.out_proj.bias": "attn_output.bias",
        "fc1.weight": "ffn_up.weight",
        "fc1.bias": "ffn_up.bias",
        "fc2.weight": "ffn_down.weight",
        "fc2.bias": "ffn_down.bias",
    }
    if rest not in mapping:
        raise ValueError(f"unhandled audio layer tensor: {name}")
    return f"audio.blk.{layer}.{mapping[rest]}"


def collect_tensor_map(files: Iterable[Path]) -> dict[str, str]:
    tensor_map: dict[str, str] = {}
    for file in files:
        with safe_open(file, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                mapped = rename_tensor(name)
                if mapped in tensor_map.values():
                    raise ValueError(f"duplicate mapped tensor name: {mapped}")
                tensor_map[name] = mapped
    return tensor_map


def _add_metadata(writer, cfg: dict) -> None:
    thinker = cfg.get("thinker_config", cfg)
    text = thinker["text_config"]
    audio = thinker["audio_config"]

    writer.add_string("general.name", cfg.get("_name_or_path", "Qwen3-ASR"))
    writer.add_string("qwen3-asr.arch", ARCH)
    writer.add_uint32("qwen3-asr.audio.sample_rate", 16000)
    writer.add_uint32("qwen3-asr.audio.num_mel_bins", audio["num_mel_bins"])
    writer.add_uint32("qwen3-asr.audio.d_model", audio["d_model"])
    writer.add_uint32("qwen3-asr.audio.encoder_layers", audio["encoder_layers"])
    writer.add_uint32("qwen3-asr.audio.encoder_attention_heads", audio["encoder_attention_heads"])
    writer.add_uint32("qwen3-asr.audio.encoder_ffn_dim", audio["encoder_ffn_dim"])
    writer.add_uint32("qwen3-asr.audio.downsample_hidden_size", audio["downsample_hidden_size"])
    writer.add_uint32("qwen3-asr.audio.output_dim", audio["output_dim"])
    writer.add_uint32("qwen3-asr.text.vocab_size", text["vocab_size"])
    writer.add_uint32("qwen3-asr.text.hidden_size", text["hidden_size"])
    writer.add_uint32("qwen3-asr.text.intermediate_size", text["intermediate_size"])
    writer.add_uint32("qwen3-asr.text.num_hidden_layers", text["num_hidden_layers"])
    writer.add_uint32("qwen3-asr.text.num_attention_heads", text["num_attention_heads"])
    writer.add_uint32("qwen3-asr.text.num_key_value_heads", text["num_key_value_heads"])
    writer.add_uint32("qwen3-asr.text.head_dim", text["head_dim"])
    writer.add_float32("qwen3-asr.text.rope_theta", float(text["rope_theta"]))
    writer.add_float32("qwen3-asr.text.rms_norm_eps", float(text["rms_norm_eps"]))
    writer.add_uint32("qwen3-asr.token.audio_token_id", thinker.get("audio_token_id", 151676))
    writer.add_uint32("qwen3-asr.token.audio_start_token_id", thinker.get("audio_start_token_id", 151669))
    writer.add_uint32("qwen3-asr.token.audio_end_token_id", thinker.get("audio_end_token_id", 151670))


def write_gguf(
    checkpoint: Path,
    out_path: Path,
    tensor_map: dict[str, str],
    cfg: dict,
    *,
    metadata_only: bool = False,
) -> None:
    try:
        import gguf
    except Exception as exc:  # pragma: no cover - depends on optional external package
        raise RuntimeError("writing GGUF requires the optional `gguf` Python module") from exc

    writer = gguf.GGUFWriter(str(out_path), ARCH)
    _add_metadata(writer, cfg)

    if not metadata_only:
        files = _iter_safetensors(checkpoint)
        for file in files:
            with safe_open(file, framework="pt", device="cpu") as handle:
                for name in handle.keys():
                    tensor = handle.get_tensor(name)
                    if str(tensor.dtype) == "torch.bfloat16":
                        tensor = tensor.float()
                    writer.add_tensor(tensor_map[name], tensor.numpy())

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    if not metadata_only:
        writer.write_tensors_to_file()
    writer.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert or validate Qwen3-ASR native tensor names")
    parser.add_argument("checkpoint", type=Path, help="HF snapshot directory")
    parser.add_argument("-o", "--out", type=Path, default=None, help="output GGUF path")
    parser.add_argument("--dry-run", action="store_true", help="validate mapping without writing GGUF")
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="write only GGUF metadata; useful for validating the native loader without large tensors",
    )
    parser.add_argument("--dump-map", type=Path, default=None, help="write JSON tensor map")
    args = parser.parse_args()

    cfg = _load_config(args.checkpoint)
    files = _iter_safetensors(args.checkpoint)
    tensor_map = collect_tensor_map(files)

    if args.dump_map:
        args.dump_map.write_text(json.dumps(tensor_map, indent=2, sort_keys=True) + "\n")

    thinker = cfg.get("thinker_config", cfg)
    text_layers = thinker["text_config"]["num_hidden_layers"]
    audio_layers = thinker["audio_config"]["encoder_layers"]
    print(f"checkpoint={args.checkpoint}")
    print(f"safetensors={len(files)}")
    print(f"tensors={len(tensor_map)}")
    print(f"text_layers={text_layers}")
    print(f"audio_layers={audio_layers}")

    if args.dry_run or args.out is None:
        print("status=dry-run-ok")
        return 0

    write_gguf(args.checkpoint, args.out, tensor_map, cfg, metadata_only=args.metadata_only)
    print(f"wrote={args.out}")
    if args.metadata_only:
        print("metadata_only=true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
