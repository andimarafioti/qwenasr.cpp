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
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    from safetensors import safe_open
except ModuleNotFoundError:  # pragma: no cover - exercised only without optional deps
    safe_open = None


ARCH = "qwen3-asr"


@dataclass(frozen=True)
class BpeMetadata:
    tokens: list[str]
    token_types: list[int]
    merges: list[str]
    token_ids: dict[str, int]


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


def load_bpe_vocab(checkpoint: Path, vocab_size: int) -> BpeMetadata:
    vocab_path = checkpoint / "vocab.json"
    merges_path = checkpoint / "merges.txt"
    tokenizer_config_path = checkpoint / "tokenizer_config.json"
    if not vocab_path.is_file():
        raise FileNotFoundError(f"missing vocab.json in {checkpoint}")
    if not merges_path.is_file():
        raise FileNotFoundError(f"missing merges.txt in {checkpoint}")
    if not tokenizer_config_path.is_file():
        raise FileNotFoundError(f"missing tokenizer_config.json in {checkpoint}")

    vocab = json.loads(vocab_path.read_text())
    tok_cfg = json.loads(tokenizer_config_path.read_text())
    added = tok_cfg.get("added_tokens_decoder", {})

    max_id = max(max(vocab.values()), vocab_size - 1)
    for sid in added.keys():
        max_id = max(max_id, int(sid))

    tokens: list[str | None] = [None] * (max_id + 1)
    token_types = [1] * (max_id + 1)  # 1 = normal
    token_ids: dict[str, int] = {}

    for token, token_id in vocab.items():
        tokens[token_id] = token
        token_ids[token] = token_id

    for sid_str, info in added.items():
        token_id = int(sid_str)
        token = info["content"]
        tokens[token_id] = token
        token_ids[token] = token_id
        token_types[token_id] = 4  # 4 = user-defined / special

    for i, token in enumerate(tokens):
        if token is None:
            tokens[i] = f"<|unused-{i}|>"
            token_types[i] = 5  # 5 = unused

    merges_lines = merges_path.read_text().splitlines()
    merges = [line for line in merges_lines if line and not line.startswith("#")]

    return BpeMetadata(tokens=[str(token) for token in tokens], token_types=token_types, merges=merges, token_ids=token_ids)


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


def _audio_conv_freq_bins(mel_bins: int) -> int:
    bins = mel_bins
    bins = (bins + 1) // 2
    bins = (bins + 1) // 2
    bins = (bins + 1) // 2
    return bins


def expected_hf_shapes(cfg: dict) -> dict[str, tuple[int, ...]]:
    thinker = cfg.get("thinker_config", cfg)
    text = thinker["text_config"]
    audio = thinker["audio_config"]

    text_hidden = text["hidden_size"]
    text_q_dim = text["num_attention_heads"] * text["head_dim"]
    text_kv_dim = text["num_key_value_heads"] * text["head_dim"]
    text_intermediate = text["intermediate_size"]
    text_vocab = text["vocab_size"]
    audio_hidden = audio["d_model"]
    audio_downsample = audio["downsample_hidden_size"]
    audio_ffn = audio["encoder_ffn_dim"]
    audio_output = audio["output_dim"]
    audio_conv_out = audio_downsample * _audio_conv_freq_bins(audio["num_mel_bins"])

    shapes: dict[str, tuple[int, ...]] = {
        "text.token_embd.weight": (text_vocab, text_hidden),
        "text.output_norm.weight": (text_hidden,),
        "text.output.weight": (text_vocab, text_hidden),
        "audio.conv.0.weight": (audio_downsample, 1, 3, 3),
        "audio.conv.0.bias": (audio_downsample,),
        "audio.conv.1.weight": (audio_downsample, audio_downsample, 3, 3),
        "audio.conv.1.bias": (audio_downsample,),
        "audio.conv.2.weight": (audio_downsample, audio_downsample, 3, 3),
        "audio.conv.2.bias": (audio_downsample,),
        "audio.conv_out.weight": (audio_hidden, audio_conv_out),
        "audio.post_norm.weight": (audio_hidden,),
        "audio.post_norm.bias": (audio_hidden,),
        "audio.proj.0.weight": (audio_hidden, audio_hidden),
        "audio.proj.0.bias": (audio_hidden,),
        "audio.proj.1.weight": (audio_output, audio_hidden),
        "audio.proj.1.bias": (audio_output,),
    }

    for layer in range(text["num_hidden_layers"]):
        prefix = f"text.blk.{layer}."
        shapes[prefix + "attn_norm.weight"] = (text_hidden,)
        shapes[prefix + "ffn_norm.weight"] = (text_hidden,)
        shapes[prefix + "attn_q.weight"] = (text_q_dim, text_hidden)
        shapes[prefix + "attn_k.weight"] = (text_kv_dim, text_hidden)
        shapes[prefix + "attn_v.weight"] = (text_kv_dim, text_hidden)
        shapes[prefix + "attn_output.weight"] = (text_hidden, text_q_dim)
        shapes[prefix + "attn_q_norm.weight"] = (text["head_dim"],)
        shapes[prefix + "attn_k_norm.weight"] = (text["head_dim"],)
        shapes[prefix + "ffn_gate.weight"] = (text_intermediate, text_hidden)
        shapes[prefix + "ffn_up.weight"] = (text_intermediate, text_hidden)
        shapes[prefix + "ffn_down.weight"] = (text_hidden, text_intermediate)

    for layer in range(audio["encoder_layers"]):
        prefix = f"audio.blk.{layer}."
        shapes[prefix + "attn_norm.weight"] = (audio_hidden,)
        shapes[prefix + "attn_norm.bias"] = (audio_hidden,)
        shapes[prefix + "ffn_norm.weight"] = (audio_hidden,)
        shapes[prefix + "ffn_norm.bias"] = (audio_hidden,)
        shapes[prefix + "attn_q.weight"] = (audio_hidden, audio_hidden)
        shapes[prefix + "attn_q.bias"] = (audio_hidden,)
        shapes[prefix + "attn_k.weight"] = (audio_hidden, audio_hidden)
        shapes[prefix + "attn_k.bias"] = (audio_hidden,)
        shapes[prefix + "attn_v.weight"] = (audio_hidden, audio_hidden)
        shapes[prefix + "attn_v.bias"] = (audio_hidden,)
        shapes[prefix + "attn_output.weight"] = (audio_hidden, audio_hidden)
        shapes[prefix + "attn_output.bias"] = (audio_hidden,)
        shapes[prefix + "ffn_up.weight"] = (audio_ffn, audio_hidden)
        shapes[prefix + "ffn_up.bias"] = (audio_ffn,)
        shapes[prefix + "ffn_down.weight"] = (audio_hidden, audio_ffn)
        shapes[prefix + "ffn_down.bias"] = (audio_hidden,)

    return shapes


def _tensor_shape(handle, name: str) -> tuple[int, ...]:
    try:
        return tuple(handle.get_slice(name).get_shape())
    except AttributeError:
        return tuple(handle.get_tensor(name).shape)


def collect_tensor_map(
    files: Iterable[Path],
    expected_shapes: dict[str, tuple[int, ...]] | None = None,
) -> dict[str, str]:
    if safe_open is None:
        raise RuntimeError("reading safetensors requires the `safetensors` Python package")

    tensor_map: dict[str, str] = {}
    seen_native: set[str] = set()
    for file in files:
        with safe_open(file, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                mapped = rename_tensor(name)
                if mapped in tensor_map.values():
                    raise ValueError(f"duplicate mapped tensor name: {mapped}")
                if expected_shapes is not None:
                    expected = expected_shapes.get(mapped)
                    if expected is None:
                        raise ValueError(f"unexpected mapped tensor name: {mapped}")
                    actual = _tensor_shape(handle, name)
                    if actual != expected:
                        raise ValueError(
                            f"shape mismatch for {name} -> {mapped}: expected {expected}, got {actual}"
                        )
                    seen_native.add(mapped)
                tensor_map[name] = mapped
    if expected_shapes is not None:
        missing = sorted(set(expected_shapes) - seen_native)
        if missing:
            preview = ", ".join(missing[:8])
            suffix = "" if len(missing) <= 8 else f", ... ({len(missing)} total)"
            raise ValueError(f"missing expected tensors: {preview}{suffix}")
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


def _add_tokenizer_metadata(writer, tokenizer: BpeMetadata) -> None:
    endoftext_id = tokenizer.token_ids.get("<|endoftext|>", 151643)
    im_start_id = tokenizer.token_ids.get("<|im_start|>", 151644)
    im_end_id = tokenizer.token_ids.get("<|im_end|>", 151645)
    asr_text_id = tokenizer.token_ids.get("<asr_text>", 151704)

    writer.add_string("tokenizer.ggml.model", "gpt2")
    writer.add_array("tokenizer.ggml.tokens", tokenizer.tokens)
    writer.add_array("tokenizer.ggml.token_type", tokenizer.token_types)
    writer.add_array("tokenizer.ggml.merges", tokenizer.merges)
    writer.add_uint32("tokenizer.ggml.eos_token_id", im_end_id)

    writer.add_uint32("qwen3-asr.token.endoftext_token_id", endoftext_id)
    writer.add_uint32("qwen3-asr.token.im_start_token_id", im_start_id)
    writer.add_uint32("qwen3-asr.token.im_end_token_id", im_end_id)
    writer.add_uint32("qwen3-asr.token.asr_text_token_id", asr_text_id)


def write_gguf(
    checkpoint: Path,
    out_path: Path,
    tensor_map: dict[str, str],
    cfg: dict,
    tokenizer: BpeMetadata,
    *,
    metadata_only: bool = False,
) -> None:
    try:
        import gguf
    except Exception as exc:  # pragma: no cover - depends on optional external package
        raise RuntimeError("writing GGUF requires the optional `gguf` Python module") from exc

    writer = gguf.GGUFWriter(str(out_path), ARCH)
    _add_metadata(writer, cfg)
    _add_tokenizer_metadata(writer, tokenizer)

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
    thinker = cfg.get("thinker_config", cfg)
    tokenizer = load_bpe_vocab(args.checkpoint, thinker["text_config"]["vocab_size"])
    shape_specs = expected_hf_shapes(cfg)
    tensor_map = collect_tensor_map(files, expected_shapes=shape_specs)

    if args.dump_map:
        args.dump_map.write_text(json.dumps(tensor_map, indent=2, sort_keys=True) + "\n")

    text_layers = thinker["text_config"]["num_hidden_layers"]
    audio_layers = thinker["audio_config"]["encoder_layers"]
    print(f"checkpoint={args.checkpoint}")
    print(f"safetensors={len(files)}")
    print(f"tensors={len(tensor_map)}")
    print(f"shape_checks={len(shape_specs)}")
    print(f"tokenizer_tokens={len(tokenizer.tokens)}")
    print(f"tokenizer_merges={len(tokenizer.merges)}")
    print(f"text_layers={text_layers}")
    print(f"audio_layers={audio_layers}")

    if args.dry_run or args.out is None:
        print("status=dry-run-ok")
        return 0

    write_gguf(args.checkpoint, args.out, tensor_map, cfg, tokenizer, metadata_only=args.metadata_only)
    print(f"wrote={args.out}")
    if args.metadata_only:
        print("metadata_only=true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
