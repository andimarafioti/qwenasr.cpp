#!/usr/bin/env python3
"""Compare native GGUF BPE prompt tokens against the HF Qwen3-ASR tokenizer."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def _parse_kv(output: str) -> dict[str, str]:
    meta: dict[str, str] = {}
    for line in output.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            meta[key] = value
    return meta


def _native_ids(cpp_bin: Path, gguf: Path, audio_tokens: int, system: str, language: str) -> list[int]:
    cmd = [str(cpp_bin), str(gguf), "--audio-tokens", str(audio_tokens)]
    if system:
        cmd += ["--system", system]
    if language:
        cmd += ["--language", language]
    result = subprocess.run(cmd, check=True, text=True, capture_output=True)
    meta = _parse_kv(result.stdout)
    return [int(item) for item in meta["ids"].split(",") if item]


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
    )
    return tokenizer(_prompt(audio_tokens, system, language), add_special_tokens=False).input_ids


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate native ASR tokenizer prompt ids")
    parser.add_argument("checkpoint", type=Path, help="HF snapshot directory")
    parser.add_argument("gguf", type=Path, help="GGUF with tokenizer metadata")
    parser.add_argument("--cpp-bin", type=Path, default=Path("build/qwen-asr-tokenize"))
    parser.add_argument("--audio-tokens", type=int, default=143)
    parser.add_argument("--system", default="")
    parser.add_argument("--language", default="")
    args = parser.parse_args()

    native = _native_ids(args.cpp_bin, args.gguf, args.audio_tokens, args.system, args.language)
    ref = _hf_ids(args.checkpoint, args.audio_tokens, args.system, args.language)
    if native != ref:
        for i, (a, b) in enumerate(zip(native, ref)):
            if a != b:
                raise SystemExit(f"token mismatch at {i}: native={a} hf={b}")
        raise SystemExit(f"token length mismatch: native={len(native)} hf={len(ref)}")

    print(f"tokens={len(native)}")
    print(f"audio_tokens={args.audio_tokens}")
    print(f"language={args.language}")
    print("status=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
