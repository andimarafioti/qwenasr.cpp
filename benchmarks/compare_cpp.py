#!/usr/bin/env python3
"""Compare the C++ qwen-asr bridge CLI against the Python backend."""

from __future__ import annotations

import argparse
import os
import site
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str], *, env: dict[str, str] | None = None) -> dict[str, str]:
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )
    data: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            data[key.strip()] = value.strip()
    data["_raw"] = proc.stdout
    return data


def _pythonpath(extra: str | None) -> str:
    parts = [str(ROOT)]
    if extra:
        parts.extend(item for item in extra.split(os.pathsep) if item)
    for item in site.getsitepackages():
        if item not in parts:
            parts.append(item)
    user_site = site.getusersitepackages()
    if user_site and user_site not in parts:
        parts.append(user_site)
    existing = os.environ.get("PYTHONPATH")
    if existing:
        parts.extend(item for item in existing.split(os.pathsep) if item)
    return os.pathsep.join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare C++ bridge and Python Qwen3-ASR speed")
    parser.add_argument("audio")
    parser.add_argument("--cpp-bin", default=str(ROOT / "build" / "qwen-asr"))
    parser.add_argument("--size", default="0.6B", choices=["0.6B", "1.7B", "small", "large"])
    parser.add_argument("--backend", default="torch", choices=["auto", "torch", "transformers", "vllm"])
    parser.add_argument("--dtype", default="bf16", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("--language", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--pythonpath", default=None, help="extra paths for the embedded Python runtime")
    args = parser.parse_args()

    cpp_bin = Path(args.cpp_bin)
    if not cpp_bin.is_file():
        raise SystemExit(f"C++ binary not found: {cpp_bin}. Build with `cmake -S . -B build && cmake --build build`.")

    dtype = [] if args.dtype == "auto" else ["--dtype", args.dtype]
    language = [] if args.language is None else ["--language", args.language]
    local = ["--local-files-only"] if args.local_files_only else []

    py_cmd = [
        sys.executable,
        "benchmarks/throughput.py",
        args.audio,
        "--size",
        args.size,
        "--backend",
        args.backend,
        "--warmup",
        str(args.warmup),
        "--runs",
        str(args.runs),
        *dtype,
        *language,
        *local,
    ]
    py_env = os.environ.copy()
    py_env["PYTHONPATH"] = _pythonpath(args.pythonpath)
    py = _run(py_cmd, env=py_env)

    embedded_path = _pythonpath(args.pythonpath)
    cpp_cmd = [
        str(cpp_bin),
        args.audio,
        "--size",
        args.size,
        "--backend",
        args.backend,
        "--python-path",
        embedded_path,
        "--warmup",
        str(args.warmup),
        "--runs",
        str(args.runs),
        *dtype,
        *language,
        *local,
    ]
    cpp_env = os.environ.copy()
    cpp_env["PYTHONPATH"] = embedded_path
    cpp = _run(cpp_cmd, env=cpp_env)

    py_best = float(py["best_sec"])
    cpp_best = float(cpp["best_sec"])
    print(f"audio={args.audio}")
    print(f"python_model={py.get('model', '')}")
    print(f"python_backend={py.get('backend', '')}")
    print(f"python_best_sec={py_best:.4f}")
    print(f"python_rtf={float(py.get('rtf', '0') or 0):.2f}")
    print(f"cpp_model={cpp.get('model', '')}")
    print(f"cpp_backend={cpp.get('backend', '')}")
    print(f"cpp_bridge={cpp.get('bridge', '')}")
    print(f"cpp_best_sec={cpp_best:.4f}")
    print(f"cpp_rtf={float(cpp.get('rtf', '0') or 0):.2f}")
    print(f"cpp_vs_python={cpp_best / py_best:.3f}x")
    print(f"python_text={py.get('text', '')!r}")
    print(f"cpp_text={cpp.get('text', '')!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
