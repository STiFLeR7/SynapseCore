#!/usr/bin/env python3
"""
bench.py

Run simple inference benchmarks for listed models and save results.

Outputs:
  - data/outputs/<model>_results.jsonl
  - data/outputs/summary_<timestamp>.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import yaml

# ---------------------------------------------------------------------
# Type-checking stubs to satisfy MyPy/Pylance
# ---------------------------------------------------------------------
if TYPE_CHECKING:
    class _Llama:
        def __init__(self, model: str, n_ctx: int) -> None: ...
        def create(self, **kwargs: Any) -> Dict[str, Any]: ...
    # import pynvml as _pynvml  # type: ignore
    # import torch as _torch  # type: ignore
    # from tqdm import tqdm  # type: ignore

# ---------------------------------------------------------------------
# Runtime imports with graceful degradation
# ---------------------------------------------------------------------
LLaMA_AVAILABLE: bool = False
NVML_AVAILABLE: bool = False
TORCH_AVAILABLE: bool = False

try:
    from llama_cpp import Llama  # type: ignore
    LLaMA_AVAILABLE = True
except Exception:
    # llama-cpp-python not installed; bench will run in dry-run mode.
    pass

# pynvml often ships as deprecated; warn but try to use it
try:
    import warnings
    import pynvml  # type: ignore
    # suppress the specific FutureWarning emitted by some pynvml builds
    warnings.filterwarnings("ignore", category=FutureWarning)
    try:
        pynvml.nvmlInit()
        NVML_AVAILABLE = True
    except Exception:
        NVML_AVAILABLE = False
except Exception:
    NVML_AVAILABLE = False

try:
    import torch  # type: ignore
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

# Ensure tqdm symbol always exists (import or fallback).
try:
    from tqdm import tqdm  # type: ignore
except Exception:
    # Minimal safe tqdm fallback for environments without tqdm.
    def tqdm(iterable, **_kwargs):  # type: ignore
        for i in iterable:
            yield i

# ---------------------------------------------------------------------
# Paths and defaults
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT / "configs" / "base_config.yaml"
EVAL_CONFIG = ROOT / "configs" / "eval_config.yaml"
LLAMA_CONFIG = ROOT / "configs" / "llama_config.yaml"
OUTPUT_DIR = ROOT / "data" / "outputs"
MODELS_DIR = ROOT / "models"
DEFAULT_PROMPT_SUITE = ROOT / "data" / "prompts" / "instruction_suite.jsonl"

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def load_yaml(p: Path) -> Dict[str, Any]:
    if not p.exists():
        raise FileNotFoundError(f"Config missing: {p}")
    return yaml.safe_load(p.read_text())


def get_gpu_mem_mb() -> Optional[int]:
    """Return GPU memory in MiB, or None if unavailable."""
    if NVML_AVAILABLE:
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return int(mem.used / 1024**2)
        except Exception:
            # fall through to torch-based fallback
            pass

    if TORCH_AVAILABLE and torch.cuda.is_available():
        try:
            return int(torch.cuda.memory_allocated(0) / 1024**2)
        except Exception:
            return None

    return None


def load_prompts(eval_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Load prompts from JSONL or fallback to canned prompts."""
    ppath = Path(eval_cfg.get("prompt_suite", DEFAULT_PROMPT_SUITE))
    prompts: List[Dict[str, Any]] = []

    if ppath.exists():
        text = ppath.read_text(encoding="utf-8").strip()
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if "prompt" in obj:
                    prompts.append(obj)
                    continue
            except Exception:
                prompts.append({"id": len(prompts) + 1, "prompt": line})
    else:
        prompts = [
            {"id": 1, "prompt": "Summarize: The quick brown fox..."},
            {"id": 2, "prompt": "Write a Python function to reverse a list."},
        ]

    return prompts


def find_local_model_path(base: Path,
                          key: str,
                          repo_id: str,
                          _cfg: Dict[str, Any]) -> Path:
    repo_name = repo_id.split("/")[-1]
    candidate = base / key / repo_name
    if candidate.exists():
        return candidate

    alt = base / repo_name
    if alt.exists():
        return alt

    raise FileNotFoundError(f"Model folder not found: {candidate}")


def choose_weight_file(model_folder: Path,
                       gguf_name: Optional[str]) -> Path:
    if gguf_name:
        fpath = model_folder / gguf_name
        if fpath.exists():
            return fpath

    for ext in ("*.gguf", "*.safetensors", "*.pt", "*.bin"):
        found = list(model_folder.rglob(ext))
        if found:
            return found[0]

    return model_folder


def run_inference_llama(model_path: Path,
                        prompt: str,
                        gen_cfg: Dict[str, Any]) -> tuple[str, float]:
    """Run a single prompt via llama-cpp-python; return (text, elapsed_sec)."""
    model_arg = str(model_path)
    n_ctx = int(gen_cfg.get("n_ctx", 4096))
    llama = Llama(model=model_arg, n_ctx=n_ctx)  # type: ignore
    start = time.perf_counter()
    out = llama.create(  # type: ignore[attr-defined]
        prompt=prompt,
        max_tokens=int(gen_cfg.get("max_new_tokens", 256)),
        temperature=float(gen_cfg.get("temperature", 0.0)),
        top_p=float(gen_cfg.get("top_p", 0.95)),
        stop=None,
        echo=False,
    )
    end = time.perf_counter()

    try:
        text = out["choices"][0]["text"]
    except Exception:
        text = str(out)

    return text, end - start


def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Benchmarking
# ---------------------------------------------------------------------
def bench_model(key: str,
                repo_id: str,
                base_cfg: Dict[str, Any],
                eval_cfg: Dict[str, Any],
                llama_cfg: Dict[str, Any]) -> Dict[str, Any]:
    print(f"\n=== BENCH: {key} ({repo_id}) ===")
    model_folder = find_local_model_path(MODELS_DIR, key, repo_id, llama_cfg)
    weight_path = choose_weight_file(
        model_folder, llama_cfg.get("gguf_name", "") or None
    )
    print("model_folder:", model_folder)
    print("weight_path:", weight_path)

    prompts = load_prompts(eval_cfg)
    out_dir = OUTPUT_DIR / key
    safe_mkdir(out_dir)
    results_path = out_dir / f"{key}_results.jsonl"
    res_f = results_path.open("w", encoding="utf-8")

    latencies: List[float] = []
    tps: List[float] = []
    gpu_mems: List[Optional[int]] = []

    gen_cfg = eval_cfg.get("generation", {})

    if LLaMA_AVAILABLE:
        print("Using llama-cpp-python backend.")
    else:
        print("llama-cpp not available. Running dry-run timing only.")

    for item in tqdm(prompts, desc=f"prompts:{key}"):
        pid = item.get("id")
        prompt = item.get("prompt", str(item))
        mem_before = get_gpu_mem_mb()
        start = time.perf_counter()

        if LLaMA_AVAILABLE:
            try:
                text, elapsed = run_inference_llama(
                    weight_path, prompt, gen_cfg
                )
            except Exception as exc:
                text = f"[ERROR] {exc}"
                elapsed = time.perf_counter() - start
        else:
            time.sleep(0.15)
            text = "[DRYRUN] llama-cpp not installed"
            elapsed = time.perf_counter() - start

        mem_after = get_gpu_mem_mb()
        token_count = len(text.split())
        tps_val = token_count / elapsed if elapsed > 0 else 0.0

        latencies.append(elapsed * 1000.0)
        tps.append(tps_val)
        gpu_mems.append(mem_before if mem_before else mem_after)

        out_obj = {
            "model": key,
            "prompt_id": pid,
            "prompt": prompt,
            "output": text,
            "latency_s": elapsed,
            "tokens": token_count,
            "tokens_per_sec": tps_val,
            "gpu_mem_before": mem_before,
            "gpu_mem_after": mem_after,
            "timestamp": time.time(),
        }
        res_f.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
        res_f.flush()

    res_f.close()
    summary = {
        "model": key,
        "count": len(latencies),
        "latency_ms_mean": statistics.mean(latencies) if latencies else None,
        "tokens_per_sec_mean": statistics.mean(tps) if tps else None,
        "gpu_mem_mb_max": (
            max([m for m in gpu_mems if m is not None])
            if any(m is not None for m in gpu_mems) else None
        ),
    }
    print("Summary:", summary)
    return summary


# ---------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models", nargs="+", default=["llama3"],
        help="Model keys under models/ (e.g., llama3 phi3 qwen25)."
    )
    parser.add_argument("--out-csv", default=None,
                        help="Optional summary CSV path.")
    args = parser.parse_args()

    base_cfg = load_yaml(BASE_CONFIG)
    eval_cfg = load_yaml(EVAL_CONFIG)
    llama_cfg = load_yaml(LLAMA_CONFIG)

    model_map = {
        "llama3": "meta-llama/Llama-3.2-3B-Instruct",
        "phi3": "microsoft/Phi-3-mini-4k-instruct",
        "qwen25": "Qwen/Qwen2.5-3B-Instruct",
    }

    safe_mkdir(OUTPUT_DIR)
    summaries: List[Dict[str, Any]] = []

    for key in args.models:
        if key not in model_map:
            print(f"Unknown model key: {key} -> {list(model_map.keys())}")
            continue
        repo_id = model_map[key]
        try:
            s = bench_model(key, repo_id, base_cfg, eval_cfg, llama_cfg)
        except Exception as exc:
            print(f"Error benchmarking {key}: {exc}", file=sys.stderr)
            s = {"model": key, "error": str(exc)}
        summaries.append(s)

    csv_path = Path(args.out_csv) if args.out_csv else (
        OUTPUT_DIR / f"summary_{int(time.time())}.csv"
    )

    if summaries:
        with open(csv_path, "w", newline="", encoding="utf-8") as cf:
            writer = csv.DictWriter(cf, fieldnames=list(summaries[0].keys()))
            writer.writeheader()
            writer.writerows(summaries)
    print(f"\nWrote summary CSV -> {csv_path}")


if __name__ == "__main__":
    main()
