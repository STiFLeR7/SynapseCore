#!/usr/bin/env python3
"""
model_download.py

Download three core models into the models/ subfolders.

Assumes you have already authenticated with:
  huggingface-cli login
or set HF_TOKEN in the environment.
"""

from pathlib import Path
import os
from huggingface_hub import snapshot_download

BASE_DIR = Path(__file__).resolve().parents[1] / "models"

MODEL_SPECS = {
    "llama3": "meta-llama/Llama-3.2-3B-Instruct",
    "phi3": "microsoft/Phi-3-mini-4k-instruct",
    "qwen25": "Qwen/Qwen2.5-3B-Instruct",
}


def _get_token():
    """Return HF token from env or None to use CLI auth."""
    return os.environ.get("HF_TOKEN", None)


def _download_one(repo_id: str, target_dir: Path, token: str | None):
    """Snapshot download a single repo into target_dir."""
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"⬇️  Downloading {repo_id} -> {target_dir}")
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=target_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
            token=token,
            allow_patterns=[
                "*.bin",
                "*.safetensors",
                "*.json",
                "*.py",
                "*.txt",
                "*.model",
                "*.gguf",
            ],
        )
    except Exception as exc:
        print(f"⚠️  Error for {repo_id}: {exc}")


def main():
    token = _get_token()
    if token:
        print("🔐 Using HF_TOKEN from environment.")
    else:
        print(
            "🔐 No HF_TOKEN found — using huggingface-cli auth. "
            "Run 'huggingface-cli login' if needed."
        )

    for key, repo in MODEL_SPECS.items():
        out_dir = BASE_DIR / key / repo.split("/")[-1]
        _download_one(repo, out_dir, token)

    print("🏁 Downloads attempted. Inspect models/ for results.")


if __name__ == "__main__":
    main()
