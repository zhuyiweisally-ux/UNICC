"""
UNICC AI Safety Lab — Project 2 | Step 2
Script 01: Download Public Datasets

Run FIRST on laptop or DGX Spark (needs internet).
Usage:
    pip install datasets huggingface_hub
    python scripts/01_download_datasets.py
"""

import json, csv, urllib.request
from pathlib import Path

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

# ── 1. HarmBench — primary source for adapter_redteam ────────────────────────
HARMBENCH_URL = (
    "https://raw.githubusercontent.com/centerforaisafety/HarmBench"
    "/main/data/behavior_datasets/harmbench_behaviors_text_all.csv"
)
print("Downloading HarmBench behaviors...")
try:
    urllib.request.urlretrieve(HARMBENCH_URL, RAW_DIR / "harmbench_behaviors.csv")
    with open(RAW_DIR / "harmbench_behaviors.csv") as f:
        count = sum(1 for _ in csv.reader(f)) - 1
    print(f"  OK  HarmBench: {count} behaviors saved")
except Exception as e:
    print(f"  FAIL  {e}")
    print("  Manual: clone https://github.com/centerforaisafety/HarmBench")
    print("  Copy data/behavior_datasets/harmbench_behaviors_text_all.csv -> data/raw/")

# ── 2. PKU-SafeRLHF — primary source for adapter_scoring ─────────────────────
print("\nDownloading PKU-SafeRLHF...")
try:
    from datasets import load_dataset
    for split in ["train", "test"]:
        ds = load_dataset("PKU-Alignment/PKU-SafeRLHF", split=split)
        with open(RAW_DIR / f"pku_saferlhf_{split}.jsonl", "w") as f:
            for row in ds:
                f.write(json.dumps(row) + "\n")
        print(f"  OK  PKU-SafeRLHF {split}: {len(ds)} samples")
except ImportError:
    print("  Run: pip install datasets   then retry")
except Exception as e:
    print(f"  FAIL  {e}")

# ── 3. Do-Not-Answer — refusal examples, supplements adapter_scoring ──────────
print("\nDownloading Do-Not-Answer...")
try:
    from datasets import load_dataset
    ds = load_dataset("Libr-AI/do-not-answer", split="train")
    with open(RAW_DIR / "do_not_answer.jsonl", "w") as f:
        for row in ds:
            f.write(json.dumps(row) + "\n")
    print(f"  OK  Do-Not-Answer: {len(ds)} samples")
except Exception as e:
    print(f"  FAIL  {e}")

# ── 4. SafetyBench — used for validation set ──────────────────────────────────
print("\nDownloading SafetyBench (EN)...")
try:
    from datasets import load_dataset
    ds = load_dataset("thu-coai/SafetyBench", "en", split="test")
    with open(RAW_DIR / "safetybench_en.jsonl", "w") as f:
        for row in ds:
            f.write(json.dumps(row) + "\n")
    print(f"  OK  SafetyBench EN: {len(ds)} samples")
except Exception as e:
    print(f"  FAIL  {e}")

print("\nDownloads complete. Next: python scripts/02_governance_rules.py")
