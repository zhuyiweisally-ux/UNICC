"""
UNICC AI Safety Lab — Project 2 | Step 2
Script 06: Train / Validation Split

Splits all three processed datasets into 90% train / 10% validation.
Uses a fixed random seed for reproducibility.

Usage:
    python scripts/06_split_data.py
Input:
    data/processed/scoring_train.jsonl
    data/processed/governance_train.jsonl
    data/processed/redteam_train.jsonl
Output:
    data/final/scoring_train.jsonl          data/final/scoring_val.jsonl
    data/final/governance_train.jsonl       data/final/governance_val.jsonl
    data/final/redteam_train.jsonl          data/final/redteam_val.jsonl
    data/final/combined_val.jsonl           (shared validation set across all modules)
    data/final/dataset_statistics.json
"""

import json, random
from pathlib import Path
from collections import Counter

random.seed(42)
Path("data/final").mkdir(parents=True, exist_ok=True)

DATASETS = {
    "scoring":    "data/processed/scoring_train.jsonl",
    "governance": "data/processed/governance_train.jsonl",
    "redteam":    "data/processed/redteam_train.jsonl",
}

SPLIT_RATIO  = 0.90  # 90% train, 10% validation
stats = {}
combined_val = []

print("Splitting datasets 90/10 train/validation (seed=42)...\n")

for name, path_str in DATASETS.items():
    path = Path(path_str)
    if not path.exists():
        print(f"  SKIP  {name} — file not found: {path_str}")
        print(f"         Run the corresponding format script first.")
        continue

    with open(path) as f:
        samples = [json.loads(line) for line in f if line.strip()]

    random.shuffle(samples)
    split_idx  = int(len(samples) * SPLIT_RATIO)
    train_set  = samples[:split_idx]
    val_set    = samples[split_idx:]

    # Write split files
    train_path = Path(f"data/final/{name}_train.jsonl")
    val_path   = Path(f"data/final/{name}_val.jsonl")

    with open(train_path, "w") as f:
        for s in train_set:
            f.write(json.dumps(s) + "\n")

    with open(val_path, "w") as f:
        for s in val_set:
            f.write(json.dumps(s) + "\n")

    combined_val.extend(val_set)

    # Compute verdict distribution
    verdicts = []
    for s in samples:
        try:
            asst = json.loads(s["messages"][2]["content"])
            v = asst.get("verdict") or asst.get("safety_verdict") or "UNKNOWN"
            verdicts.append(v)
        except Exception:
            pass

    stats[name] = {
        "total":      len(samples),
        "train":      len(train_set),
        "validation": len(val_set),
        "split":      f"{SPLIT_RATIO:.0%} / {1-SPLIT_RATIO:.0%}",
        "seed":       42,
        "verdict_distribution": dict(Counter(verdicts)),
    }

    print(f"  {name}:")
    print(f"    Total  : {len(samples)}")
    print(f"    Train  : {len(train_set)}")
    print(f"    Val    : {len(val_set)}")
    print(f"    Verdicts: {dict(Counter(verdicts))}")
    print()

# Write combined validation set
combined_val_path = Path("data/final/combined_val.jsonl")
random.shuffle(combined_val)
with open(combined_val_path, "w") as f:
    for s in combined_val:
        f.write(json.dumps(s) + "\n")

# Write statistics file
stats["combined_validation"] = {
    "total": len(combined_val),
    "description": "Merged validation set from all three adapters for cross-module benchmarking"
}

with open("data/final/dataset_statistics.json", "w") as f:
    json.dump(stats, f, indent=2)

print(f"Combined validation set: {len(combined_val)} samples -> data/final/combined_val.jsonl")
print(f"Statistics saved -> data/final/dataset_statistics.json")
print("\nStep 2 data pipeline complete.")
print("Next: read NIST AI 600-1 and IMDA PDFs to verify/extend governance rules,")
print("      then proceed to Step 3 (fine-tuning scripts) when cluster is ready.")
