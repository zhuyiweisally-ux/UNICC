#!/bin/bash
# UNICC AI Safety Lab — Project 2 | Step 3
# run_all_finetuning.sh
#
# Runs all three fine-tuning scripts in sequence on the DGX Spark cluster.
# Each adapter trains one after the other to avoid memory conflicts.
#
# Usage (on DGX Spark):
#   chmod +x scripts/run_all_finetuning.sh
#   ./scripts/run_all_finetuning.sh
#
# Expected total time: 4-8 hours depending on cluster load

set -e  # stop on any error

echo "================================================"
echo " UNICC AI Safety Lab — Fine-tuning Pipeline"
echo " Started: $(date)"
echo "================================================"

# Install dependencies if not already installed
echo ""
echo "Checking dependencies..."
pip install transformers peft trl bitsandbytes accelerate datasets -q

# Verify GPU
echo ""
echo "GPU status:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

echo ""
echo "================================================"
echo " ADAPTER 1/3: adapter_scoring"
echo "================================================"
python scripts/07_finetune_adapter_scoring.py
echo "adapter_scoring COMPLETE at $(date)"

echo ""
echo "================================================"
echo " ADAPTER 2/3: adapter_governance"
echo "================================================"
python scripts/08_finetune_adapter_governance.py
echo "adapter_governance COMPLETE at $(date)"

echo ""
echo "================================================"
echo " ADAPTER 3/3: adapter_redteam"
echo "================================================"
python scripts/09_finetune_adapter_redteam.py
echo "adapter_redteam COMPLETE at $(date)"

echo ""
echo "================================================"
echo " ALL ADAPTERS COMPLETE"
echo " Finished: $(date)"
echo ""
echo " Saved to:"
echo "   models/adapter_scoring/"
echo "   models/adapter_governance/"
echo "   models/adapter_redteam/"
echo ""
echo " Next: python scripts/10_verify_adapters.py"
echo "================================================"
