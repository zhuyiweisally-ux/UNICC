"""
UNICC AI Safety Lab — Project 2 | Step 3
Script 07: Fine-tune adapter_scoring

Trains a QLoRA adapter on top of Llama 3.1 8B Instruct to produce
the ScoringExpert module — calibrated 0-5 safety dimension scoring.

RUN ON DGX SPARK CLUSTER ONLY — requires GPU with 16GB+ VRAM.

Setup (run once on cluster):
    pip install transformers peft trl bitsandbytes accelerate datasets

Usage:
    python scripts/07_finetune_adapter_scoring.py

Output:
    models/adapter_scoring/   — saved LoRA adapter checkpoint
    logs/scoring_training.log — training loss curve
"""

import os, json, logging
from pathlib import Path
from datetime import datetime

# ── Logging ───────────────────────────────────────────────────────────────────
Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("logs/scoring_training.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
BASE_MODEL    = "meta-llama/Llama-3.1-8B-Instruct"
ADAPTER_NAME  = "adapter_scoring"
TRAIN_FILE    = "data/final/scoring_train.jsonl"
VAL_FILE      = "data/final/scoring_val.jsonl"
OUTPUT_DIR    = f"models/{ADAPTER_NAME}"
SEED          = 42

# QLoRA hyperparameters
LORA_R        = 16       # rank — higher = more capacity, more memory
LORA_ALPHA    = 32       # scaling factor (typically 2x rank)
LORA_DROPOUT  = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]

# Training hyperparameters
NUM_EPOCHS         = 3
BATCH_SIZE         = 4    # per device — increase if VRAM allows
GRAD_ACCUM_STEPS   = 4    # effective batch = BATCH_SIZE * GRAD_ACCUM = 16
LEARNING_RATE      = 2e-4
MAX_SEQ_LEN        = 2048
WARMUP_RATIO       = 0.05
LR_SCHEDULER       = "cosine"
SAVE_STEPS         = 100
EVAL_STEPS         = 100
LOGGING_STEPS      = 10

def load_dataset_from_jsonl(path):
    """Load JSONL file into HuggingFace Dataset format."""
    from datasets import Dataset
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return Dataset.from_list(records)

def format_chat_to_text(example, tokenizer):
    """
    Convert messages array to the model's chat template string.
    Llama 3.1 uses a specific <|begin_of_text|> format.
    """
    return tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False
    )

def main():
    log.info(f"Starting fine-tuning: {ADAPTER_NAME}")
    log.info(f"Base model: {BASE_MODEL}")
    log.info(f"Train file: {TRAIN_FILE}")
    log.info(f"Output dir: {OUTPUT_DIR}")

    # Verify data files exist
    for f in [TRAIN_FILE, VAL_FILE]:
        if not Path(f).exists():
            log.error(f"File not found: {f}. Run Step 2 scripts first.")
            raise FileNotFoundError(f)

    import torch
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM,
        BitsAndBytesConfig, TrainingArguments
    )
    from peft import LoraConfig, get_peft_model, TaskType
    from trl import SFTTrainer

    log.info(f"GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log.info(f"GPU: {torch.cuda.get_device_name(0)}")
        log.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── 1. Load tokenizer ─────────────────────────────────────────────────────
    log.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── 2. Load base model in 4-bit (QLoRA) ──────────────────────────────────
    log.info("Loading base model in 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",          # NormalFloat4 — best for LLMs
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,     # saves ~0.4 bits per parameter
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",                  # auto-distribute across GPUs
        trust_remote_code=True,
    )
    model.config.use_cache = False          # required for gradient checkpointing
    model.config.pretraining_tp = 1

    # ── 3. Attach LoRA adapter ────────────────────────────────────────────────
    log.info(f"Attaching LoRA adapter (r={LORA_R}, alpha={LORA_ALPHA})...")
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    # Expected output: ~0.05% of parameters are trainable

    # ── 4. Load datasets ──────────────────────────────────────────────────────
    log.info("Loading training and validation datasets...")
    train_dataset = load_dataset_from_jsonl(TRAIN_FILE)
    val_dataset   = load_dataset_from_jsonl(VAL_FILE)
    log.info(f"Train: {len(train_dataset)} samples | Val: {len(val_dataset)} samples")

    # ── 5. Training arguments ─────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type=LR_SCHEDULER,
        warmup_ratio=WARMUP_RATIO,
        fp16=False,
        bf16=True,                          # DGX Spark supports bfloat16
        max_grad_norm=0.3,
        weight_decay=0.001,
        optim="paged_adamw_32bit",          # memory-efficient optimizer
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        evaluation_strategy="steps",
        eval_steps=EVAL_STEPS,
        logging_steps=LOGGING_STEPS,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",                   # set to "wandb" if you have it
        seed=SEED,
        gradient_checkpointing=True,        # saves VRAM at cost of speed
    )

    # ── 6. SFT Trainer ────────────────────────────────────────────────────────
    log.info("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
        max_seq_length=MAX_SEQ_LEN,
        dataset_text_field=None,
        formatting_func=lambda x: [
            format_chat_to_text(ex, tokenizer) for ex in x
        ] if isinstance(x, list) else format_chat_to_text(x, tokenizer),
        packing=False,
    )

    # ── 7. Train ──────────────────────────────────────────────────────────────
    log.info("Starting training...")
    start_time = datetime.now()
    trainer.train()
    elapsed = datetime.now() - start_time
    log.info(f"Training complete in {elapsed}")

    # ── 8. Save adapter ───────────────────────────────────────────────────────
    log.info(f"Saving adapter to {OUTPUT_DIR}...")
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Save training metadata
    metadata = {
        "adapter_name":   ADAPTER_NAME,
        "base_model":     BASE_MODEL,
        "train_samples":  len(train_dataset),
        "val_samples":    len(val_dataset),
        "epochs":         NUM_EPOCHS,
        "lora_r":         LORA_R,
        "lora_alpha":     LORA_ALPHA,
        "learning_rate":  LEARNING_RATE,
        "training_time":  str(elapsed),
        "completed_at":   datetime.now().isoformat(),
    }
    with open(f"{OUTPUT_DIR}/training_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    log.info(f"adapter_scoring saved to {OUTPUT_DIR}/")
    log.info("Next: python scripts/08_finetune_adapter_governance.py")

if __name__ == "__main__":
    main()
