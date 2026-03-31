"""
UNICC AI Safety Lab — Project 2 | Step 3
Script 09: Fine-tune adapter_redteam

Trains a QLoRA adapter to produce the RedTeamExpert module —
generates adversarial probes and judges refusal quality.

RUN ON DGX SPARK CLUSTER ONLY.

Usage:
    python scripts/09_finetune_adapter_redteam.py

Output:
    models/adapter_redteam/
    logs/redteam_training.log
"""

import os, json, logging
from pathlib import Path
from datetime import datetime

Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("logs/redteam_training.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
BASE_MODEL   = "meta-llama/Llama-3.1-8B-Instruct"
ADAPTER_NAME = "adapter_redteam"
TRAIN_FILE   = "data/final/redteam_train.jsonl"
VAL_FILE     = "data/final/redteam_val.jsonl"
OUTPUT_DIR   = f"models/{ADAPTER_NAME}"
SEED         = 42

# LoRA — standard rank for redteam
LORA_R        = 16
LORA_ALPHA    = 32
LORA_DROPOUT  = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]

# Training
NUM_EPOCHS       = 3
BATCH_SIZE       = 4
GRAD_ACCUM_STEPS = 4
LEARNING_RATE    = 2e-4
MAX_SEQ_LEN      = 2048
WARMUP_RATIO     = 0.05
LR_SCHEDULER     = "cosine"
SAVE_STEPS       = 100
EVAL_STEPS       = 100
LOGGING_STEPS    = 10

def load_dataset_from_jsonl(path):
    from datasets import Dataset
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return Dataset.from_list(records)

def format_chat_to_text(example, tokenizer):
    return tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False
    )

def main():
    log.info(f"Starting fine-tuning: {ADAPTER_NAME}")
    log.info("This adapter learns BOTH to attack and to judge refusals")

    for f in [TRAIN_FILE, VAL_FILE]:
        if not Path(f).exists():
            log.error(f"File not found: {f}")
            raise FileNotFoundError(f)

    import torch
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM,
        BitsAndBytesConfig, TrainingArguments
    )
    from peft import LoraConfig, get_peft_model, TaskType
    from trl import SFTTrainer

    log.info(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NOT FOUND'}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

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

    train_dataset = load_dataset_from_jsonl(TRAIN_FILE)
    val_dataset   = load_dataset_from_jsonl(VAL_FILE)
    log.info(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")

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
        bf16=True,
        max_grad_norm=0.3,
        weight_decay=0.001,
        optim="paged_adamw_32bit",
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        evaluation_strategy="steps",
        eval_steps=EVAL_STEPS,
        logging_steps=LOGGING_STEPS,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        seed=SEED,
        gradient_checkpointing=True,
    )

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

    log.info("Training...")
    start_time = datetime.now()
    trainer.train()
    elapsed = datetime.now() - start_time
    log.info(f"Training complete in {elapsed}")

    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    metadata = {
        "adapter_name":  ADAPTER_NAME,
        "base_model":    BASE_MODEL,
        "train_samples": len(train_dataset),
        "val_samples":   len(val_dataset),
        "epochs":        NUM_EPOCHS,
        "lora_r":        LORA_R,
        "lora_alpha":    LORA_ALPHA,
        "learning_rate": LEARNING_RATE,
        "training_time": str(elapsed),
        "completed_at":  datetime.now().isoformat(),
    }
    with open(f"{OUTPUT_DIR}/training_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    log.info(f"adapter_redteam saved to {OUTPUT_DIR}/")
    log.info("All three adapters complete! Proceed to Step 4.")

if __name__ == "__main__":
    main()
