import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model

# -----------------------
# Config (env overridable)
# -----------------------
MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
DATA_PATH = os.environ.get("DATA_PATH", "training_data.jsonl")
OUT_DIR = os.environ.get("OUT_DIR", "out_lora_mac")

# Mac-friendly defaults (reduce if OOM)
MAX_SEQ_LEN = int(os.environ.get("MAX_SEQ_LEN", "256"))
MAX_STEPS   = int(os.environ.get("MAX_STEPS", "300"))
BATCH_SIZE  = int(os.environ.get("BATCH_SIZE", "1"))
GRAD_ACCUM  = int(os.environ.get("GRAD_ACCUM", "8"))
LR          = float(os.environ.get("LR", "2e-4"))

if not torch.backends.mps.is_available():
    raise RuntimeError("MPS not available. You need Apple Silicon + macOS with MPS support.")

# -----------------------
# Load dataset
# -----------------------
ds = load_dataset("json", data_files=DATA_PATH, split="train")

tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

def format_text(ex):
    # One supervised sample = prompt + response
    text = f"{ex['input']}{ex['output']}{tok.eos_token}"
    return {"text": text}

ds = ds.map(format_text, remove_columns=ds.column_names, desc="Formatting dataset")

def tokenize_fn(ex):
    out = tok(
        ex["text"],
        truncation=True,
        max_length=MAX_SEQ_LEN,
        padding="max_length",
    )
    # Labels = input_ids for causal LM
    out["labels"] = out["input_ids"].copy()
    return out

tokenized = ds.map(tokenize_fn, remove_columns=["text"], desc="Tokenizing")

# -----------------------
# Load model (fp16 on MPS) + LoRA
# -----------------------
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    device_map={"": "mps"},
    torch_dtype=torch.float16,  # best bet on Mac
)

lora = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)
model = get_peft_model(model, lora)

# -----------------------
# Trainer + tqdm progress bar (ETA is built-in)
# -----------------------
args = TrainingArguments(
    output_dir=OUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    max_steps=MAX_STEPS,
    warmup_steps=20,
    logging_steps=10,
    save_steps=100,
    report_to="none",
    disable_tqdm=False,      # <-- tqdm progress bar + ETA
    log_level="info",
    fp16=True,
    bf16=False,
    optim="adamw_torch",     # Mac-safe
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized,
    data_collator=data_collator,
)

trainer.train()

# Save LoRA adapter + tokenizer
model.save_pretrained(OUT_DIR)
tok.save_pretrained(OUT_DIR)
print(f"\n✅ Done. Saved LoRA adapter to: {OUT_DIR}\n")
