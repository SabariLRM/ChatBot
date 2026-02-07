# chat_lora_mac.py
# Chat with your LoRA adapter on macOS (MPS)
# Uses BASE_MODEL + ADAPTER_DIR env vars if you want.
#
# Run:
#   python chat_lora_mac.py
# Or:
#   BASE_MODEL=Qwen/Qwen2.5-1.5B-Instruct ADAPTER_DIR=out_lora_mac python chat_lora_mac.py

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
ADAPTER_DIR = os.environ.get("ADAPTER_DIR", "out_lora_mac")

if not torch.backends.mps.is_available():
    raise RuntimeError("MPS not available. This script is for Apple Silicon macOS.")

# Load tokenizer (saved with adapter, falls back to base if needed)
try:
    tok = AutoTokenizer.from_pretrained(ADAPTER_DIR, use_fast=True)
except Exception:
    tok = AutoTokenizer.from_pretrained(BASE, use_fast=True)

if tok.pad_token is None:
    tok.pad_token = tok.eos_token

# Load base model on MPS (fp16)
base = AutoModelForCausalLM.from_pretrained(
    BASE,
    device_map={"": "mps"},
    torch_dtype=torch.float16,
)

# Attach LoRA adapter
model = PeftModel.from_pretrained(base, ADAPTER_DIR)
model.eval()

def reply(history_text: str, max_new_tokens=200):
    inputs = tok(history_text, return_tensors="pt").to("mps")
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.8,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tok.eos_token_id,
        )
    return tok.decode(out[0], skip_special_tokens=True)

print("✅ Chat ready (Mac/MPS). Ctrl+C to exit.")
print(f"Base: {BASE}")
print(f"Adapter: {ADAPTER_DIR}\n")

history = ""
while True:
    try:
        user = input("You: ").strip()
        if not user:
            continue

        history += f"User: {user}\nAssistant:"
        full = reply(history, max_new_tokens=220)

        # Extract only the newest assistant part
        if "Assistant:" in full:
            bot = full.split("Assistant:")[-1].strip()
        else:
            bot = full.strip()

        # Stop if it starts printing a new user turn
        bot = bot.split("\nUser:", 1)[0].strip()
        print("Bot:", bot, "\n")

        # Keep context from growing forever (last ~4000 chars)
        history = full[-4000:]

    except KeyboardInterrupt:
        print("\nbye")
        break
