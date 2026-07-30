"""
Model A: text-only GPT-2, trained locally on M1. Reduced word
budget (see prepare_data.py). Saves loss history and eval
perplexity so we can compare against Model B afterward.
"""
import json
import math
import torch
from datasets import Dataset
from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"using device: {device}")

SEQ_LEN = 256
BATCH_SIZE = 8
GRAD_ACCUM = 4
NUM_EPOCHS = 3

data = json.load(open("data/model_a_data.json"))
random_seed_split = int(len(data) * 0.98)
train_data = data[:random_seed_split]
eval_data = data[random_seed_split:]
print(f"train examples: {len(train_data)}, eval examples: {len(eval_data)}")

tokenizer = GPT2TokenizerFast.from_pretrained("BabyLM-community/BabyLM-2026-Baseline-GPT2-Strict-Small")
tokenizer.pad_token = tokenizer.eos_token

def tokenize_fn(batch):
    return tokenizer(batch["text"], truncation=True, max_length=SEQ_LEN)

train_ds = Dataset.from_list(train_data).map(tokenize_fn, batched=True, remove_columns=["text"])
eval_ds = Dataset.from_list(eval_data).map(tokenize_fn, batched=True, remove_columns=["text"])

config = GPT2Config(
    vocab_size=tokenizer.vocab_size,
    n_positions=SEQ_LEN,
    n_embd=768,
    n_layer=12,
    n_head=12,
    activation_function="gelu_new",
    bos_token_id=1,
    eos_token_id=2,
)
model = GPT2LMHeadModel(config).to(device)
print(f"model params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

args = TrainingArguments(
    output_dir="checkpoints/model_a",
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=NUM_EPOCHS,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    logging_steps=50,
    report_to="none",
)

trainer = Trainer(
    model=model, args=args,
    train_dataset=train_ds, eval_dataset=eval_ds,
    data_collator=collator,
)

trainer.train()

eval_results = trainer.evaluate()
perplexity = math.exp(eval_results["eval_loss"])
print(f"final eval loss: {eval_results['eval_loss']:.4f}")
print(f"final eval perplexity: {perplexity:.4f}")

trainer.save_model("checkpoints/model_a/final")
tokenizer.save_pretrained("checkpoints/model_a/final")

json.dump(trainer.state.log_history, open("model_a_log_history.json", "w"))
json.dump({"eval_loss": eval_results["eval_loss"], "perplexity": perplexity}, open("model_a_final_metrics.json", "w"))

print("done, final model and logs saved")
