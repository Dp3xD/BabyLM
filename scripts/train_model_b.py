"""
Model B: grounded GPT-2, trained locally on M1. Same reduced word
budget as Model A, split 70-30 text/captions (see prepare_data.py).
Saves loss history and eval perplexity for comparison against A.
"""
import json
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset as TorchDataset
from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast
from transformers import Trainer, TrainingArguments
from transformers.modeling_outputs import CausalLMOutput

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"using device: {device}")

SEQ_LEN = 256
EMBED_DIM = 768
BATCH_SIZE = 8
GRAD_ACCUM = 4
NUM_EPOCHS = 3

tokenizer = GPT2TokenizerFast.from_pretrained("BabyLM-community/BabyLM-2026-Baseline-GPT2-Strict-Small")
tokenizer.pad_token = tokenizer.eos_token


class GroundedGPT2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gpt2 = GPT2LMHeadModel(config)
        self.image_proj = nn.Linear(EMBED_DIM, config.n_embd)
        self.no_image_embed = nn.Parameter(torch.randn(config.n_embd) * 0.02)

    def forward(self, input_ids, attention_mask, labels=None, image_embedding=None, has_image=None):
        wte = self.gpt2.transformer.wte(input_ids)
        batch_size = input_ids.shape[0]
        no_image = self.no_image_embed.unsqueeze(0).expand(batch_size, -1)
        if image_embedding is not None and has_image is not None:
            projected = self.image_proj(image_embedding).to(no_image.dtype)
            mask = has_image.to(no_image.dtype).unsqueeze(-1)  # (batch, 1)
            slot = mask * projected + (1 - mask) * no_image
        else:
            slot = no_image
        slot = slot.unsqueeze(1)

        inputs_embeds = torch.cat([slot, wte], dim=1)
        full_attention_mask = torch.cat(
            [torch.ones(batch_size, 1, device=attention_mask.device), attention_mask], dim=1
        )
        full_labels = None
        if labels is not None:
            ignore_col = torch.full((batch_size, 1), -100, device=labels.device, dtype=labels.dtype)
            full_labels = torch.cat([ignore_col, labels], dim=1)

        outputs = self.gpt2(inputs_embeds=inputs_embeds, attention_mask=full_attention_mask, labels=full_labels)
        return CausalLMOutput(loss=outputs.loss, logits=outputs.logits)


class GroundedDataset(TorchDataset):
    def __init__(self, examples, tokenizer, seq_len):
        # tokenize everything once upfront, batched, instead of
        # calling the tokenizer per example on every training step
        texts = [ex["text"] for ex in examples]
        enc = tokenizer(
            texts, truncation=True, max_length=seq_len,
            padding="max_length", return_tensors="pt",
        )
        self.input_ids = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]

        labels = self.input_ids.clone()
        labels[self.attention_mask == 0] = -100
        self.labels = labels

        has_image = [ex["embedding"] is not None for ex in examples]
        self.has_image = torch.tensor(has_image)
        embeddings = [
            torch.tensor(ex["embedding"], dtype=torch.float32) if ex["embedding"] is not None else torch.zeros(EMBED_DIM)
            for ex in examples
        ]
        self.embeddings = torch.stack(embeddings)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
            "image_embedding": self.embeddings[idx],
            "has_image": self.has_image[idx],
        }


combined = json.load(open("data/model_b_combined.json"))
split_idx = int(len(combined) * 0.98)
train_data = combined[:split_idx]
eval_data = combined[split_idx:]
print(f"train examples: {len(train_data)}, eval examples: {len(eval_data)}")

train_ds = GroundedDataset(train_data, tokenizer, SEQ_LEN)
eval_ds = GroundedDataset(eval_data, tokenizer, SEQ_LEN)

config = GPT2Config(
    vocab_size=tokenizer.vocab_size,
    n_positions=SEQ_LEN + 1,
    n_embd=768,
    n_layer=12,
    n_head=12,
    activation_function="gelu_new",
    bos_token_id=1,
    eos_token_id=2,
)
model = GroundedGPT2(config).to(device)
print(f"model params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

args = TrainingArguments(
    output_dir="checkpoints/model_b",
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=NUM_EPOCHS,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    logging_steps=50,
    report_to="none",
    save_safetensors=False,  # tied weights crash safetensors, use plain torch save instead
)

trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds)

trainer.train()

eval_results = trainer.evaluate()
perplexity = math.exp(eval_results["eval_loss"])
print(f"final eval loss: {eval_results['eval_loss']:.4f}")
print(f"final eval perplexity: {perplexity:.4f}")

trainer.save_model("checkpoints/model_b/final")
tokenizer.save_pretrained("checkpoints/model_b/final")

json.dump(trainer.state.log_history, open("model_b_log_history.json", "w"))
json.dump({"eval_loss": eval_results["eval_loss"], "perplexity": perplexity}, open("model_b_final_metrics.json", "w"))

print("done, final model and logs saved")