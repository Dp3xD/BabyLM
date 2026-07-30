"""
Runs BLiMP-style evaluation on both models using the same scoring
logic, so the comparison is apples to apples rather than mixing
the official harness (built for standard HF models) with a custom
script for the grounded model.

For each sentence pair, computes total log-likelihood under the
model and checks whether sentence_good scores higher than
sentence_bad, this is the standard BLiMP method.

Usage:
    python scripts/eval_blimp.py a checkpoints/model_a/final
    python scripts/eval_blimp.py b checkpoints/model_b/final
"""
import os
import json
import glob
import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast
from transformers.modeling_outputs import CausalLMOutput

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"using device: {device}")

SEQ_LEN = 256
EMBED_DIM = 768

tokenizer = GPT2TokenizerFast.from_pretrained("BabyLM-community/BabyLM-2026-Baseline-GPT2-Strict-Small")
tokenizer.pad_token = tokenizer.eos_token


class GroundedGPT2(nn.Module):
    _keys_to_ignore_on_save = None

    def __init__(self, config):
        super().__init__()
        self.gpt2 = GPT2LMHeadModel(config)
        self.image_proj = nn.Linear(EMBED_DIM, config.n_embd)
        self.no_image_embed = nn.Parameter(torch.randn(config.n_embd) * 0.02)

    def forward(self, input_ids, attention_mask, labels=None):
        wte = self.gpt2.transformer.wte(input_ids)
        batch_size = input_ids.shape[0]
        slot = self.no_image_embed.unsqueeze(0).expand(batch_size, -1).unsqueeze(1)

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


def sentence_log_likelihood(model, sentence):
    enc = tokenizer(sentence, truncation=True, max_length=SEQ_LEN, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    labels = input_ids.clone()

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    # loss is mean negative log-likelihood per token, convert to
    # total log-likelihood so longer/shorter sentences compare fairly
    num_tokens = input_ids.shape[1]
    total_log_likelihood = -out.loss.item() * num_tokens
    return total_log_likelihood


def run_blimp(model, data_dir):
    results = {}
    files = sorted(glob.glob(os.path.join(data_dir, "*.jsonl")))
    for filepath in files:
        task_name = os.path.basename(filepath).replace(".jsonl", "")
        correct = 0
        total = 0
        with open(filepath) as f:
            for line in f:
                item = json.loads(line)
                good_ll = sentence_log_likelihood(model, item["sentence_good"])
                bad_ll = sentence_log_likelihood(model, item["sentence_bad"])
                if good_ll > bad_ll:
                    correct += 1
                total += 1
        acc = correct / total if total > 0 else 0
        results[task_name] = acc
        print(f"{task_name}: {acc:.4f} ({correct}/{total})")
    overall = sum(results.values()) / len(results) if results else 0
    print(f"\noverall blimp accuracy: {overall:.4f}")
    return results, overall


if __name__ == "__main__":
    import sys
    model_type = sys.argv[1]  # "a" or "b"
    model_path = sys.argv[2]  # path to final model dir
    data_dir = "babylm-eval/strict/evaluation_data/fast_eval/blimp_fast"

    if model_type == "a":
        model = GPT2LMHeadModel.from_pretrained(model_path).to(device)
    else:
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
        model = GroundedGPT2(config)
        state_dict_path = os.path.join(model_path, "pytorch_model.bin")
        if not os.path.exists(state_dict_path):
            state_dict_path = os.path.join(model_path, "model.safetensors")
            from safetensors.torch import load_file
            state_dict = load_file(state_dict_path)
        else:
            state_dict = torch.load(state_dict_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        model = model.to(device)

    model.eval()
    results, overall = run_blimp(model, data_dir)

    out_name = f"blimp_results_model_{model_type}.json"
    json.dump({"per_task": results, "overall": overall}, open(out_name, "w"))
    print(f"saved to {out_name}")