"""
Runs EWoK evaluation on both models, same shared scoring approach
as eval_blimp.py, so the comparison stays consistent.

EWoK format: two contexts, two targets, each context correctly
pairs with one target. For Context1, model should favor Target1
over Target2, and vice versa for Context2. Since context is
identical within each comparison, comparing total log-likelihood
of the concatenated context+target directly gives the correct
conditional ranking.

Usage:
    python scripts/eval_ewok.py a checkpoints/model_a/final
    python scripts/eval_ewok.py b checkpoints/model_b/final
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


def sequence_log_likelihood(model, text):
    enc = tokenizer(text, truncation=True, max_length=SEQ_LEN, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    labels = input_ids.clone()

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    num_tokens = input_ids.shape[1]
    return -out.loss.item() * num_tokens


def run_ewok(model, data_dir):
    results = {}
    files = sorted(glob.glob(os.path.join(data_dir, "*.jsonl")))
    for filepath in files:
        task_name = os.path.basename(filepath).replace(".jsonl", "")
        correct = 0
        total = 0
        with open(filepath) as f:
            for line in f:
                item = json.loads(line)
                c1, c2 = item["Context1"], item["Context2"]
                t1, t2 = item["Target1"], item["Target2"]

                # context1 should favor target1
                ll_c1_t1 = sequence_log_likelihood(model, f"{c1} {t1}")
                ll_c1_t2 = sequence_log_likelihood(model, f"{c1} {t2}")
                if ll_c1_t1 > ll_c1_t2:
                    correct += 1
                total += 1

                # context2 should favor target2
                ll_c2_t2 = sequence_log_likelihood(model, f"{c2} {t2}")
                ll_c2_t1 = sequence_log_likelihood(model, f"{c2} {t1}")
                if ll_c2_t2 > ll_c2_t1:
                    correct += 1
                total += 1

        acc = correct / total if total > 0 else 0
        results[task_name] = acc
        print(f"{task_name}: {acc:.4f} ({correct}/{total})")
    overall = sum(results.values()) / len(results) if results else 0
    print(f"\noverall ewok accuracy: {overall:.4f}")
    return results, overall


if __name__ == "__main__":
    import sys
    model_type = sys.argv[1]
    model_path = sys.argv[2]
    data_dir = "babylm-eval/strict/evaluation_data/fast_eval/ewok_fast/evaluation_data/fast_eval/ewok_fast"

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
        state_dict_path = os.path.join(model_path, "model.safetensors")
        if os.path.exists(state_dict_path):
            from safetensors.torch import load_file
            state_dict = load_file(state_dict_path)
        else:
            state_dict = torch.load(os.path.join(model_path, "pytorch_model.bin"), map_location=device)
        model.load_state_dict(state_dict, strict=False)
        model = model.to(device)

    model.eval()
    results, overall = run_ewok(model, data_dir)

    out_name = f"ewok_results_model_{model_type}.json"
    json.dump({"per_task": results, "overall": overall}, open(out_name, "w"))
    print(f"saved to {out_name}")