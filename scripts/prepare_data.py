"""
Builds training data for both models with a matched, reduced word
budget (scaled down from the full 10M for local time constraints).
Model A: all plain text. Model B: 70-30 text/caption split.
Same total word budget for both, so the comparison stays fair.
"""
import json
import random
import numpy as np
from datasets import load_dataset

random.seed(42)

TOTAL_WORD_BUDGET = 3_500_000  # ~35% of full 10M, adjust if needed
TEXT_RATIO_B = 0.7

strict_small = load_dataset("BabyLM-community/BabyLM-2026-Strict-Small", split="train")
rows = list(strict_small)
random.shuffle(rows)

# ---- Model A: all text, TOTAL_WORD_BUDGET words ----
model_a_examples = []
word_count = 0
for row in rows:
    words = row["text"].split()
    if word_count + len(words) > TOTAL_WORD_BUDGET:
        continue
    model_a_examples.append({"text": row["text"]})
    word_count += len(words)
    if word_count >= TOTAL_WORD_BUDGET:
        break

print(f"model a: {len(model_a_examples)} examples, {word_count} words")
json.dump(model_a_examples, open("data/model_a_data.json", "w"))

# ---- Model B: 70% text + 30% captions, same TOTAL_WORD_BUDGET ----
text_budget_b = int(TOTAL_WORD_BUDGET * TEXT_RATIO_B)
caption_budget_b = TOTAL_WORD_BUDGET - text_budget_b

random.shuffle(rows)  # reshuffle so B's text slice isn't identical to A's
text_examples_b = []
word_count = 0
for row in rows:
    words = row["text"].split()
    if word_count + len(words) > text_budget_b:
        continue
    text_examples_b.append({"text": row["text"], "embedding": None})
    word_count += len(words)
    if word_count >= text_budget_b:
        break

print(f"model b text portion: {len(text_examples_b)} examples, {word_count} words")

local_narr_captions = json.load(open("data/multimodal_data/local_narr_captions.json"))
local_narr_embeds = np.load("data/multimodal_data/local_narr_dino_v2_states.npy", mmap_mode="r")

local_idx = list(range(len(local_narr_captions)))
random.shuffle(local_idx)

grounded_examples = []
cap_word_count = 0
for i in local_idx:
    cap = local_narr_captions[i]
    words = cap.split()
    if cap_word_count + len(words) > caption_budget_b:
        continue
    grounded_examples.append({"text": cap, "embedding": np.array(local_narr_embeds[i]).tolist()})
    cap_word_count += len(words)
    if cap_word_count >= caption_budget_b:
        break

combined_b = text_examples_b + grounded_examples
random.shuffle(combined_b)

print(f"model b total: {len(combined_b)} examples, {word_count + cap_word_count} words")

# split into a light text-only json and a compact numpy embeddings
# array, aligned by index, avoids parsing huge float lists from json
texts_out = []
embeds_out = np.zeros((len(combined_b), 768), dtype=np.float32)
for i, ex in enumerate(combined_b):
    texts_out.append({"text": ex["text"], "has_image": ex["embedding"] is not None})
    if ex["embedding"] is not None:
        embeds_out[i] = np.array(ex["embedding"], dtype=np.float32)

json.dump(texts_out, open("data/model_b_texts.json", "w"))
np.save("data/model_b_embeddings.npy", embeds_out)
print("saved data/model_b_texts.json and data/model_b_embeddings.npy")

print(f"model b total: {len(combined_b)} examples, {word_count + cap_word_count} words")
json.dump(combined_b, open("data/model_b_combined.json", "w"))

print("saved data/model_a_data.json and data/model_b_combined.json")
