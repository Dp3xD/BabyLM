"""
Run this after both train_model_a.py and train_model_b.py finish.
Prints a side by side comparison of eval loss and perplexity.
"""
import json

a = json.load(open("model_a_final_metrics.json"))
b = json.load(open("model_b_final_metrics.json"))

print("model      eval_loss   perplexity")
print(f"A (text)   {a['eval_loss']:.4f}      {a['perplexity']:.4f}")
print(f"B (ground) {b['eval_loss']:.4f}      {b['perplexity']:.4f}")

diff = a["perplexity"] - b["perplexity"]
if diff > 0:
    print(f"\nModel B has lower perplexity by {diff:.4f}, grounding helped on this held-out set")
elif diff < 0:
    print(f"\nModel A has lower perplexity by {-diff:.4f}, grounding did not help on this held-out set")
else:
    print("\nno difference in perplexity")
