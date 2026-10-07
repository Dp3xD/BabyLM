# Does Visual Grounding Help a Small Language Model Learn More Efficiently?

![Python](https://img.shields.io/badge/python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)
![License](https://img.shields.io/badge/license-MIT-green)

A controlled study under BabyLM Challenge constraints — **CS6120 (NLP), Northeastern University**.

We pretrain a GPT-2 model from scratch under the BabyLM Challenge's small-data setting and ask whether adding paired image–caption data (visual grounding) helps it learn more efficiently than a text-only model trained on the same word budget. Two models are trained under an identical, matched budget:

- **Model A (text-only):** standard GPT-2, trained on plain text from the Strict-Small corpus.
- **Model B (grounded):** the same backbone, with a single prepended "grounding slot" per sequence. Caption examples get a projected DINOv2 image embedding in that slot; text-only examples get a learned placeholder. The model is 70% text / 30% image–caption pairs by word count.

**Full report:** `Does_Visual_Grounding_Help_a_Small_Language_Model_Learn_More_Efficiently.pdf`

## Key Results

| Metric | Model A | Model B |
| --- | --- | --- |
| Eval loss | 3.591 | **3.104** |
| Perplexity | 36.27 | **22.29** |
| BLiMP (overall) | **60.43%** | 58.46% |
| EWoK (overall) | 50.09% | 49.95% |
| GLUE (avg. accuracy) | 63.03% | — (see Limitations) |

Grounding produced a large perplexity improvement (39% reduction) but did not improve — and slightly hurt — BLiMP performance (statistically significant, z=3.28, p=0.001), concentrated in negative polarity items and negation (−5.6 pts), while quantifiers/existentials actually favored Model B (+3.1 pts). Both models sat at chance on EWoK. Full analysis, category-level breakdown, and statistical testing are in the report.

## Repository Structure

```
.
├── scripts/
│   ├── prepare_data.py            # builds matched-budget training data for both models
│   ├── train_model_a.py           # trains Model A locally
│   ├── train_model_b.py           # trains Model B locally (grounded architecture)
│   ├── train_model_b_colab.ipynb  # Colab GPU version of Model B training
│   ├── eval_blimp.py              # custom BLiMP scorer, works for both models
│   ├── eval_ewok.py               # custom EWoK scorer, works for both models
│   ├── eval_glue_colab.ipynb      # GLUE fine-tuning via the official harness (Model A)
│   └── compare_results.py         # prints side-by-side comparison of both models
├── babylm_glue_results/           # Model A's GLUE fine-tuning results (all 7 tasks)
├── model_a_final_metrics.json / model_a_log_history.json
├── model_b_final_metrics.json / model_b_log_history.json
├── blimp_results_model_a.json / blimp_results_model_b.json
├── ewok_results_model_a.json / ewok_results_model_b.json
├── requirements.txt
└── .gitignore
```

## Reproducing This Work

**1. Set up environment**

```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**2. Prepare training data**

```
python scripts/prepare_data.py
```

Downloads Strict-Small from HuggingFace and builds a matched-budget split for both models. Requires the multimodal caption/embedding files (Localized Narratives + Conceptual Captions with DINOv2 embeddings) locally under `data/multimodal_data/` — sourced from the BabyLM 2025 OSF release, since the 2026 HuggingFace collection did not include the multimodal component at the time of this project. See the report's Data section for details.

**3. Train**

```
python scripts/train_model_a.py
python scripts/train_model_b.py     # or use train_model_b_colab.ipynb for GPU training
```

**4. Evaluate**

```
python scripts/eval_blimp.py a checkpoints/model_a/final
python scripts/eval_blimp.py b checkpoints/model_b/final
python scripts/eval_ewok.py a checkpoints/model_a/final
python scripts/eval_ewok.py b checkpoints/model_b/final
```

GLUE fine-tuning for Model A uses the official `babylm-org/babylm-eval` harness (`scripts/eval_glue_colab.ipynb`); Model B's GLUE evaluation was not completed (see Limitations).

**5. Compare**

```
python scripts/compare_results.py
```

## Limitations

Both models were trained on a reduced 3.5M-word budget (35% of the full 10M) for 3 epochs rather than the full challenge budget, due to compute and time constraints. Model A trained locally in fp32; Model B trained on a cloud GPU in fp16. Model B's GLUE evaluation and the RQ3 grounding-method ablation were not completed in the project timeline. Full details are in the report's Limitations section.

## Data Sources

- **Strict-Small text corpus:** `BabyLM-community/BabyLM-2026-Strict-Small` (HuggingFace)
- **Official baseline architecture/tokenizer:** `BabyLM-community/BabyLM-2026-Baseline-GPT2-Strict-Small` (HuggingFace)
- **Multimodal data (captions + DINOv2 embeddings):** BabyLM 2025 OSF release
- **Evaluation pipeline:** `babylm-org/babylm-eval`

## Team

Divya Patel, James Jacob, Ramandeep Singh — Khoury College of Computer Sciences, Northeastern University

## License

MIT
