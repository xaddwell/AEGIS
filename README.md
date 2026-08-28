# AEGIS: Beyond Over-Refusal

**Defending Indirect Prompt Injection via Latent Instruction Manifolds**

Official code for the EMNLP 2026 (Findings) paper *"Beyond Over-Refusal: Defending Indirect Prompt Injection via Latent Instruction Manifolds"*.

AEGIS is a training-free defense against indirect prompt injection (IPI). It leverages the spectral asymmetry between the compact *instruction manifold* and the high-entropy *knowledge manifold* of instruction-tuned LLMs:

1. **Instruction-Sensitive Projector** — a closed-form projection direction `w ∝ Σ_I⁻¹(μ_I − μ_K)` that isolates "imperative intent" from background knowledge (Asymmetric LDA with Ledoit–Wolf shrinkage).
2. **Distribution-Aware Safety Calibration** — the decision boundary is anchored to the empirical quantile of benign projection scores, mathematically bounding the false positive rate.
3. **Unified Multi-Layer Consensus** — hard/soft voting across multiple network depths; an input is flagged only when a malicious signal persists across layers (exponential FPR bound via Hoeffding).

## Repository structure

```
inststeer/                  core package
├── utils/steer.py          AsymmetricLDA, CalibratedAsymmetricLDA, MultiLayerVotingDetector (core algorithm)
├── utils/hidden_state.py   hidden-state extraction utilities
├── utils/module.py         weight-editing helpers
├── model/                  model configs (config.json) and loader
└── dataset/                calibration/test data loading and prompt formatting
train.py                    training: fit the instruction-sensitive projector
test.py                     inference/test: evaluate the fitted projector
steering.py                 SteeringDefense / SteeringConfig (defense mode)
benchmark_inststeer.py      main benchmark (detection / defense / multilayer modes)
eval_multilayer_voting.py   multi-layer consensus evaluation
demo.py                     quick end-to-end inference demo
data/
├── TrainData/data/         calibration set (50 benign + 50 injected samples)
└── TestData/               demo test set (20 benign + 20 injected samples)
```

## Setup

```bash
# Python >= 3.10, CUDA-capable GPU recommended
pip install -e .
```

## Download a model

Models are loaded from `$HF_HOME/<repo-id>` (flat layout; `HF_HOME` defaults to `~/.cache/huggingface`):

```bash
HF_HOME=${HF_HOME:-~/.cache/huggingface}
huggingface-cli download Qwen/Qwen3-4B-Instruct-2507 --local-dir $HF_HOME/Qwen/Qwen3-4B-Instruct-2507
# Llama models require accepting the license on Hugging Face first:
# huggingface-cli download meta-llama/Llama-3.1-8B-Instruct --local-dir $HF_HOME/meta-llama/Llama-3.1-8B-Instruct
```

> If you are in a region with restricted access to huggingface.co, you may set `HF_ENDPOINT=https://hf-mirror.com` before downloading.

## 1. Train (fit the projector)

```bash
python train.py \
    --model_name qwen3-4b \
    --num_samples_per_class 50 \
    --extract_layer_position 0.5 \
    --extract_token_position last \
    --device 0
```

The script formats the calibration set (`data/TrainData/data`), extracts hidden states at the
selected layer, fits `AsymmetricLDA`, and saves the projector to `data/models/`.

**Parameter selection (used in the paper):**

| Parameter | Value | Note |
|---|---|---|
| `num_samples_per_class` | 200 | converges already at ~100 per class |
| `extract_layer_position` | 0.5 | middle layers show the best ACC / lowest FPR (layers 18–27 on Qwen3-4B) |
| voting window `\|L\|` | 3 (Qwen3-4B), 7 (Llama-3/3.1-8B) | start at 50% network depth |
| per-layer target FPR | 0.05 | distribution-aware calibration; global FPR 0.29% after consensus |
| aggregation | soft voting | see `eval_multilayer_voting.py` for hard/soft modes |

## 2. Test (inference)

```bash
python test.py --model_name qwen3-4b --device 0
python benchmark_inststeer.py --mode detection --model_name qwen3-4b
python benchmark_inststeer.py --mode multilayer --model_name qwen3-4b
python benchmark_inststeer.py --mode defense --model_name qwen3-4b
python eval_multilayer_voting.py --model_name qwen3-4b
```

Metrics reported: accuracy (ACC), false positive rate (FPR), false negative rate (FNR),
true positive rate (TPR), F1, plus inference latency and memory overhead.

## 3. Demo

```bash
python demo.py                  # fits the projector on the packaged calibration set and runs detection
python demo.py --model_name llama3.1-8b
```

The demo prints per-example projection scores and aggregate ACC/FPR/FNR on the packaged test set.

## Calibration / test data format

Each dataset directory contains:

- `data.json` — a list of examples `[{"instruction": "", "data_prompt": "<text>"}]`
  (`data_prompt` is the untrusted external content);
- `label.json` — aligned list of labels (`0` = benign, `1` = injected).

`data/TestData/dataset_candidates.json` lists `[clean_dir, malicious_dir]` pairs.

## Citation

```bibtex
@misc{chen2026overrefusaldefendingindirectprompt,
      title={Beyond Over-Refusal: Defending Indirect Prompt Injection via Latent Instruction Manifolds},
      author={Jiahao Chen and Rui Yin and Xinfeng Li and Qianli Ma and Tianyu Du and Zhihui Fu and Jun Wang and Zhaoxiang Wang and Shouling Ji},
      year={2026},
      eprint={2608.22248},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
      url={https://arxiv.org/abs/2608.22248},
}
```

## Disclaimer

This repository contains research code. It ships with a small demonstration calibration set;
model weights, hidden states, and internal infrastructure are intentionally not included.
