# EigenTruth Benchmarks

Reproducible evaluation scripts that turn EigenTruth's diagnostics into measurable
numbers. Unlike the qualitative scripts in [`examples/`](../examples/README.md),
these produce **AUROC** against labeled data so the core hypotheses can be tested
and ablated.

可复现评测脚本，把 EigenTruth 的诊断信号变成可度量的数字（AUROC），用于检验和消融核心假设。

## `eval_truthfulqa.py`

Tests whether hidden-state geometry separates **true** from **false** statements on
TruthfulQA, in a deterministic, judge-free, single-forward-pass setup (SAPLMA-style).

### What it answers

1. **Is the manifold distance a useful detector, and does it beat perplexity?**
   `maha_last` (Mahalanobis distance from the truth manifold) vs `nll_answer`
   (answer perplexity — a cheap, strong baseline any new method must beat).
2. **Does the hyperbolic projection earn its keep?**
   `disp_hse` (Hyperbolic Semantic Entropy) vs `disp_euclid` (the same dispersion
   computed in Euclidean space). If `disp_hse` does not beat `disp_euclid`, the
   hyperbolic machinery is decoration.

### Method

- The truth manifold is built **only** from the correct answers of a held-out block
  of questions (`--manifold-questions`); evaluation runs on the remaining questions.
  Manifold-build and eval questions are disjoint, so there is no label leakage.
- Each candidate answer is scored with one forward pass at the target layer. The
  positive class (label 1) is an **incorrect** answer (the hallucination we want to
  flag); the negative class is a correct answer.
- AUROC is reported per signal. AUROC = P(score(false) > score(true)); 0.5 is chance,
  1.0 is perfect separation.

### Install and run

```bash
python -m pip install -e ".[eval]"   # adds `datasets`

# Real benchmark (downloads model weights + TruthfulQA):
python benchmarks/eval_truthfulqa.py --model Qwen/Qwen2.5-0.5B-Instruct --layer -8 --limit 200

# Sweep the target layer to find where the signal lives:
python benchmarks/eval_truthfulqa.py --layer -4
python benchmarks/eval_truthfulqa.py --layer -8
python benchmarks/eval_truthfulqa.py --layer -12

# Fast pipeline self-check (tiny model, bundled statements, no dataset download):
python benchmarks/eval_truthfulqa.py --model sshleifer/tiny-gpt2 --offline
```

Use `--json results.json` to save structured output (config + AUROC per signal) for
the record.

### How to read the results

- `maha_last > 0.5` means the manifold distance ranks false statements above true ones.
- Compare `maha_last` against `nll_answer`: geometry is only interesting if it adds
  signal over plain perplexity.
- Compare `disp_hse` against `disp_euclid`: this is the decisive ablation for the
  hyperbolic component.
- Results depend strongly on the target layer; sweep it.

### First results (indicative — `gpt2`, a weak base model)

A first end-to-end run on real TruthfulQA, committed as `results_gpt2_l-8.json`:

| signal | AUROC |
|---|---|
| `maha_last` | **0.622** |
| `disp_euclid` | 0.484 |
| `disp_hse` | 0.474 |
| `nll_answer` | 0.411 |

Setup: `gpt2` (124M base model), layer −8, manifold from 266 true statements (80 held-out
questions), 1075 eval statements (592 false / 483 true), seed 0.

Two takeaways, both consistent with the project's stated caveats:

1. **The manifold distance carries real signal.** `maha_last` (0.62) is well above chance
   and clearly beats the perplexity baseline (`nll_answer` = 0.41 — anti-correlated here,
   because a base LM finds common misconceptions *more* fluent, not less).
2. **The hyperbolic projection does not earn its keep here.** `disp_hse` (0.474) is
   marginally *below* its Euclidean counterpart `disp_euclid` (0.484), and both sit at
   chance. On this setup the hyperbolic machinery adds nothing over Euclidean dispersion.

This is committed for reproducibility, not as a strong claim. `gpt2` is a weak 2019 base
model and within-statement dispersion is a cheap proxy for sample-based semantic entropy.
Re-run on an instruction-tuned model (e.g. `Qwen2.5-0.5B-Instruct`, which needs more RAM
than an 8 GB machine comfortably provides) and sweep layers before drawing conclusions.

### Limitations (read before quoting any number)

- Forced-answer statement scoring is a **proxy** for open-generation hallucination.
  It cleanly tests the representation hypothesis but is not the same as detecting
  hallucination during free generation.
- Within-statement token dispersion is a **cheap proxy** for sample-based semantic
  entropy (Farquhar et al., 2024); it is not the full multi-sample method.
- A small model (e.g. 0.5B) and a few hundred items give wide confidence intervals.
  Treat AUROC values as indicative, not conclusive, and report `n`.
- Beating these in-house baselines is necessary but not sufficient; a real claim needs
  comparison against published detectors (semantic entropy, INSIDE/EigenScore, SAPLMA)
  on standard splits.

## 说明

`eval_truthfulqa.py` 在 TruthfulQA 上以确定性、无需 LLM 裁判、单次前向的方式，检验隐状态
几何能否分离真/假陈述。真值流形仅用留出题目的正确答案构建（与评测题目不重叠，无泄漏），
正类为错误答案（幻觉）。逐信号报告 AUROC（0.5 为随机，1.0 为完美分离）。

两个关键对比：`maha_last` vs `nll_answer`（几何是否优于困惑度基线）、`disp_hse` vs
`disp_euclid`（双曲投影是否真的有用）。结果强依赖目标层，应扫层。陈述级打分是开放生成
幻觉的代理，小模型 + 数百样本的 AUROC 置信区间较宽，结论需对照已发表方法。
