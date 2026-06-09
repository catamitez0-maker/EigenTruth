"""EigenTruth 基准评测 — 隐状态几何信号能否分离真/假陈述 (AUROC)。
EigenTruth benchmark — can hidden-state geometry separate true vs. false statements (AUROC)?

研究问题 / Research questions:
    1. 流形马氏距离 (maha_last) 能否把"假陈述"排在"真陈述"之上？ 它打得过困惑度基线 (nll) 吗？
       Does manifold Mahalanobis distance rank false statements above true ones, and beat a
       perplexity baseline?
    2. 双曲离散度 (disp_hse) 是否优于欧氏离散度 (disp_euclid)？ —— 即"双曲几何有没有用"的消融。
       Does hyperbolic dispersion (disp_hse) beat Euclidean dispersion (disp_euclid)? — the
       "does the hyperbolic projection earn its keep?" ablation.

方法 / Method (SAPLMA 式、确定性、无需 LLM 裁判 / SAPLMA-style, deterministic, judge-free):
    - 从**留出**题目的*正确*答案构建真值流形（无标签泄漏）。
      Build the truth manifold from the *correct* answers of **held-out** questions (no leakage).
    - 对其余题目的每条候选答案（正确=负类, 错误=正类/幻觉）做单次前向，提取目标层隐状态。
      For each candidate answer of the remaining questions (correct=negative, incorrect=positive),
      run a single forward pass and read the target-layer hidden states.
    - 每条陈述计算 4 个信号的分数，分别报告 AUROC。
      Score each statement with 4 signals and report AUROC per signal.

局限 / Limitations:
    - 强制候选答案的"陈述级"打分是开放生成幻觉的*代理*，干净地检验表征假设但不等同于在线检测。
      Forced-answer statement scoring is a *proxy* for open-generation hallucination; it cleanly
      tests the representation hypothesis but is not the same as online detection.
    - 句内 token 离散度是基于多次采样的语义熵 (Farquhar et al.) 的廉价代理。
      Within-statement token dispersion is a cheap proxy for sample-based semantic entropy.
    - 小模型 + 几百条样本 → AUROC 置信区间较宽，结果仅供参考，非定论。
      Small model + a few hundred items → wide AUROC CIs; results are indicative, not conclusive.

用法 / Usage:
    # 真实基准 / real benchmark (downloads model + TruthfulQA):
    python benchmarks/eval_truthfulqa.py --model Qwen/Qwen2.5-0.5B-Instruct --layer -8 --limit 200
    # 快速管线自检 / fast pipeline smoke check (tiny model, bundled statements, no dataset):
    python benchmarks/eval_truthfulqa.py --model sshleifer/tiny-gpt2 --offline
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from typing import List, Optional

import torch

from eigentruth.core.math_engine import (
    TruthManifold,
    hyperbolic_semantic_entropy,
    mahalanobis_distance,
    poincare_map,
)
from eigentruth.eval.metrics import euclidean_dispersion, roc_auc

SIGNALS = ["maha_last", "disp_euclid", "disp_hse", "nll_answer"]


@dataclass
class Statement:
    question: str
    answer: str
    is_false: int  # 1 = 错误答案(正类/幻觉) / incorrect (positive), 0 = 正确答案(负类)


# ---------------------------------------------------------------------------
# 离线烟雾集（仅用于验证管线，不构成基准）/ Offline smoke set (pipeline check only)
# ---------------------------------------------------------------------------

_TRUE_SMOKE = [
    "The capital of France is Paris.",
    "Water boils at 100 degrees Celsius at sea level.",
    "The Earth revolves around the Sun.",
    "Humans have two lungs.",
    "The Pacific is the largest ocean on Earth.",
    "Ice is frozen water.",
    "A triangle has three sides.",
    "The Sun rises in the east.",
    "Honey is made by bees.",
    "Mount Everest is the tallest mountain above sea level.",
]
_FALSE_SMOKE = [
    "The capital of France is Berlin.",
    "Water boils at 30 degrees Celsius at sea level.",
    "The Sun revolves around the Earth.",
    "Humans have five lungs.",
    "The Pacific is the smallest ocean on Earth.",
    "Ice is boiling water.",
    "A triangle has seven sides.",
    "The Sun rises in the west.",
    "Honey is made by spiders.",
    "Mount Everest is the shortest mountain on Earth.",
]


def load_offline() -> tuple[List[str], List[Statement]]:
    """返回 (流形构建用真陈述, 评测陈述)。"""
    manifold_texts = _TRUE_SMOKE[:6]
    eval_stmts: List[Statement] = []
    for t in _TRUE_SMOKE[6:]:
        eval_stmts.append(Statement("", t, 0))
    for f in _FALSE_SMOKE[6:]:
        eval_stmts.append(Statement("", f, 1))
    return manifold_texts, eval_stmts


def load_truthfulqa(
    manifold_questions: int, limit: int
) -> tuple[List[str], List[Statement]]:
    """加载 TruthfulQA multiple_choice，切分为流形集 / 评测集（题目层面不重叠）。"""
    from datasets import load_dataset  # lazy

    # 新旧版 datasets 的数据集 id 不同，依次尝试 / dataset id differs across versions
    last_err: Optional[Exception] = None
    ds = None
    for dataset_id in ("truthfulqa/truthful_qa", "truthful_qa"):
        try:
            ds = load_dataset(dataset_id, "multiple_choice")["validation"]
            break
        except Exception as exc:  # noqa: BLE001
            last_err = exc
    if ds is None:
        raise RuntimeError(
            f"Could not load TruthfulQA. Last error: {last_err}. "
            f"Try `pip install -U datasets` or run with --offline for a pipeline check."
        )

    manifold_texts: List[str] = []
    eval_stmts: List[Statement] = []
    for i, row in enumerate(ds):
        q = row["question"]
        targets = row["mc2_targets"]
        choices, labels = targets["choices"], targets["labels"]
        if i < manifold_questions:
            # 仅用正确答案构建流形 / manifold from correct answers only
            for c, lab in zip(choices, labels):
                if lab == 1:
                    manifold_texts.append(f"{q} {c}")
        else:
            for c, lab in zip(choices, labels):
                eval_stmts.append(Statement(q, c, is_false=int(lab == 0)))
        if limit and (i - manifold_questions + 1) >= limit and i >= manifold_questions:
            break
    return manifold_texts, eval_stmts


# ---------------------------------------------------------------------------
# 模型与表征提取 / Model and representation extraction
# ---------------------------------------------------------------------------

_DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


def load_model(model_name: str, device: torch.device, dtype: str = "float32"):
    from transformers import AutoModelForCausalLM, AutoTokenizer  # lazy

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # dtype= 替代已弃用的 torch_dtype=；bfloat16 减半权重内存；
    # low_cpu_mem_usage 避免加载时 2× 峰值内存（低内存机器关键）
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=_DTYPES[dtype], low_cpu_mem_usage=True
    )
    model.to(device).eval()
    return model, tokenizer


@torch.no_grad()
def statement_reps(model, tokenizer, stmt: Statement, layer: int,
                   device: torch.device, max_length: int) -> Optional[dict]:
    """单次前向，返回答案 token 的隐状态、末 token 隐状态、答案 NLL。"""
    q_ids = tokenizer(stmt.question, add_special_tokens=True).input_ids if stmt.question \
        else tokenizer(tokenizer.bos_token or tokenizer.eos_token or " ").input_ids
    a_ids = tokenizer(" " + stmt.answer.strip(), add_special_tokens=False).input_ids
    if len(a_ids) == 0:
        return None
    ids = (q_ids + a_ids)[:max_length]
    n_ans = len(ids) - len(q_ids)
    if n_ans <= 0:
        return None

    input_ids = torch.tensor([ids], device=device)
    out = model(input_ids=input_ids, output_hidden_states=True)

    # hidden_states[-k] 对应 layers[-k] 的输出（负索引对齐，与 wrapper 约定一致）
    hs = out.hidden_states[layer][0]  # [T, D]
    ans_hs = hs[-n_ans:, :].float().cpu()
    last_hs = hs[-1, :].float().cpu()

    # 答案 token 的平均负对数似然（困惑度基线）
    logits = out.logits[0].float()  # [T, V]
    logp = torch.log_softmax(logits[:-1], dim=-1)  # 由位置 t 预测 t+1
    targets = input_ids[0, 1:]
    tok_logp = logp[torch.arange(logp.shape[0]), targets]  # [T-1]
    ans_logp = tok_logp[-n_ans:] if n_ans <= tok_logp.shape[0] else tok_logp
    nll = float((-ans_logp.mean()).item())

    return {"ans_hs": ans_hs, "last_hs": last_hs, "nll": nll}


def build_manifold(model, tokenizer, texts: List[str], layer: int,
                   device: torch.device, max_length: int) -> TruthManifold:
    manifold = TruthManifold()
    for t in texts:
        reps = statement_reps(model, tokenizer, Statement("", t, 0), layer, device, max_length)
        if reps is not None:
            manifold.update(reps["last_hs"])
    return manifold


# ---------------------------------------------------------------------------
# 主流程 / Main
# ---------------------------------------------------------------------------

def run(args) -> dict:
    # Windows 控制台默认 cp1252，确保非 ASCII 不崩溃；行缓冲让进度实时可见（被杀也不丢日志）
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    if args.offline:
        manifold_texts, eval_stmts = load_offline()
        print("[!] OFFLINE SMOKE MODE - pipeline check only, NOT a benchmark.\n")
    else:
        manifold_texts, eval_stmts = load_truthfulqa(args.manifold_questions, args.limit)

    print(f"Loading {args.model} on {device} (dtype={args.dtype}) ...")
    model, tokenizer = load_model(args.model, device, args.dtype)

    print(f"Building truth manifold from {len(manifold_texts)} true statements "
          f"(layer={args.layer}) ...")
    manifold = build_manifold(model, tokenizer, manifold_texts, args.layer, device, args.max_length)
    if not manifold.is_ready():
        print("[X] Manifold not ready (need >=2 statements). Aborting.")
        sys.exit(1)
    print(f"   manifold: n={manifold.n}, hidden_dim={manifold.hidden_dim}\n")

    scores: dict[str, List[float]] = {s: [] for s in SIGNALS}
    labels: List[int] = []

    print(f"Scoring {len(eval_stmts)} eval statements ...")
    for k, stmt in enumerate(eval_stmts):
        reps = statement_reps(model, tokenizer, stmt, args.layer, device, args.max_length)
        if reps is None:
            continue
        last = reps["last_hs"]
        ans = reps["ans_hs"]

        scores["maha_last"].append(
            float(mahalanobis_distance(last, manifold.mean.cpu(), manifold.cov_inv.cpu()).item())
        )
        scores["disp_euclid"].append(float(euclidean_dispersion(ans).item()))
        scores["disp_hse"].append(
            float(hyperbolic_semantic_entropy(poincare_map(ans)).item())
        )
        scores["nll_answer"].append(reps["nll"])
        labels.append(stmt.is_false)

        if (k + 1) % 50 == 0:
            print(f"   {k + 1}/{len(eval_stmts)}")

    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    results = {s: roc_auc(scores[s], labels) for s in SIGNALS}

    # ---- 输出 ----
    print("\n" + "=" * 56)
    print("  AUROC  (positive = false/hallucinated statement)")
    print(f"  model={args.model}  layer={args.layer}  n_pos={n_pos}  n_neg={n_neg}")
    print("=" * 56)
    print(f"  {'signal':<14}{'AUROC':>10}   interpretation")
    print("  " + "-" * 52)
    for s in SIGNALS:
        print(f"  {s:<14}{results[s]:>10.3f}")
    print("  " + "-" * 52)
    # 关键对比 / key comparisons
    if not (results["disp_hse"] != results["disp_hse"]):  # not NaN
        delta = results["disp_hse"] - results["disp_euclid"]
        verdict = "hyperbolic HELPS" if delta > 0.01 else (
            "hyperbolic HURTS" if delta < -0.01 else "no meaningful difference")
        print(f"  disp_hse - disp_euclid = {delta:+.3f}  ->  {verdict}")
    geo = max(results["maha_last"], results["disp_hse"], results["disp_euclid"])
    if not (results["nll_answer"] != results["nll_answer"]):
        print(f"  best geometry ({geo:.3f}) vs nll baseline ({results['nll_answer']:.3f})  ->  "
              f"{'geometry wins' if geo > results['nll_answer'] + 0.01 else 'baseline competitive'}")
    print("=" * 56)

    payload = {
        "config": {"model": args.model, "layer": args.layer, "offline": args.offline,
                   "manifold_n": manifold.n, "hidden_dim": manifold.hidden_dim,
                   "n_pos": n_pos, "n_neg": n_neg, "seed": args.seed},
        "auroc": results,
    }
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\nWrote structured results to {args.json}")
    print("\nJSON:", json.dumps(payload["auroc"]))
    return payload


def main():
    p = argparse.ArgumentParser(description="EigenTruth TruthfulQA AUROC benchmark")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--dtype", default="float32", choices=list(_DTYPES),
                   help="model weight dtype; bfloat16 halves memory on low-RAM machines")
    p.add_argument("--layer", type=int, default=-8, help="target layer index (negative ok)")
    p.add_argument("--limit", type=int, default=200, help="max eval questions (0 = all)")
    p.add_argument("--manifold-questions", type=int, default=80,
                   help="held-out questions whose correct answers build the manifold")
    p.add_argument("--max-length", type=int, default=64)
    p.add_argument("--offline", action="store_true",
                   help="use bundled smoke statements (pipeline check, not a benchmark)")
    p.add_argument("--json", default=None, help="optional path to write structured results")
    p.add_argument("--seed", type=int, default=0)
    run(p.parse_args())


if __name__ == "__main__":
    main()
