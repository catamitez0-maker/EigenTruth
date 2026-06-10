"""E1 — 共形校准覆盖率验证 / Split-conformal coverage validation.

研究问题 / Research question:
    split conformal 能否把原始分数（马氏距离 / 对比方向投影）变成有保证的报警阈值，
    取代拍脑袋的固定阈值？
    Can split conformal turn raw scores into alarm thresholds with honest coverage,
    replacing hand-picked thresholds?

方法 / Method:
    消费 eval_truthfulqa.py --dump-scores 的逐陈述分数（无需再跑模型）。
    把"真陈述"作为可交换的正常总体，随机对半切成 校准/测试，多次重复取平均：
    - 误报率 / false-alarm rate: 真陈述中 score > threshold(alpha) 的比例，应 <= alpha (+3%)
    - 检出率 / detection rate:   假陈述中 score > threshold(alpha) 的比例（power，仅报告）
    Uses per-statement scores dumped by eval_truthfulqa.py. True statements form the
    exchangeable "normal" population, split 50/50 into calibration/test over multiple
    seeded repeats. Gate: |false-alarm − alpha| within tolerance at every alpha.

判据 (E1 gate): 在 alpha ∈ {0.05, 0.1, 0.2} 上 |经验误报率 − alpha| <= 0.03。

用法 / Usage:
    python benchmarks/eval_conformal.py --scores benchmarks/scores_gpt2_l-8.json --signal maha_last
"""

from __future__ import annotations

import argparse
import json
import sys

import torch

from eigentruth.eval.conformal import conformal_threshold

ALPHAS = (0.05, 0.10, 0.20)
TOLERANCE = 0.03


def run(args) -> dict:
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass

    with open(args.scores, encoding="utf-8") as f:
        dump = json.load(f)
    labels = torch.tensor(dump["labels"])
    scores = torch.tensor(dump["scores"][args.signal], dtype=torch.float64)

    true_scores = scores[labels == 0]   # 正常总体（可交换假设的对象）
    false_scores = scores[labels == 1]  # 希望被报警的对象（仅报告 power）
    n_true, n_false = true_scores.numel(), false_scores.numel()
    print(f"signal={args.signal}  n_true={n_true}  n_false={n_false}  "
          f"repeats={args.repeats}\n")

    fa_sum = {a: 0.0 for a in ALPHAS}
    det_sum = {a: 0.0 for a in ALPHAS}
    for r in range(args.repeats):
        g = torch.Generator().manual_seed(args.seed + r)
        perm = torch.randperm(n_true, generator=g)
        half = n_true // 2
        calib = true_scores[perm[:half]]
        test_true = true_scores[perm[half:]]
        for a in ALPHAS:
            t = conformal_threshold(calib, a)
            fa_sum[a] += (test_true > t).double().mean().item()
            det_sum[a] += (false_scores > t).double().mean().item()

    print(f"  {'alpha':>6} {'nominal_cov':>12} {'false_alarm':>12} "
          f"{'emp_cov':>9} {'detect':>8}   gate(|fa-a|<={TOLERANCE})")
    print("  " + "-" * 66)
    results = {}
    all_pass = True
    for a in ALPHAS:
        fa = fa_sum[a] / args.repeats
        det = det_sum[a] / args.repeats
        ok = abs(fa - a) <= TOLERANCE
        all_pass &= ok
        results[str(a)] = {"false_alarm": fa, "coverage": 1.0 - fa, "detection": det,
                           "pass": ok}
        print(f"  {a:>6.2f} {1 - a:>12.2f} {fa:>12.3f} {1 - fa:>9.3f} "
              f"{det:>8.3f}   {'PASS' if ok else 'FAIL'}")
    print("  " + "-" * 66)
    print(f"\n  E1 verdict: {'ACCEPT' if all_pass else 'REJECT'} "
          f"(coverage tracks nominal within {TOLERANCE} at all alphas)"
          if all_pass else
          f"\n  E1 verdict: REJECT (coverage deviates more than {TOLERANCE})")

    payload = {"config": {"scores": args.scores, "signal": args.signal,
                          "repeats": args.repeats, "seed": args.seed,
                          "n_true": n_true, "n_false": n_false},
               "results": results, "verdict": "ACCEPT" if all_pass else "REJECT"}
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\nWrote {args.json}")
    return payload


def main():
    p = argparse.ArgumentParser(description="E1: split-conformal coverage validation")
    p.add_argument("--scores", required=True,
                   help="scores JSON from eval_truthfulqa.py --dump-scores")
    p.add_argument("--signal", default="maha_last",
                   help="which signal to calibrate (maha_last / truth_proj / ...)")
    p.add_argument("--repeats", type=int, default=20, help="number of seeded 50/50 splits")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json", default=None, help="optional path for structured results")
    run(p.parse_args())


if __name__ == "__main__":
    main()
