"""Paired McNemar comparison between two eval_suite.sh runs on the same tasks.

Two point estimates with overlapping confidence intervals don't tell you
whether a difference is real - McNemar does, because it pairs by doc_id and
asks specifically about the items the two models disagree on (discordant
pairs), which is the only quantity that actually carries information about
a difference. Reports the discordant count first, same as
scripts/eval/mcnemar_compare.py's convention, so an underpowered comparison
is visible as underpowered rather than silently reported as a confident null.

Usage:
    python mcnemar_eval_suite.py BASELINE_OUT CANDIDATE_OUT [TASK ...]

BASELINE_OUT/CANDIDATE_OUT are the --output_path roots eval_suite.sh was run
with (i.e. contain <task>/<model_dir>/samples_<task>_*.jsonl). Defaults to
the three eval_suite.sh tasks if none given.
"""

from __future__ import annotations

import glob
import json
import math
import sys

METRIC_KEYS = {
    "humaneval_instruct": ["pass@1,create_test", "pass@1"],
    "humaneval_plus_instruct": ["pass@1,create_test", "pass@1"],
    "mbpp_plus_instruct": ["pass_at_1,extract_code", "pass_at_1"],
}
DEFAULT_TASKS = list(METRIC_KEYS)


def load(root: str, task: str, metric_keys: list[str]) -> dict[int, tuple[str | None, float]]:
    files = glob.glob(f"{root}/{task}/**/samples_*.jsonl", recursive=True)
    if not files:
        raise SystemExit(f"no samples under {root}/{task}")
    out: dict[int, tuple[str | None, float]] = {}
    with open(files[0]) as f:
        for line in f:
            d = json.loads(line)
            key = next((mk for mk in metric_keys if mk in d), None)
            if key is None:
                raise SystemExit(f"no metric key found in {list(d.keys())}")
            out[d["doc_id"]] = (d.get("doc_hash"), float(d[key]))
    return out


def mcnemar(baseline_root: str, candidate_root: str, task: str) -> None:
    metric_keys = METRIC_KEYS[task]
    a = load(baseline_root, task, metric_keys)
    b = load(candidate_root, task, metric_keys)
    ids = sorted(set(a) & set(b))
    if len(ids) != len(a) or len(ids) != len(b):
        print(f"  !! doc_id set mismatch: baseline={len(a)} candidate={len(b)} common={len(ids)}")

    mismatched_hash = 0
    n01 = n10 = n11 = n00 = 0
    for i in ids:
        ha, ra = a[i]
        hb, rb = b[i]
        if ha != hb:
            mismatched_hash += 1
        ca, cb = ra >= 0.5, rb >= 0.5
        if ca and cb:
            n11 += 1
        elif ca and not cb:
            n10 += 1
        elif not ca and cb:
            n01 += 1
        else:
            n00 += 1

    n_disc = n01 + n10
    print(f"=== {task} (n={len(ids)}) ===")
    if mismatched_hash:
        print(f"  !! {mismatched_hash} doc_hash mismatches - pairing may be invalid")
    print(f"  baseline pass rate:  {(n11 + n10) / len(ids):.4f}")
    print(f"  candidate pass rate: {(n11 + n01) / len(ids):.4f}")
    print(f"  discordant pairs: {n_disc}  (baseline-only correct: {n10}, candidate-only correct: {n01})")
    if n_disc == 0:
        print("  no discordant pairs - identical outcomes on every item")
        return
    chi2 = (abs(n10 - n01) - 1) ** 2 / n_disc  # continuity-corrected
    p = math.erfc(math.sqrt(chi2 / 2))  # chi-square survival, 1 dof, no scipy needed
    print(f"  McNemar chi2={chi2:.3f}  p={p:.4f}")
    print(f"  {'SIGNIFICANT at alpha=0.05' if p < 0.05 else 'NOT significant at alpha=0.05 - could be noise'}")
    if n_disc < 40:
        print(
            f"  NOTE: only {n_disc} discordant pairs (this project's convention flags "
            "~40-50 as needed for 80% power at alpha=0.05) - a null result here is "
            "not strong evidence of equivalence"
        )


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    baseline_root, candidate_root = sys.argv[1], sys.argv[2]
    tasks = sys.argv[3:] or DEFAULT_TASKS
    for i, t in enumerate(tasks):
        if i:
            print()
        mcnemar(baseline_root, candidate_root, t)
