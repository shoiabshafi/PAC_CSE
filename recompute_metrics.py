#!/usr/bin/env python3
"""Recompute AUROC, AUPR, and AURAC from a PAC-CSE per-case JSON file."""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import auc, average_precision_score, roc_auc_score


def aurac(y, score):
    order = np.argsort(score)
    retained = (1 - np.asarray(y, int))[order]
    coverage = np.linspace(0.02, 1.0, 101)
    accuracy = [retained[:max(1, int(c * len(y)))].mean() for c in coverage]
    return 100 * auc(coverage, accuracy)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-case", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    data = json.loads(args.per_case.read_text())
    out = {}
    for cell, rec in data.items():
        y, op, score = map(np.asarray, (rec["y"], rec["open"], rec["PAC"]))
        out[cell] = {
            "all_auroc": 100 * roc_auc_score(y, score),
            "open_auroc": 100 * roc_auc_score(y[op.astype(bool)], score[op.astype(bool)]),
            "aupr": 100 * average_precision_score(y, score),
            "aurac": aurac(y, score), "n": int(len(y)),
            "open_n": int(op.sum()),
        }
    means = {key: float(np.mean([r[key] for r in out.values()]))
             for key in ("all_auroc", "open_auroc", "aupr", "aurac")}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"per_setting": out, "means": means}, indent=2) + "\n")


if __name__ == "__main__":
    main()
