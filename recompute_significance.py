#!/usr/bin/env python3
"""Paired DeLong tests and paired case-bootstrap confidence intervals."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from scipy.stats import norm
from sklearn.metrics import average_precision_score, auc


def ranks(x):
    order = np.argsort(x); out = np.empty(len(x), float); i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[order[j]] == x[order[i]]: j += 1
        out[order[i:j]] = (i + j + 1) / 2; i = j
    return out


def delong(y, a, b):
    order = np.argsort(-y); y = y[order]; z = np.vstack((a[order], b[order]))
    m, n = int(y.sum()), len(y) - int(y.sum())
    all_ranks = np.vstack([ranks(row) for row in z])
    pos, neg = z[:, :m], z[:, m:]
    v01 = (all_ranks[:, :m] - np.vstack([ranks(row) for row in pos])) / n
    v10 = 1 - (all_ranks[:, m:] - np.vstack([ranks(row) for row in neg])) / m
    aucs = (all_ranks[:, :m].sum(1) - m * (m + 1) / 2) / (m * n)
    cov = np.cov(v01) / m + np.cov(v10) / n
    variance = max(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1], 1e-15)
    p = 2 * norm.sf(abs((aucs[0] - aucs[1]) / np.sqrt(variance)))
    return float(aucs[0] - aucs[1]), float(p)


def aurac(y, score):
    order = np.argsort(score); retained = (1 - y[order]); c = np.linspace(.02, 1, 101)
    return auc(c, [retained[:max(1, int(v * len(y)))].mean() for v in c])


def bootstrap(y, a, b, rng, count):
    n = len(y); da, dr = [], []
    for _ in range(count):
        ix = rng.integers(0, n, n); yy = y[ix]
        da.append(average_precision_score(yy, a[ix]) - average_precision_score(yy, b[ix]))
        dr.append(aurac(yy, a[ix]) - aurac(yy, b[ix]))
    return np.quantile(da, [.025, .975]), np.quantile(dr, [.025, .975])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--pac", type=Path, required=True)
    ap.add_argument("--baseline-dir", type=Path, required=True); ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--bootstrap", type=int, default=10000); args = ap.parse_args()
    data = json.loads(args.pac.read_text())
    names = {"VQA-RAD/Hulu-Med":"hulumed_vqarad.csv", "VQA-RAD/Lingshu":"lingshu_vqarad.csv",
             "VQA-RAD/CheXagent":"chexagent_vqarad.csv", "SLAKE/Hulu-Med":"hulumed_slake.csv",
             "SLAKE/Lingshu":"lingshu_slake.csv", "SLAKE/CheXagent":"chexagent_slake.csv",
             "SLAKE/MedGemma":"medgemma_slake.csv"}
    rng = np.random.default_rng(20260829); rows = []
    for cell, filename in names.items():
        d = data[cell]; y, pac = np.asarray(d["y"], int), np.asarray(d["PAC"], float)
        csv_rows = list(csv.DictReader((args.baseline_dir / filename).open()))
        if [int(r["img_idx"]) for r in csv_rows] != d["idx"]: raise ValueError(cell)
        for baseline in ("SE", "VASE", "RadFlag"):
            base = np.asarray([float(r[baseline]) for r in csv_rows], float)
            if baseline == "RadFlag": base = -base
            delta, p = delong(y, pac, base); ap_ci, ar_ci = bootstrap(y, pac, base, rng, args.bootstrap)
            rows.append({"cell":cell, "baseline":baseline, "delta_auroc":100*delta, "raw_p":p,
                         "aupr_ci":(100*ap_ci).tolist(), "aurac_ci":(100*ar_ci).tolist(), "n":len(y)})
    for rank, i in enumerate(np.argsort([r["raw_p"] for r in rows])):
        previous = rows[i - 1].get("holm_p", 0) if rank else 0
        rows[i]["holm_p"] = max(previous, min(1, (len(rows) - rank) * rows[i]["raw_p"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"seed":20260829, "bootstrap":args.bootstrap, "rows":rows}, indent=2) + "\n")


if __name__ == "__main__": main()
