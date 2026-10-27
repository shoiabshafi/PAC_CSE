#!/usr/bin/env python3
"""Validate the response-arm layout used by score_pac_cse.py."""
import argparse
from pathlib import Path
import numpy as np
from protocol import ARM_SIZE, PERTURBATIONS


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", type=Path, required=True); args = ap.parse_args()
    expected = [p[0] for p in PERTURBATIONS]; errors = []
    for clean in sorted(args.root.glob("*_clean.npz")):
        stem = clean.name[:-10]; pert = args.root / f"{stem}_perturbed.npz"; peer = args.root / f"{stem}_peer.npz"
        if not pert.exists() or not peer.exists(): errors.append(stem); continue
        with np.load(clean, allow_pickle=True) as c, np.load(pert, allow_pickle=True) as x, np.load(peer, allow_pickle=True) as p:
            cond = [str(v) for v in c["conditions"]]; names = [str(v) for v in x["perturbation"]]
            valid = (cond.count("greedy") == 1 and cond.count("original") == ARM_SIZE
                     and len(p["answers"]) == ARM_SIZE and names == expected)
            if not valid: errors.append(stem)
    if errors: raise SystemExit(f"invalid cases: {len(errors)}")
    print(f"validated {len(list(args.root.glob('*_clean.npz')))} cases")


if __name__ == "__main__": main()
