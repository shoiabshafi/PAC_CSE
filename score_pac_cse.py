#!/usr/bin/env python3
"""Score PAC-CSE from saved clean, perturbed, and peer NPZ arms."""
import argparse
import json
from pathlib import Path

import numpy as np

from protocol import ARM_SIZE, JEFFREYS, LAMBDA_P, LAMBDA_X, PERTURBATIONS


def entropy(p):
    p = np.asarray(p, float)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def load_arm(path, condition="original"):
    with np.load(path, allow_pickle=True) as z:
        answers = np.asarray([str(x) for x in z["answers"]])
        greedy = None
        if "conditions" in z.files:
            cond = np.asarray([str(x) for x in z["conditions"]])
            if (cond == "greedy").any():
                greedy = str(answers[cond == "greedy"][0])
            keep = cond == condition
            if keep.any():
                answers = answers[keep]
        if greedy is None and "reference_answer" in z.files:
            greedy = str(z["reference_answer"].item())
        question = str(z["question"].item()) if "question" in z.files else ""
    return list(answers[:ARM_SIZE]), greedy, question


class NLIClusterer:
    def __init__(self, model_name="microsoft/deberta-large-mnli", batch=64):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device).eval()
        labels = {int(k): str(v).lower() for k, v in self.model.config.id2label.items()}
        self.entail = next(k for k, v in labels.items() if "entail" in v)
        self.batch = batch
        self.cache = {}

    def _probabilities(self, pairs):
        out = []
        for start in range(0, len(pairs), self.batch):
            left, right = zip(*pairs[start:start + self.batch])
            batch = self.tokenizer(list(left), list(right), return_tensors="pt",
                                   padding=True, truncation=True, max_length=256)
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with self.torch.inference_mode():
                out.append(self.model(**batch).logits.softmax(-1).cpu().numpy())
        return np.concatenate(out) if out else np.empty((0, 3))

    def cluster(self, answers, question):
        reps, labels = [], []
        prefix = f"Question: {question}\n" if question else ""
        for answer in answers:
            current = prefix + "Answer: " + answer
            pairs = []
            for rep in reps:
                other = prefix + "Answer: " + rep
                pairs.extend(((current, other), (other, current)))
            probs = self._probabilities(pairs).argmax(-1) if pairs else []
            hit = None
            for i in range(len(reps)):
                if probs[2 * i] == self.entail and probs[2 * i + 1] == self.entail:
                    hit = i
                    break
            if hit is None:
                reps.append(answer)
                hit = len(reps) - 1
            labels.append(hit)
        return labels


def pac_score(clean, perturbed, peer, labels):
    support = sorted(set(labels))
    n = np.array([labels[:ARM_SIZE].count(c) for c in support], float)
    m = np.array([labels[ARM_SIZE:2 * ARM_SIZE].count(c) for c in support], float)
    r = np.array([labels[2 * ARM_SIZE:3 * ARM_SIZE].count(c) for c in support], float)
    po, pp = n / n.sum(), r / r.sum()
    midpoint = (po + pp) / 2
    js = 0.5 * np.sum(po * np.log(po / midpoint)) + 0.5 * np.sum(pp * np.log(pp / midpoint))
    pi = n + LAMBDA_X * m + LAMBDA_P * r + JEFFREYS
    pi /= pi.sum()
    return entropy(pi) + float(js)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean-dir", type=Path, required=True)
    ap.add_argument("--perturbed-dir", type=Path, required=True)
    ap.add_argument("--peer-dir", type=Path, required=True)
    ap.add_argument("--labels", type=Path, help="JSON index -> {hallucinated: bool}")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--model", default="microsoft/deberta-large-mnli")
    args = ap.parse_args()
    label_map = json.loads(args.labels.read_text()) if args.labels else {}
    clusterer = NLIClusterer(args.model)
    scores = {}
    clean_files = list(args.clean_dir.glob("*_clean.npz")) or list(args.clean_dir.glob("*.npz"))
    for clean_path in sorted(clean_files, key=lambda p: int(p.stem.split("_")[0])):
        idx = clean_path.stem.removesuffix("_clean")
        pert_path = args.perturbed_dir / f"{idx}_perturbed.npz"
        peer_path = args.peer_dir / f"{idx}_peer.npz"
        if not pert_path.exists(): pert_path = args.perturbed_dir / f"{idx}.npz"
        if not peer_path.exists(): peer_path = args.peer_dir / f"{idx}.npz"
        if not pert_path.exists() or not peer_path.exists():
            continue
        clean, greedy, question = load_arm(clean_path)
        perturbed, _, _ = load_arm(pert_path)
        peer, _, _ = load_arm(peer_path)
        labels = clusterer.cluster(clean + perturbed + peer, question)
        rec = {"PAC": pac_score(clean, perturbed, peer, labels), "n_clean": len(clean),
               "n_perturbed": len(perturbed), "n_peer": len(peer)}
        if idx in label_map:
            value = label_map[idx]
            rec["hallucinated"] = bool(value.get("hallucinated", value) if isinstance(value, dict) else value)
        scores[idx] = rec
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(scores, indent=2) + "\n")


if __name__ == "__main__":
    main()
