#!/usr/bin/env python3
"""Collect one PAC-CSE case through a model adapter.

The adapter must implement ``generate(image, question, sample, temperature,
top_p, max_new_tokens)`` and return one decoded answer. This keeps model-
specific processor code outside the method while fixing the PAC-CSE protocol.
"""
from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter

from protocol import ARM_SIZE, MODEL_SAMPLING, PAC_MAX_NEW_TOKENS, PERTURBATIONS

SEEDS = tuple(131 + 977 * i for i in range(ARM_SIZE))


def perturb(image, name, value, rng):
    if name.startswith("gauss_noise"):
        x = np.asarray(image, dtype=np.float32) / 255.0
        return Image.fromarray(np.uint8(np.clip(x + rng.normal(0, value, x.shape), 0, 1) * 255))
    if name.startswith("gauss_blur"):
        return image.filter(ImageFilter.GaussianBlur(value))
    if name.startswith("contrast"):
        x = np.asarray(image, dtype=np.float32) / 255.0
        mean = x.mean(axis=(0, 1), keepdims=True)
        return Image.fromarray(np.uint8(np.clip((x - mean) * value + mean, 0, 1) * 255))
    if name.startswith("jpeg"):
        import io
        buf = io.BytesIO(); image.save(buf, "JPEG", quality=int(value)); buf.seek(0)
        return Image.open(buf).convert("RGB")
    width, height = image.size
    small = image.resize((max(1, int(width * value)), max(1, int(height * value))), Image.Resampling.BOX)
    return small.resize((width, height), Image.Resampling.BOX)


def _sample(adapter, image, question, model_name, seed, sample=True):
    temperature, top_p = MODEL_SAMPLING[model_name]
    return str(adapter.generate(image, question, sample=sample, temperature=temperature,
                                top_p=top_p, max_new_tokens=PAC_MAX_NEW_TOKENS, seed=seed)).strip()


def _name(adapter):
    value = getattr(adapter, "model_name", getattr(adapter, "name", ""))
    aliases = {"HuluMed": "Hulu-Med", "Chexagent": "CheXagent", "Medgemma": "MedGemma"}
    return aliases.get(value, value)


def collect_case(target, peer, image, question, case_index, output_dir):
    """Write ``clean``, ``perturbed``, and ``peer`` NPZ files for one case."""
    image = image.convert("RGB"); output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    target_name, peer_name = _name(target), _name(peer)
    greedy = _sample(target, image, question, target_name, 0, sample=False)
    clean = [_sample(target, image, question, target_name, seed) for seed in SEEDS]
    rng = np.random.default_rng(1000 + int(case_index))
    perturbed, names = [], []
    for name, family, value in PERTURBATIONS:
        perturbed.append(_sample(target, perturb(image, name, value, rng), question, target_name, SEEDS[len(perturbed)]))
        names.append(name)
    peer_answers = [_sample(peer, image, question, peer_name, seed) for seed in SEEDS]
    np.savez(output_dir / f"{case_index}_clean.npz", answers=np.asarray([greedy] + clean, dtype=object),
             conditions=np.asarray(["greedy"] + ["original"] * ARM_SIZE), question=question,
             reference_answer=greedy)
    np.savez(output_dir / f"{case_index}_perturbed.npz", answers=np.asarray(perturbed, dtype=object),
             perturbation=np.asarray(names), question=question)
    np.savez(output_dir / f"{case_index}_peer.npz", answers=np.asarray(peer_answers, dtype=object),
             question=question)
