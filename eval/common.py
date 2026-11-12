"""Shared paths, dataset loading and evaluation protocol helpers.

The evaluation mirrors how SnapClass is used in practice:

* every registered student is enrolled from ONE photo (one-shot enrollment), and
* the system must also reject people who were never registered (open-set).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

EVAL_DIR = Path(__file__).resolve().parent
DATA_DIR = EVAL_DIR / "data"
GT_DIR = DATA_DIR / "gt_db"
LABEL_DIR = DATA_DIR / "labels"
GROUP_DIR = DATA_DIR / "group_photos"
CACHE_DIR = EVAL_DIR / "cache"
RESULTS_DIR = EVAL_DIR / "results"

N_SUBJECTS = 50
N_IMAGES = 15

# Protocol constants
SEEDS = [0, 1, 2, 3, 4]
KNOWN_PER_HALF = 18      # registered students in each identity half
UNKNOWN_PER_HALF = 7     # never-registered people in each identity half
TARGET_UNKNOWN_FAR = 0.01  # threshold is tuned so <=1% of strangers are accepted


@dataclass(frozen=True)
class Image:
    image_id: str      # e.g. "s07/12"
    subject: str       # e.g. "s07"
    path: Path
    box: tuple[int, int, int, int]  # manual face box: left, top, right, bottom


# The dataset contains two pixel-identical copies (s19/14 == s19/08, s19/15 == s19/11).
# The copies are dropped so an enrollment photo can never be probed with itself.
DUPLICATE_IMAGES = {"s19/14", "s19/15"}


def list_images() -> list[Image]:
    """The 748 unique Georgia Tech images with their manually labelled face boxes."""
    images = []
    for k in range(1, N_SUBJECTS + 1):
        subject = f"s{k:02d}"
        for j in range(1, N_IMAGES + 1):
            if f"{subject}/{j:02d}" in DUPLICATE_IMAGES:
                continue
            label_file = LABEL_DIR / f"lab{(k - 1) * 20 + j:03d}"
            left, top, right, bottom = map(int, label_file.read_text().split()[:4])
            images.append(Image(
                image_id=f"{subject}/{j:02d}",
                subject=subject,
                path=GT_DIR / subject / f"{j:02d}.jpg",
                box=(left, top, right, bottom),
            ))
    return images


def subjects() -> list[str]:
    return [f"s{k:02d}" for k in range(1, N_SUBJECTS + 1)]


def box_center_inside(det_box, gt_box) -> bool:
    """A detection counts as the labelled face if its centre lies inside the manual box."""
    cx = (det_box[0] + det_box[2]) / 2
    cy = (det_box[1] + det_box[3]) / 2
    return gt_box[0] <= cx <= gt_box[2] and gt_box[1] <= cy <= gt_box[3]


def identity_halves(seed: int) -> tuple[list[str], list[str]]:
    """Split the 50 people into two disjoint halves of 25.

    Thresholds are tuned on one half and tested on the other, so the reported
    numbers never come from people the threshold was tuned on.
    """
    rng = np.random.default_rng(seed)
    order = list(rng.permutation(subjects()))
    return sorted(order[:25]), sorted(order[25:])


def make_gallery(half: list[str], seed: int, candidates: dict[str, list[str]]):
    """Pick registered vs unregistered people and one enrollment photo per registered person.

    `candidates` maps subject -> image ids usable for enrollment (a face was
    detected by every compared model, so all models enroll from the same photo).
    """
    rng = np.random.default_rng(seed + 1000)
    order = list(rng.permutation(half))
    known = sorted(order[:KNOWN_PER_HALF])
    unknown = sorted(order[KNOWN_PER_HALF:KNOWN_PER_HALF + UNKNOWN_PER_HALF])
    enroll = {s: str(rng.choice(sorted(candidates[s]))) for s in known}
    return known, unknown, enroll


def load_cache(model_name: str, phase: str = "singles") -> dict:
    import pickle
    with open(CACHE_DIR / f"{model_name}_{phase}.pkl", "rb") as f:
        return pickle.load(f)


def enrollment_candidates(model_names: list[str]) -> dict[str, list[str]]:
    """Per person, the photos in which EVERY compared model found the face.

    Enrollment photos are drawn from this set so that all models enroll each
    student from the same photo, as a registration that failed would simply be retaken.
    """
    caches = [load_cache(m) for m in model_names]
    out = {s: [] for s in subjects()}
    for img in list_images():
        if all(c[img.image_id]["embedding"] is not None for c in caches):
            out[img.subject].append(img.image_id)
    return out


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


def load_json(path: Path):
    return json.loads(path.read_text())
