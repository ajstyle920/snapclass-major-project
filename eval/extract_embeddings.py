"""Run each face model once over every image and cache the detections and embeddings.

All metrics are computed later from these caches, so each model sees each photo
exactly once and every experiment uses identical embeddings.

    python eval/extract_embeddings.py --phase singles        # 748 single-person photos
    python eval/extract_embeddings.py --phase groups         # composite classroom photos
"""
from __future__ import annotations

import argparse
import pickle
import time

import numpy as np
from PIL import Image as PILImage

from common import CACHE_DIR, GROUP_DIR, box_center_inside, list_images, load_json
from models import RECOGNITION_MODELS, get_model


def load_rgb(path) -> np.ndarray:
    return np.array(PILImage.open(path).convert("RGB"))


def run_singles(model_name: str):
    model = get_model(model_name)
    model(load_rgb(list_images()[0].path))  # warm-up so timing excludes one-off setup
    records = {}
    images = list_images()
    for i, img in enumerate(images):
        rgb = load_rgb(img.path)
        t0 = time.perf_counter()
        faces = model(rgb)
        elapsed = time.perf_counter() - t0
        # keep the detection that corresponds to the manually labelled face
        matches = [f for f in faces if box_center_inside(f.box, img.box)]
        best = max(matches, key=lambda f: (f.box[2] - f.box[0]) * (f.box[3] - f.box[1]), default=None)
        records[img.image_id] = {
            "n_faces": len(faces),
            "embedding": None if best is None else best.embedding,
            "box": None if best is None else best.box,
            "seconds": elapsed,
        }
        if (i + 1) % 150 == 0:
            print(f"  {model_name}: {i + 1}/{len(images)}", flush=True)
    return records


def run_groups(model_name: str):
    meta = load_json(GROUP_DIR / "groups.json")
    model = get_model(model_name)
    records = {}
    for g in meta["photos"]:
        rgb = load_rgb(GROUP_DIR / g["file"])
        t0 = time.perf_counter()
        faces = model(rgb)
        records[g["file"]] = {
            "faces": [(f.box, f.embedding) for f in faces],
            "seconds": time.perf_counter() - t0,
        }
        print(f"  {model_name}: {g['file']} -> {len(faces)} faces in {records[g['file']]['seconds']:.1f}s", flush=True)
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["singles", "groups"], required=True)
    parser.add_argument("--models", nargs="+", default=None)
    args = parser.parse_args()

    default = RECOGNITION_MODELS + (["dlib_cnn"] if args.phase == "groups" else [])
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for name in args.models or default:
        out = CACHE_DIR / f"{name}_{args.phase}.pkl"
        if out.exists():
            print(f"✓ {out.name} exists, skipping")
            continue
        print(f"▶ {name} ({args.phase})")
        records = run_singles(name) if args.phase == "singles" else run_groups(name)
        with open(out, "wb") as f:
            pickle.dump(records, f)
        secs = np.median([r["seconds"] for r in records.values()])
        print(f"✓ saved {out.name}  (median {secs:.3f}s per image)")


if __name__ == "__main__":
    main()
