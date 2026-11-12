"""Build composite "classroom" photos from the single-person dataset photos.

The Georgia Tech database has one person per photo, but SnapClass is used on
photos of a whole class. To test that case, this script tiles several people's
photos into one 1280x960 image:

* front rows: 4 x 4 grid, 16 people, faces about 75 px wide
* back rows:  6 x 6 grid, 36 people, faces about 50 px wide

Each photo mixes registered students and unregistered people, and never reuses a
student's enrollment photo. These are synthetic composites, not real classroom
photos: they test small faces and many faces per image, not real crowd effects
such as occlusion or people turning away.

    python eval/build_group_photos.py
"""
import numpy as np
from PIL import Image as PILImage

from common import GROUP_DIR, enrollment_candidates, list_images, save_json, subjects
from models import RECOGNITION_MODELS

SEED = 2026
CANVAS = (1280, 960)
LAYOUTS = {"front_rows": (4, 4), "back_rows": (6, 6)}
PHOTOS_PER_LAYOUT = 10
N_KNOWN = 35  # registered students in the class; the other 15 people are strangers


def main():
    rng = np.random.default_rng(SEED)
    candidates = enrollment_candidates(RECOGNITION_MODELS)
    people = list(rng.permutation(subjects()))
    known, unknown = sorted(people[:N_KNOWN]), sorted(people[N_KNOWN:])
    enroll = {s: str(rng.choice(sorted(candidates[s]))) for s in known}

    by_subject = {}
    for img in list_images():
        by_subject.setdefault(img.subject, []).append(img)

    GROUP_DIR.mkdir(parents=True, exist_ok=True)
    photos = []
    for layout, (cols, rows) in LAYOUTS.items():
        tile_w, tile_h = CANVAS[0] // cols, CANVAS[1] // rows
        for n in range(PHOTOS_PER_LAYOUT):
            canvas = PILImage.new("RGB", CANVAS, (40, 40, 40))
            chosen = rng.choice(subjects(), size=cols * rows, replace=False)
            tiles = []
            for idx, subject in enumerate(chosen):
                options = [i for i in by_subject[subject] if i.image_id != enroll.get(subject)]
                img = options[rng.integers(len(options))]
                photo = PILImage.open(img.path).convert("RGB")
                sx, sy = tile_w / photo.width, tile_h / photo.height
                x0, y0 = (idx % cols) * tile_w, (idx // cols) * tile_h
                canvas.paste(photo.resize((tile_w, tile_h), PILImage.LANCZOS), (x0, y0))
                l, t, r, b = img.box
                tiles.append({
                    "subject": str(subject),
                    "image_id": img.image_id,
                    "box": [x0 + l * sx, y0 + t * sy, x0 + r * sx, y0 + b * sy],
                })
            name = f"{layout}_{n + 1:02d}.jpg"
            canvas.save(GROUP_DIR / name, quality=92)
            photos.append({"file": name, "layout": layout, "tiles": tiles})
            print(f"✓ {name}: {len(tiles)} people")

    save_json({"seed": SEED, "known": known, "unknown": unknown, "enroll": enroll, "photos": photos},
              GROUP_DIR / "groups.json")
    print(f"\n{len(photos)} group photos, {len(known)} registered / {len(unknown)} unregistered people")


if __name__ == "__main__":
    main()
