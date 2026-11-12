"""Download the Georgia Tech Face Database and its face-position labels into eval/data/.

Dataset: 50 people x 15 colour photos (640x480), recorded in 1999 at the Center for
Signal and Image Processing, Georgia Institute of Technology, and published by
Ara V. Nefian for research use: https://www.anefian.com/research/face_reco.htm

The images are NOT committed to this repository; run this script to fetch them.

    python eval/download_data.py

If the server answers with a web page instead of a file (it sometimes shows a
"please wait" check page to scripts), download the two archives in a browser from
https://www.anefian.com/research/gt_db.zip and
https://www.anefian.com/research/labels_gt.zip, put them in eval/data/, and run
this script again: it will only extract them.
"""
import shutil
import sys
import urllib.request
import zipfile

from common import DATA_DIR, GT_DIR, LABEL_DIR

BASE = "https://www.anefian.com/research/"
FILES = {"gt_db.zip": GT_DIR, "labels_gt.zip": LABEL_DIR}


def download(name: str):
    target = DATA_DIR / name
    if target.exists() and zipfile.is_zipfile(target):
        print(f"✓ {name} already downloaded")
        return target
    print(f"↓ downloading {name} ...")
    with urllib.request.urlopen(BASE + name) as response, open(target, "wb") as f:
        shutil.copyfileobj(response, f)
    if not zipfile.is_zipfile(target):
        target.unlink()
        sys.exit(
            f"✗ The server returned a web page instead of {name}.\n"
            f"  Download {BASE + name} in a browser, save it to {DATA_DIR}/, "
            "and run this script again."
        )
    return target


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name, out_dir in FILES.items():
        if out_dir.exists() and any(out_dir.iterdir()):
            print(f"✓ {out_dir.name}/ already present")
            continue
        archive = download(name)
        tmp = DATA_DIR / f"_extract_{out_dir.name}"
        with zipfile.ZipFile(archive) as z:
            z.extractall(tmp)
        # each archive contains a single top-level folder (gt_db/ and labels/)
        (inner,) = [p for p in tmp.iterdir() if p.is_dir()]
        shutil.move(str(inner), out_dir)
        shutil.rmtree(tmp)
        print(f"✓ extracted to {out_dir}")

    n_images = len(list(GT_DIR.glob("s*/*.jpg")))
    n_labels = len(list(LABEL_DIR.glob("lab*")))
    print(f"\n{n_images} images, {n_labels} label files")
    assert n_images == 750 and n_labels == 750, "unexpected dataset size"


if __name__ == "__main__":
    main()
