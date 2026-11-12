"""Compute every benchmark metric from the cached embeddings and write eval/results/.

    python eval/evaluate.py

Two versions of the SnapClass face matching are measured:

* v1 (original app): linear SVM picks a student, accepted if Euclidean distance <= 0.6
* v2 (current app):  nearest enrolled embedding, accepted if distance <= 0.5.
  v2 is evaluated by calling the app's own function, src/pipelines/face_matching.py.

Experiments
-----------
1. Open-set identification, the SnapClass use case: one enrollment photo per
   registered student, probes from registered students and from strangers.
   Thresholds of the compared models are tuned on one half of the people and
   tested on the other half (2 folds x 5 random seeds = 10 test runs).
2. dlib at different thresholds.
3. Matching rule ablation: v1's SVM-then-distance rule vs plain nearest neighbour.
4. Verification: same-person vs different-person distance distributions, ROC AUC
   and EER (threshold-free quality of each model's embeddings).
5. Class-size effect: how often strangers are accepted as the number of
   registered students grows.
6. Detection rate and speed.
7. Composite classroom photos: face detection and attendance accuracy.
8. Failure cases of v1 and v2.
"""
from __future__ import annotations

import sys
import warnings

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.svm import SVC

from common import (EVAL_DIR, GROUP_DIR, RESULTS_DIR, SEEDS, TARGET_UNKNOWN_FAR, box_center_inside,
                    enrollment_candidates, identity_halves, list_images, load_cache, load_json,
                    make_gallery, save_json, subjects)
from models import RECOGNITION_MODELS, DlibHOG, FaceModel, FaceNet, InsightFace

sys.path.insert(0, str(EVAL_DIR.parent))  # repository root, to import the app's matching code
from src.pipelines.face_matching import FACE_MATCH_THRESHOLD, match_face, match_faces  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

warnings.filterwarnings("ignore")

V1_THRESHOLD = 0.6                   # original app: dlib's suggested pair-comparison threshold
V2_THRESHOLD = FACE_MATCH_THRESHOLD  # current app: read from the app code so this never drifts
TARGET_WRONG_ID = 0.01  # tuning also requires <=1% of registered students matched to someone else

V1 = f"v1 app (SVM, threshold {V1_THRESHOLD})"
V2 = f"v2 app (nearest match, threshold {V2_THRESHOLD})"

# lightweight metadata objects (distance metric + labels) without loading the networks
META: dict[str, FaceModel] = {}
for cls, kwargs in [(DlibHOG, {}), (FaceNet, {}), (InsightFace, {"pack": "buffalo_s"}),
                    (InsightFace, {"pack": "buffalo_l"})]:
    obj = cls.__new__(cls)
    if cls is InsightFace:
        pack = kwargs["pack"]
        obj.name = f"insightface_{pack}"
        obj.label = f"InsightFace ArcFace ({'large' if pack == 'buffalo_l' else 'small'}, {pack})"
    META[obj.name] = obj

SHORT = {
    "dlib_hog": "dlib",
    "facenet": "FaceNet",
    "insightface_buffalo_s": "InsightFace small",
    "insightface_buffalo_l": "InsightFace large",
}
COLORS = {"dlib_hog": "#d1495b", "facenet": "#edae49", "insightface_buffalo_s": "#66a182",
          "insightface_buffalo_l": "#2e4057"}


def threshold_grid(model: FaceModel) -> np.ndarray:
    return np.round(np.arange(0.0, 1.2001, 0.005) if model.metric == "euclidean"
                    else np.arange(0.0, 1.0001, 0.0025), 4)


# --------------------------------------------------------------------------- identification
def v1_classifier(gallery, known):
    """The classifier the original app trained (src/pipelines/face_pipeline.py before v2)."""
    return SVC(kernel="linear", probability=True, class_weight="balanced").fit(gallery, known)


def identify(model_name, cache, known, unknown, enroll, rule):
    """Run every probe of one split. Returns per-probe true id, predicted id and distance.

    rule: "nn"  nearest enrolled embedding in the model's own metric
          "svm" v1 app: linear SVM picks the student
          "app" v2 app: the app's match_face function (dlib embeddings only)
    """
    images = list_images()
    meta = META[model_name]
    gallery = np.stack([cache[enroll[s]]["embedding"] for s in known])
    clf = v1_classifier(gallery, known) if rule == "svm" else None
    known_set, enroll_ids = set(known), set(enroll.values())
    probes = [i for i in images
              if (i.subject in known_set and i.image_id not in enroll_ids) or i.subject in unknown]
    out = []
    for p in probes:
        emb = cache[p.image_id]["embedding"]
        row = {"image_id": p.image_id, "true": p.subject, "is_known": p.subject in known_set,
               "detected": emb is not None, "pred": None, "dist": np.inf}
        if emb is not None:
            if rule == "app":
                row["pred"], row["dist"] = match_face(emb, gallery, known, threshold=np.inf)
            else:
                d = meta.distance(emb, gallery)[0]
                idx = int(np.argmin(d)) if rule == "nn" else known.index(clf.predict([emb])[0])
                row["pred"], row["dist"] = known[idx], float(d[idx])
        out.append(row)
    return pd.DataFrame(out)


def rates(res: pd.DataFrame, t: float) -> dict:
    kn, un = res[res.is_known], res[~res.is_known]
    acc_k = kn.detected & (kn.dist <= t)
    acc_u = un.detected & (un.dist <= t)
    return {
        "correct_id": float((acc_k & (kn.pred == kn.true)).mean()),
        "wrong_id": float((acc_k & (kn.pred != kn.true)).mean()),
        "missed": float((~acc_k).mean()),
        "unknown_accepted": float(acc_u.mean()),
    }


def curve(res, grid):
    return pd.DataFrame([{"threshold": t, **rates(res, t)} for t in grid])


def tune(curve_df):
    """Highest correct-ID rate with <=1% strangers accepted and <=1% wrong-person matches.

    Several neighbouring thresholds usually reach that same best rate; the middle
    one is returned, which leaves a safety margin on both sides.
    """
    ok = curve_df[(curve_df.unknown_accepted <= TARGET_UNKNOWN_FAR) & (curve_df.wrong_id <= TARGET_WRONG_ID)]
    if ok.empty:
        return float(curve_df.threshold.min())
    plateau = ok[ok.correct_id == ok.correct_id.max()].threshold.to_numpy()
    return float(plateau[len(plateau) // 2])


def identification(caches, candidates):
    fold_rows, test_curves, raw = [], {}, {}
    for m in RECOGNITION_MODELS:
        grid = threshold_grid(META[m])
        for rule in ["nn", "svm"] + (["app"] if m == "dlib_hog" else []):
            pooled = []
            for seed in SEEDS:
                a, b = identity_halves(seed)
                for fold, (tune_half, test_half) in enumerate([(a, b), (b, a)]):
                    tk, tu, te = make_gallery(tune_half, seed, candidates)
                    xk, xu, xe = make_gallery(test_half, seed, candidates)
                    t_star = tune(curve(identify(m, caches[m], tk, tu, te, rule), grid))
                    test_res = identify(m, caches[m], xk, xu, xe, rule)
                    raw[(m, rule, seed, fold)] = test_res
                    pooled.append(curve(test_res, grid))
                    settings = [("tuned", t_star)]
                    if (m, rule) == ("dlib_hog", "svm"):
                        settings.append((V1, V1_THRESHOLD))
                    if (m, rule) == ("dlib_hog", "app"):
                        settings.append((V2, V2_THRESHOLD))
                    for mode, t in settings:
                        fold_rows.append({"model": m, "rule": rule, "threshold_mode": mode, "seed": seed,
                                          "fold": fold, "threshold": t, **rates(test_res, t)})
            test_curves[(m, rule)] = pd.concat(pooled).groupby("threshold").mean().reset_index()
    # how often does v1's SVM pick the same student as plain nearest neighbour?
    agreement = []
    for m in RECOGNITION_MODELS:
        same = total = 0
        for (mm, rule, seed, fold), res in raw.items():
            if mm == m and rule == "svm":
                same += int((res.pred == raw[(m, "nn", seed, fold)].pred).sum())
                total += len(res)
        agreement.append({"model": SHORT[m], "probes": total, "same_prediction": same,
                          "agreement": same / total})
    return pd.DataFrame(fold_rows), test_curves, raw, pd.DataFrame(agreement)


def summarise(folds: pd.DataFrame) -> pd.DataFrame:
    metrics = ["correct_id", "wrong_id", "missed", "unknown_accepted", "threshold"]
    g = folds.groupby(["model", "rule", "threshold_mode"])[metrics]
    mean, std = g.mean(), g.std()
    out = mean.copy()
    for col in metrics:
        out[col + "_std"] = std[col]
    return out.reset_index()


# --------------------------------------------------------------------------- verification
def verification(caches):
    images = list_images()
    rows, dists = [], {}
    for m in RECOGNITION_MODELS:
        c = caches[m]
        ok = [i for i in images if c[i.image_id]["embedding"] is not None]
        emb = np.stack([c[i.image_id]["embedding"] for i in ok])
        subj = np.array([i.subject for i in ok])
        d = META[m].distance(emb, emb)
        iu = np.triu_indices(len(ok), k=1)
        dist = d[iu]
        same = subj[iu[0]] == subj[iu[1]]
        fpr, tpr, thr = roc_curve(same, -dist)
        fnr = 1 - tpr
        eer_idx = np.nanargmin(np.abs(fnr - fpr))
        rows.append({
            "model": SHORT[m],
            "genuine pairs": int(same.sum()),
            "impostor pairs": int((~same).sum()),
            "ROC AUC": roc_auc_score(same, -dist),
            "EER": float((fpr[eer_idx] + fnr[eer_idx]) / 2),
            "EER threshold": float(-thr[eer_idx]),
            "TAR @ FAR=1%": float(np.interp(0.01, fpr, tpr)),
            "TAR @ FAR=0.1%": float(np.interp(0.001, fpr, tpr)),
        })
        dists[m] = (dist[same], dist[~same])
    return pd.DataFrame(rows), dists


# --------------------------------------------------------------------------- class size
def class_size_effect(caches, candidates, recommended):
    """Strangers accepted vs number of registered students, at fixed thresholds.

    The same 15 strangers are probed against a growing class, so the only thing
    that changes is how many registered students each stranger can be confused with.
    """
    settings = [("dlib_hog", "svm", V1_THRESHOLD, f"dlib, {V1}"),
                ("dlib_hog", "app", V2_THRESHOLD, f"dlib, {V2}"),
                ("facenet", "nn", recommended["facenet"], f"FaceNet, tuned threshold {recommended['facenet']:.3f}"),
                ("insightface_buffalo_s", "nn", recommended["insightface_buffalo_s"],
                 f"InsightFace small, tuned threshold {recommended['insightface_buffalo_s']:.3f}")]
    rows = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed + 5000)
        order = [str(s) for s in rng.permutation(subjects())]
        unknown = sorted(order[-15:])
        enroll_all = {s: str(rng.choice(sorted(candidates[s]))) for s in order[:35]}
        for n in [5, 10, 20, 35]:
            known = sorted(order[:n])
            enroll = {s: enroll_all[s] for s in known}
            for m, rule, t, label in settings:
                r = rates(identify(m, caches[m], known, unknown, enroll, rule), t)
                rows.append({"setting": label, "model": m, "registered_students": n, "seed": seed, **r})
    df = pd.DataFrame(rows)
    return df.groupby(["setting", "model", "registered_students"], sort=False)[
        ["correct_id", "unknown_accepted", "wrong_id", "missed"]].mean().reset_index()


# --------------------------------------------------------------------------- detection
def detection_and_speed(caches):
    rows = []
    for m in RECOGNITION_MODELS:
        c = caches[m]
        rows.append({
            "model": SHORT[m],
            "face found in single photos": np.mean([r["embedding"] is not None for r in c.values()]),
            "median seconds per photo (CPU)": float(np.median([r["seconds"] for r in c.values()])),
            "embedding size": META[m].embedding_dim,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- group photos
def group_photos(recommended):
    meta = load_json(GROUP_DIR / "groups.json")
    known, enroll = meta["known"], meta["enroll"]
    singles = {m: load_cache(m) for m in RECOGNITION_MODELS}
    # (pipeline label, detector cache, embedding model, rule, threshold)
    configs = [
        (f"dlib HOG detector, {V1}", "dlib_hog", "dlib_hog", "svm", V1_THRESHOLD),
        (f"dlib HOG detector, {V2}", "dlib_hog", "dlib_hog", "app", V2_THRESHOLD),
        (f"dlib CNN detector, {V1}", "dlib_cnn", "dlib_hog", "svm", V1_THRESHOLD),  # same ResNet embeddings
    ] + [(f"{SHORT[m]}, tuned threshold {recommended[m]:.3f}", m, m, "nn", recommended[m])
         for m in RECOGNITION_MODELS[1:]]
    rows = []
    for label, det_model, emb_model, rule, t in configs:
        groups = load_cache(det_model, "groups")
        gallery = np.stack([singles[emb_model][enroll[s]]["embedding"] for s in known])
        clf = v1_classifier(gallery, known) if rule == "svm" else None
        for photo in meta["photos"]:
            rec = groups[photo["file"]]
            tiles = photo["tiles"]
            present = {tl["subject"] for tl in tiles if tl["subject"] in known}
            found_tiles = set()
            for box, _ in rec["faces"]:
                found_tiles.update(k for k, tl in enumerate(tiles) if box_center_inside(box, tl["box"]))
            encodings = [emb for _, emb in rec["faces"]]
            if rule == "app":  # the app's own function, exactly as predict_attendance calls it
                marked = set(match_faces(encodings, gallery, known, threshold=t))
            else:
                marked = set()
                for emb in encodings:
                    d = META[emb_model].distance(emb, gallery)[0]
                    idx = int(np.argmin(d)) if rule == "nn" else known.index(clf.predict([emb])[0])
                    if d[idx] <= t:
                        marked.add(known[idx])
            rows.append({
                "pipeline": label, "layout": photo["layout"], "file": photo["file"], "people": len(tiles),
                "faces_found": len(found_tiles), "students_present": len(present),
                "marked_correctly": len(marked & present), "false_present": len(marked - present),
                "seconds": rec["seconds"],
            })
    df = pd.DataFrame(rows)
    summary = df.groupby(["pipeline", "layout"], sort=False).agg(
        photos=("file", "count"), people=("people", "sum"), faces_found=("faces_found", "sum"),
        students_present=("students_present", "sum"), marked_correctly=("marked_correctly", "sum"),
        false_present=("false_present", "sum"), median_seconds=("seconds", "median")).reset_index()
    summary["face_detection_rate"] = summary.faces_found / summary.people
    summary["attendance_recall"] = summary.marked_correctly / summary.students_present
    return df, summary


# --------------------------------------------------------------------------- failure cases
def failure_cases(raw):
    """Every error v1 and v2 make across the 10 test runs."""
    rows = []
    for system, rule, t in [(V1, "svm", V1_THRESHOLD), (V2, "app", V2_THRESHOLD)]:
        for (m, r_rule, seed, fold), res in raw.items():
            if (m, r_rule) != ("dlib_hog", rule):
                continue
            for _, r in res.iterrows():
                accepted = r.detected and r.dist <= t
                if r.is_known and not r.detected:
                    outcome = "no face detected"
                elif r.is_known and not accepted:
                    outcome = "registered student rejected"
                elif r.is_known and r.pred != r.true:
                    outcome = "registered student matched to wrong student"
                elif not r.is_known and accepted:
                    outcome = "stranger accepted as a student"
                else:
                    continue
                rows.append({"system": system, "seed": seed, "fold": fold, "image_id": r.image_id,
                             "true_person": r.true, "registered": bool(r.is_known),
                             "nearest_student": r.pred, "distance": round(r.dist, 4), "outcome": outcome})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- plots
def plot_distributions(dists):
    fig, axes = plt.subplots(1, len(dists), figsize=(4.2 * len(dists), 3.4))
    for ax, (m, (gen, imp)) in zip(axes, dists.items()):
        bins = np.linspace(0, max(gen.max(), imp.max()), 80)
        ax.hist(imp, bins=bins, density=True, alpha=0.6, color="#999999", label="different people")
        ax.hist(gen, bins=bins, density=True, alpha=0.75, color=COLORS[m], label="same person")
        if m == "dlib_hog":
            ax.axvline(V1_THRESHOLD, color="black", ls="--", lw=1, label=f"v1 threshold {V1_THRESHOLD}")
            ax.axvline(V2_THRESHOLD, color="black", ls="-", lw=1, label=f"v2 threshold {V2_THRESHOLD}")
        ax.set_title(SHORT[m])
        ax.set_xlabel("Euclidean distance" if META[m].metric == "euclidean" else "cosine distance")
        ax.set_yticks([])
    handles, labels = axes[0].get_legend_handles_labels()
    handles[1].set_facecolor("#555555")  # neutral swatch: each panel has its own model colour
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, -0.08))
    fig.suptitle("Distance between two photos: same person vs different people", y=1.02)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "distance_distributions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_open_set(test_curves, summary):
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for m in RECOGNITION_MODELS:
        c = test_curves[(m, "nn")]
        ax.plot(c.unknown_accepted * 100, c.correct_id * 100, color=COLORS[m], lw=2, label=SHORT[m])
    for mode, marker in [(V1, "o"), (V2, "s")]:
        pt = summary[summary.threshold_mode == mode]
        ax.scatter(pt.unknown_accepted * 100, pt.correct_id * 100, color="black", marker=marker,
                   zorder=5, label=mode)
    ax.set_xlim(-0.5, 32)
    ax.set_ylim(80, 100.6)
    ax.set_xlabel("Strangers wrongly accepted (%)")
    ax.set_ylabel("Registered students correctly identified (%)")
    ax.set_title("Open-set identification (one enrollment photo per student)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "open_set_roc.png", dpi=150)
    plt.close(fig)


def plot_threshold_sweep(test_curves, recommended):
    for m in RECOGNITION_MODELS:
        c = test_curves[(m, "nn")]
        fig, ax = plt.subplots(figsize=(6.4, 4))
        ax.plot(c.threshold, c.correct_id * 100, color="#2e7d32", lw=2, label="correctly identified")
        ax.plot(c.threshold, c.missed * 100, color="#f9a825", lw=2, label="missed (marked absent)")
        ax.plot(c.threshold, c.unknown_accepted * 100, color="#c62828", lw=2, label="strangers accepted")
        ax.plot(c.threshold, c.wrong_id * 100, color="#6a1b9a", lw=2, ls=":", label="matched to wrong student")
        if m == "dlib_hog":
            ax.axvline(V1_THRESHOLD, color="grey", ls="--", lw=1.2, label=f"v1 app threshold {V1_THRESHOLD}")
            ax.axvline(V2_THRESHOLD, color="black", ls="-", lw=1.2, label=f"v2 app threshold {V2_THRESHOLD}")
            ax.set_xlim(0.2, 0.9)
        else:
            ax.axvline(recommended[m], color="black", ls="--", lw=1, label=f"tuned threshold {recommended[m]:.3f}")
            ax.set_xlim(max(0, recommended[m] - 0.4), min(1.0, recommended[m] + 0.4))
        ax.set_ylim(0, 101)
        ax.set_xlabel("distance threshold")
        ax.set_ylabel("% of probes")
        ax.set_title(f"Threshold sweep: {SHORT[m]}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7.5)
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / f"threshold_sweep_{m}.png", dpi=150)
        plt.close(fig)


def plot_class_size(size_df):
    fig, ax = plt.subplots(figsize=(6.4, 4))
    styles = ["--", "-", "-", "-"]
    for (label, g), ls in zip(size_df.groupby("setting", sort=False), styles):
        ax.plot(g.registered_students, g.unknown_accepted * 100, marker="o", lw=2, ls=ls,
                color=COLORS[g.model.iloc[0]], label=label)
    ax.set_xticks([5, 10, 20, 35])
    ax.set_xlabel("Registered students in the class")
    ax.set_ylabel("Strangers wrongly accepted (%)")
    ax.set_title("A bigger class gives a stranger more people to be confused with")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "class_size_effect.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- report
def pct(x, sd=None):
    """Percent with one decimal; two decimals when one would round a real error away (99.96% -> 100.0%)."""
    digits = 2 if (0 < x < 0.0005 or 0.9995 <= x < 1) else 1
    return f"{x * 100:.{digits}f}%" + (f" ± {sd * 100:.1f}" if sd is not None else "")


def write_summary(det, ver, summ, test_curves, agreement, size_df, group_sum, fails, recommended):
    lines = ["# Benchmark results", "",
             f"Generated by `eval/evaluate.py`. Dataset: Georgia Tech Face Database "
             f"(50 people, {len(list_images())} unique photos).", "",
             f"- **v1 app:** linear SVM picks a student, accepted if Euclidean distance ≤ {V1_THRESHOLD} "
             "(the original design).",
             f"- **v2 app:** nearest enrolled embedding, accepted if distance ≤ {V2_THRESHOLD} "
             "(the current app, evaluated through `src/pipelines/face_matching.py`).", "",
             "Identification numbers are mean ± standard deviation over 10 test runs "
             "(5 random seeds × 2 identity-disjoint folds). Each run: 18 registered students enrolled from "
             "one photo each, about 252 other photos of those students and 105 photos of 7 strangers.", ""]

    lines += ["## 1. Open-set identification (main result)", "",
              "| Pipeline | Threshold | Correctly identified | Missed | Wrong student | Strangers accepted |",
              "|---|---|---|---|---|---|"]
    order = [("dlib_hog", "svm", V1, "dlib, v1 app (SVM + distance check)"),
             ("dlib_hog", "app", V2, "dlib, v2 app (nearest match)"),
             ("dlib_hog", "nn", "tuned", "dlib, nearest match")] + \
            [(m, "nn", "tuned", f"{SHORT[m]}, nearest match") for m in RECOGNITION_MODELS[1:]]
    for m, rule, mode, name in order:
        r = summ[(summ.model == m) & (summ.rule == rule) & (summ.threshold_mode == mode)].iloc[0]
        thr = f"{r.threshold:g} (fixed)" if mode != "tuned" else f"{r.threshold:.3f} ± {r.threshold_std:.3f} (tuned)"
        lines.append(f"| {name} | {thr} | {pct(r.correct_id, r.correct_id_std)} | "
                     f"{pct(r.missed, r.missed_std)} | {pct(r.wrong_id, r.wrong_id_std)} | "
                     f"{pct(r.unknown_accepted, r.unknown_accepted_std)} |")
    lines += ["", "Tuned thresholds are chosen on the other half of the people with the rule: highest "
              "correct-identification rate such that at most 1% of strangers are accepted and at most 1% "
              "of registered students are matched to someone else. dlib uses Euclidean distance; "
              "FaceNet and InsightFace use cosine distance, so thresholds are not comparable across models.", ""]

    lines += ["## 2. dlib at different thresholds", "",
              "Averaged over the same 10 test runs (no tuning involved).", "",
              "| Threshold | Correctly identified | Missed | Wrong student | Strangers accepted |", "|---|---|---|---|---|"]
    c = test_curves[("dlib_hog", "app")]
    for t in [0.40, 0.45, 0.50, 0.55, 0.60]:
        r = c[np.isclose(c.threshold, t)].iloc[0]
        mark = " (v1 app)" if np.isclose(t, V1_THRESHOLD) else " (v2 app)" if np.isclose(t, V2_THRESHOLD) else ""
        lines.append(f"| {t:.2f}{mark} | {pct(r.correct_id)} | {pct(r.missed)} | {pct(r.wrong_id)} | {pct(r.unknown_accepted)} |")
    lines.append("")

    lines += ["## 3. Was the SVM needed?", "",
              "The v1 app first asked a linear SVM for the most likely student and then checked the distance. "
              "The table counts how often the SVM picked the same student as simply taking the nearest "
              "enrolled embedding.", "",
              "| Model | Probes | Same prediction as nearest neighbour |", "|---|---|---|"]
    for _, r in agreement.iterrows():
        lines.append(f"| {r.model} | {r.probes} | {r.same_prediction} ({pct(r.agreement)}) |")
    lines.append("")

    lines += ["## 4. Embedding quality (verification, threshold-free)", "",
              f"Every pair of photos: {int(ver['genuine pairs'].iloc[0])} same-person pairs and "
              f"{int(ver['impostor pairs'].iloc[0])} different-person pairs.", "",
              "| Model | ROC AUC | Equal error rate | True accepts @ 1% false accepts | @ 0.1% false accepts |",
              "|---|---|---|---|---|"]
    for _, r in ver.iterrows():
        lines.append(f"| {r.model} | {r['ROC AUC']:.4f} | {r.EER * 100:.2f}% | "
                     f"{pct(r['TAR @ FAR=1%'])} | {pct(r['TAR @ FAR=0.1%'])} |")
    lines.append("")

    lines += ["## 5. Class-size effect", "",
              "Share of photos of 15 strangers accepted as some registered student, as the class grows "
              "(mean of 5 seeds).", ""]
    piv = size_df.pivot(index="setting", columns="registered_students", values="unknown_accepted")
    piv = piv.loc[size_df.setting.unique()]
    lines += ["| Setting | " + " | ".join(f"{n} students" for n in piv.columns) + " |",
              "|---|" + "---|" * len(piv.columns)]
    for setting, row in piv.iterrows():
        lines.append(f"| {setting} | " + " | ".join(pct(v) for v in row) + " |")
    lines.append("")

    lines += ["## 6. Detection and speed (single-person photos, 640×480, 2-core CPU)", "",
              "| Model | Face found | Median time per photo | Embedding size |", "|---|---|---|---|"]
    for _, r in det.iterrows():
        lines.append(f"| {r.model} | {pct(r['face found in single photos'])} | "
                     f"{r['median seconds per photo (CPU)'] * 1000:.0f} ms | {r['embedding size']} |")
    lines.append("")

    lines += ["## 7. Composite classroom photos", "",
              "20 synthetic 1280×960 photos (10 with 16 people, 10 with 36 people) built from dataset photos; "
              "35 registered students, 15 strangers.", "",
              "| Pipeline | Layout | Faces found | Present students marked present | Students wrongly marked present | Median time per photo |",
              "|---|---|---|---|---|---|"]
    for layout in ["front_rows", "back_rows"]:
        for _, r in group_sum[group_sum.layout == layout].iterrows():
            lay = "16 people, faces ~75 px" if layout == "front_rows" else "36 people, faces ~50 px"
            lines.append(f"| {r.pipeline} | {lay} | {r.faces_found}/{r.people} ({pct(r.face_detection_rate)}) | "
                         f"{r.marked_correctly}/{r.students_present} ({pct(r.attendance_recall)}) | "
                         f"{r.false_present} | {r.median_seconds:.1f} s |")
    lines.append("")

    lines += ["## 8. Errors of the two app versions", "",
              "Counts over all 10 test runs (3,560 probes). Every row is in `failure_cases.csv`.", "",
              "| System | Strangers accepted | Students rejected | Students matched to wrong student |",
              "|---|---|---|---|"]
    for system in [V1, V2]:
        f = fails[fails.system == system].outcome.value_counts()
        lines.append(f"| {system} | {f.get('stranger accepted as a student', 0)} | "
                     f"{f.get('registered student rejected', 0)} | "
                     f"{f.get('registered student matched to wrong student', 0)} |")
    lines.append("")

    lines += ["## Tuned thresholds (median of the 10 runs)", ""] + \
             [f"- {SHORT[m]}: **{t:.3f}** ({META[m].metric} distance)" for m, t in recommended.items()] + [""]
    (RESULTS_DIR / "summary.md").write_text("\n".join(lines))


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    caches = {m: load_cache(m) for m in RECOGNITION_MODELS}
    candidates = enrollment_candidates(RECOGNITION_MODELS)

    folds, test_curves, raw, agreement = identification(caches, candidates)
    summ = summarise(folds)
    recommended = {m: float(folds[(folds.model == m) & (folds.rule == "nn") &
                                  (folds.threshold_mode == "tuned")].threshold.median())
                   for m in RECOGNITION_MODELS}
    ver, dists = verification(caches)
    size_df = class_size_effect(caches, candidates, recommended)
    det = detection_and_speed(caches)
    group_df, group_sum = group_photos(recommended)
    fails = failure_cases(raw)

    folds.to_csv(RESULTS_DIR / "identification_by_fold.csv", index=False)
    summ.to_csv(RESULTS_DIR / "identification_summary.csv", index=False)
    test_curves[("dlib_hog", "app")].to_csv(RESULTS_DIR / "threshold_sweep_dlib.csv", index=False)
    agreement.to_csv(RESULTS_DIR / "svm_vs_nearest_neighbour.csv", index=False)
    ver.to_csv(RESULTS_DIR / "verification.csv", index=False)
    size_df.to_csv(RESULTS_DIR / "class_size_effect.csv", index=False)
    det.to_csv(RESULTS_DIR / "detection_speed.csv", index=False)
    group_df.to_csv(RESULTS_DIR / "group_photos_by_photo.csv", index=False)
    group_sum.to_csv(RESULTS_DIR / "group_photos_summary.csv", index=False)
    fails.to_csv(RESULTS_DIR / "failure_cases.csv", index=False)
    save_json(recommended, RESULTS_DIR / "recommended_thresholds.json")

    plot_distributions(dists)
    plot_open_set(test_curves, summ)
    plot_threshold_sweep(test_curves, recommended)
    plot_class_size(size_df)
    write_summary(det, ver, summ, test_curves, agreement, size_df, group_sum, fails, recommended)
    print((RESULTS_DIR / "summary.md").read_text())


if __name__ == "__main__":
    main()
