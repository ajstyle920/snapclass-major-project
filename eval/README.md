# Face Recognition Benchmark

This folder measures how well SnapClass's face pipeline works and compares it with two other face recognition models. It is separate from the app: nothing here is imported by `app.py`, and its dependencies are not in the root `requirements.txt`.

**Questions answered**

1. How accurate is the current pipeline (dlib + SVM + distance threshold 0.6)?
2. Is 0.6 the right threshold?
3. Does the SVM add anything over plain nearest-neighbour matching?
4. Would a different model do better, and at what cost in speed?

## Summary of findings

| | Registered students correctly identified | Strangers wrongly accepted | Time per photo (CPU) |
|---|---|---|---|
| **Current app** (dlib, threshold 0.6) | 99.96% | **29.6%** | 226 ms |
| dlib, threshold 0.50 | 99.0% | 9.5% | 226 ms |
| FaceNet (tuned threshold) | 99.6% | 7.0% | 109 ms |
| **InsightFace ArcFace, small** (tuned threshold) | **100%** | **0%** | **43 ms** |
| InsightFace ArcFace, large (tuned threshold) | 100% | 0% | 560 ms |

- **The current app recognises registered students almost perfectly, but threshold 0.6 is too loose for rejecting strangers.** With 18 registered students, 29.6% of photos of unregistered people were accepted as some student. That matters most for FaceID login, where an unregistered person should never get in.
- **The problem grows with class size.** A stranger is compared with every registered student, so more students means more chances of a false match: 11% with 5 students, 37% with 35.
- **Lowering the dlib threshold is a trade-off, not a fix.** At 0.50, stranger acceptance drops to 9.5% while 1% of genuine students are missed. No dlib threshold gives both low stranger acceptance and no misses.
- **The SVM is redundant.** On all 3,560 probes it picked exactly the same student as nearest-neighbour matching. With one enrollment photo per student, each pairwise linear SVM is the perpendicular bisector between two embeddings, so the vote always selects the nearest embedding. The distance check does all the work.
- **InsightFace (ArcFace) separated registered students from strangers perfectly on this dataset**, and its small model is about 5× faster than dlib on CPU.
- All detectors found every face, including faces about 50 px wide in composite classroom photos, so this dataset does not differentiate the detectors. dlib's CNN detector gave no benefit here and took 37–41 s per group photo, against 2–4 s for the HOG detector.

These numbers come from a small, clean research dataset. Treat them as a comparison between models under identical conditions, not as the accuracy to expect in a real classroom. See [Limitations](#limitations).

## Dataset

[Georgia Tech Face Database](https://www.anefian.com/research/face_reco.htm): 50 people, 15 colour photos each (640×480), taken in two or three sessions in 1999 at the Center for Signal and Image Processing, Georgia Institute of Technology, with varied expression, lighting, scale and head tilt. Each photo comes with a manually labelled face box. Published by Ara V. Nefian for research.

- Two photos are pixel-identical copies of others (`s19/14` = `s19/08`, `s19/15` = `s19/11`). They are dropped, leaving **748 unique photos**.
- The images are not stored in this repository. `download_data.py` fetches them.

## Method

The protocol mirrors how the app is used.

**One-shot enrollment, open set.** Each registered student is enrolled from one photo, as in the app. The system is then shown (a) the other photos of registered students and (b) photos of people who were never registered. Four outcomes are counted:

| Outcome | Meaning in the app |
|---|---|
| Correctly identified | Student marked present |
| Missed | Student is there but marked absent |
| Wrong student | Student marked present under someone else's name |
| Stranger accepted | An unregistered person is taken for a student |

**Identity-disjoint tuning.** The 50 people are split into two halves of 25. In each half, 18 are registered and 7 are strangers. A model's threshold is tuned on one half and tested on the other, then the halves swap. This is repeated for 5 random seeds, giving 10 test runs. Thresholds are therefore never tested on people they were tuned on. The current app's 0.6 needs no tuning and is evaluated on the same 10 test runs.

**Tuning rule.** Highest correct-identification rate such that at most 1% of strangers are accepted and at most 1% of students are matched to the wrong person. When several thresholds tie, the middle one is used.

**Same enrollment photos for every model.** Every model found the face in all 748 photos, so all models enroll each student from the same randomly chosen photo.

**Models**

| Name | Detector | Embedding | Distance |
|---|---|---|---|
| dlib (current app) | HOG, 1× upsampling | ResNet, 128-D | Euclidean |
| FaceNet | MTCNN | InceptionResnetV1 (VGGFace2), 512-D | Cosine |
| InsightFace small (`buffalo_s`) | SCRFD-500M | MobileFaceNet ArcFace, 512-D | Cosine |
| InsightFace large (`buffalo_l`) | SCRFD-10G | ResNet-100 ArcFace, 512-D | Cosine |

The dlib wrapper reproduces `src/pipelines/face_pipeline.py` exactly, including the `SVC(kernel='linear', probability=True, class_weight='balanced')` classifier and `num_jitters=1`. All models are used pretrained; nothing is trained or fine-tuned here.

**Composite classroom photos.** The dataset has one person per photo, so `build_group_photos.py` tiles photos into 1280×960 images: 10 with 16 people (faces about 75 px) and 10 with 36 people (faces about 50 px), mixing 35 registered students and 15 strangers. These are synthetic and test many small faces per image, not occlusion or people looking away.

## Results

Full tables are in [`results/summary.md`](results/summary.md); raw numbers are in the CSV files next to it.

### 1. Open-set identification

Mean ± standard deviation over the 10 test runs.

| Model | Threshold | Correctly identified | Missed | Wrong student | Strangers accepted |
|---|---|---|---|---|---|
| dlib (current app) | 0.6 (app default) | 99.96% ± 0.1 | 0.0% ± 0.0 | 0.04% ± 0.1 | 29.6% ± 16.3 |
| dlib | 0.480 ± 0.052 (tuned) | 96.3% ± 4.4 | 3.7% ± 4.4 | 0.0% ± 0.0 | 10.1% ± 14.5 |
| FaceNet | 0.415 ± 0.062 (tuned) | 99.6% ± 0.6 | 0.4% ± 0.6 | 0.0% ± 0.0 | 7.0% ± 10.2 |
| InsightFace small | 0.600 ± 0.022 (tuned) | 100.0% ± 0.0 | 0.0% ± 0.0 | 0.0% ± 0.0 | 0.0% ± 0.0 |
| InsightFace large | 0.588 ± 0.019 (tuned) | 100.0% ± 0.0 | 0.0% ± 0.0 | 0.0% ± 0.0 | 0.0% ± 0.0 |

dlib thresholds are Euclidean distances; FaceNet and InsightFace thresholds are cosine distances, so they are not comparable across models.

For dlib and FaceNet, a threshold tuned to accept at most 1% of strangers on one group of people accepted 7–10% on the other, with a large spread between runs. Their thresholds do not transfer reliably, because a few look-alike pairs dominate the errors. InsightFace's did transfer.

![Open-set identification curve](results/open_set_roc.png)

### 2. The current app at different thresholds

| Threshold | Correctly identified | Missed | Strangers accepted |
|---|---|---|---|
| 0.40 | 87.3% | 12.7% | 0.1% |
| 0.45 | 95.7% | 4.3% | 3.3% |
| 0.50 | 99.0% | 1.0% | 9.5% |
| 0.55 | 99.8% | 0.2% | 17.6% |
| 0.60 (app default) | 99.96% | 0.0% | 29.6% |

![Threshold sweep for the current app](results/threshold_sweep_dlib_hog.png)

dlib's 0.6 was designed for comparing **one pair** of faces. In the app, each face is compared against **every** registered student, so the chance that a stranger falls within 0.6 of at least one of them adds up.

### 3. Why: distance distributions

![Distance distributions](results/distance_distributions.png)

For dlib, same-person and different-person distances overlap between about 0.35 and 0.6: 3.7% of different-person pairs are closer than 0.6. For InsightFace the two groups do not overlap at all on this dataset. The large model leaves a wider gap (largest same-person distance 0.44, smallest different-person distance 0.60) than the small one (0.55 and 0.61), so it has more safety margin on harder photos.

| Model | ROC AUC | Equal error rate | True accepts at 0.1% false accepts |
|---|---|---|---|
| dlib (current app) | 0.9996 | 0.93% | 93.0% |
| FaceNet | 1.0000 | 0.27% | 99.4% |
| InsightFace small | 1.0000 | 0.00% | 100.0% |
| InsightFace large | 1.0000 | 0.00% | 100.0% |

### 4. Class-size effect

Share of photos of 15 strangers accepted, as the number of registered students grows.

| Setting | 5 students | 10 students | 20 students | 35 students |
|---|---|---|---|---|
| dlib, app threshold 0.6 | 11.2% | 24.8% | 29.6% | 37.3% |
| dlib, threshold 0.487 | 2.8% | 6.3% | 8.7% | 10.4% |
| FaceNet, threshold 0.414 | 1.0% | 3.6% | 4.6% | 5.4% |
| InsightFace small, threshold 0.600 | 0.0% | 0.0% | 0.0% | 0.0% |

![Class-size effect](results/class_size_effect.png)

The app currently matches every face against all students in the database, not only those enrolled in the selected subject, so this effect applies to the whole user base. The thresholds in this table are the medians from experiment 1 and were tuned on overlapping people, so only the 0.6 row is fully independent.

### 5. Is the SVM needed?

| Model | Probes | Same prediction as nearest neighbour |
|---|---|---|
| dlib (current app) | 3,560 | 3,560 (100%) |
| FaceNet | 3,560 | 3,560 (100%) |
| InsightFace small | 3,560 | 3,560 (100%) |
| InsightFace large | 3,560 | 3,560 (100%) |

Removing the SVM would give identical results, remove the retraining step after every registration, and remove the one-vs-one training cost that grows with the square of the number of students.

### 6. Speed and composite classroom photos

Measured on a 2-core cloud CPU without a GPU.

| Model | Single photo (640×480) | 16-person photo | 36-person photo |
|---|---|---|---|
| dlib HOG (current app) | 226 ms | 2.0 s | 4.0 s |
| dlib CNN detector | not measured | 36.6 s | 40.8 s |
| FaceNet | 109 ms | 0.8 s | 1.4 s |
| InsightFace small | 43 ms | 0.3 s | 0.7 s |
| InsightFace large | 560 ms | 4.9 s | 10.8 s |

| Pipeline | Faces found | Present students marked present | Students wrongly marked present |
|---|---|---|---|
| dlib HOG, threshold 0.6 (current app) | 520/520 | 372/372 | 14 |
| dlib CNN detector, threshold 0.6 | 520/520 | 369/372 | 14 |
| dlib HOG, threshold 0.487 | 520/520 | 360/372 | 8 |
| FaceNet, threshold 0.414 | 520/520 | 369/372 | 7 |
| InsightFace small, threshold 0.600 | 520/520 | 372/372 | 0 |
| InsightFace large, threshold 0.589 | 520/520 | 372/372 | 0 |

Totals over the 20 composite photos. "Wrongly marked present" counts registered students who were not in the photo but were marked present because another face matched them.

### 7. Failure cases of the current app

[`results/failure_cases_current_app.csv`](results/failure_cases_current_app.csv) lists every error at threshold 0.6 across the 10 runs: 309 accepted strangers and 1 student matched to the wrong student. No student was missed. 68% of the accepted strangers had a distance between 0.5 and 0.6, and a few specific pairs of people account for many of them.

## Recommendations for the app

1. **Short term:** lower the threshold in `face_pipeline.py` from 0.6 towards 0.5 and replace the SVM with nearest-neighbour matching. This cuts stranger acceptance by about two thirds for about 1% more missed students, and removes retraining.
2. **Restrict matching to the selected subject's enrolled students** during attendance, which shrinks the number of people a stranger can be confused with.
3. **Longer term:** move to InsightFace's small ArcFace model. It was more accurate and faster here. This changes the embedding size from 128 to 512, so every student would need to re-register. InsightFace's pretrained models are released for non-commercial research use, which should be checked before any commercial deployment.

## Limitations

- **Small and easy dataset.** 50 people, mostly frontal, indoors, in one room, photographed in 1999. Perfect scores for InsightFace mean this dataset cannot separate the two InsightFace models or show their real error rate.
- **Not the deployment population.** The photos are not of Indian classrooms or phone and webcam captures. Real accuracy is likely to be lower for every model.
- **Few strangers per run.** Each test run has 7 strangers, so one look-alike pair moves the stranger-acceptance rate a lot. That is why its standard deviation is large.
- **Composite classroom photos are synthetic.** They do not include occlusion, motion blur, extreme angles or uneven lighting.
- **Detection was not stressed.** Every detector found every face, so the benchmark says nothing about which detector is better on hard images.
- **Face only.** The voice pipeline is not evaluated here.
- **No liveness test.** Whether a printed photo can pass is outside this benchmark.

## Reproducing

Tested on Python 3.13. The full run took about 40 minutes on a 2-core CPU, most of it the dlib CNN detector on the composite photos.

```bash
pip install -r eval/requirements-eval.txt
pip install --no-deps facenet-pytorch==2.6.0

python eval/download_data.py                        # dataset → eval/data/
python eval/extract_embeddings.py --phase singles   # run each model on 748 photos
python eval/build_group_photos.py                   # composite classroom photos
python eval/extract_embeddings.py --phase groups
python eval/evaluate.py                             # metrics, tables, plots → eval/results/
python eval/sanity_check.py                         # checks on the evaluation code
```

All random choices are seeded, and running `evaluate.py` twice produces byte-identical CSV files.

**Sanity checks** ([`results/sanity_checks.md`](results/sanity_checks.md)): tune and test people never overlap; enrollment photos are never used as probes; random embeddings score at chance level; enrolling students under each other's names drops correct identification to 0%; each enrollment photo matches itself at distance 0.

## Files

| File | Purpose |
|---|---|
| `common.py` | Paths, dataset loading, data splits |
| `models.py` | One wrapper per face model, same interface |
| `download_data.py` | Fetches the dataset and face-box labels |
| `extract_embeddings.py` | Runs each model once per photo and caches the embeddings |
| `build_group_photos.py` | Builds the composite classroom photos |
| `evaluate.py` | Computes all metrics, tables and plots |
| `sanity_check.py` | Verifies the evaluation code |
| `requirements-eval.txt` | Benchmark-only dependencies |
| `results/` | Tables (`.md`, `.csv`) and plots (`.png`) |
