"""Match face embeddings against enrolled students.

Kept free of Streamlit and database imports so the exact logic the app uses
can be run directly on a benchmark dataset (see eval/check_app.py).
"""
import numpy as np

# Maximum Euclidean distance between two dlib face embeddings for them to count
# as the same person. dlib suggests 0.6 for comparing ONE pair of faces. Here each
# face is compared with every enrolled student, so strangers get many chances to
# match; the benchmark in eval/ measured 29.6% of strangers accepted at 0.6 and
# 9.5% at 0.5, for about 1% more missed students. See eval/README.md.
FACE_MATCH_THRESHOLD = 0.5


def match_face(encoding, known_embeddings, known_ids, threshold=FACE_MATCH_THRESHOLD):
    """Find the enrolled student closest to one face embedding.

    Returns (student_id, distance). student_id is None when even the closest
    enrolled embedding is farther than `threshold`, i.e. the face is unknown.
    """
    distances = np.linalg.norm(np.asarray(known_embeddings) - np.asarray(encoding), axis=1)
    nearest = int(np.argmin(distances))
    distance = float(distances[nearest])
    if distance <= threshold:
        return known_ids[nearest], distance
    return None, distance


def match_faces(encodings, known_embeddings, known_ids, threshold=FACE_MATCH_THRESHOLD):
    """Match every face in a photo. Returns {student_id: True} for each student recognised."""
    detected = {}
    for encoding in encodings:
        student_id, _ = match_face(encoding, known_embeddings, known_ids, threshold)
        if student_id is not None:
            detected[student_id] = True
    return detected
