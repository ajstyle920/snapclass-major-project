import dlib
import numpy as np
import face_recognition_models
import streamlit as st

from src.database.db import get_all_students
from src.pipelines.face_matching import match_faces


@st.cache_resource
def load_dlib_models():
    detector = dlib.get_frontal_face_detector()

    sp = dlib.shape_predictor(
        face_recognition_models.pose_predictor_model_location()
    )

    facerec = dlib.face_recognition_model_v1(
        face_recognition_models.face_recognition_model_location()
    )

    return detector, sp, facerec


def get_face_embeddings(image_np):
    detector, sp, facerec = load_dlib_models()
    faces = detector(image_np, 1)

    encodings = []

    for face in faces:
        shape = sp(image_np, face)
        face_descriptor = facerec.compute_face_descriptor(image_np, shape, 1)  # 128 embedding

        encodings.append(np.array(face_descriptor))
    return encodings


@st.cache_resource
def get_face_index():
    """Face embeddings and ids of every registered student, loaded once and cached."""
    embeddings = []
    student_ids = []

    for student in get_all_students() or []:
        embedding = student.get('face_embedding')
        if embedding:
            embeddings.append(np.array(embedding))
            student_ids.append(student.get('student_id'))

    if not embeddings:
        return None

    return {'embeddings': np.stack(embeddings), 'student_ids': student_ids}


def refresh_face_index():
    """Reload registered students after a new registration.

    Only the student index is cleared. The dlib and voice models stay loaded.
    """
    get_face_index.clear()
    return bool(get_face_index())


def predict_attendance(class_image_np):
    encodings = get_face_embeddings(class_image_np)

    face_index = get_face_index()

    if not face_index:
        return {}, [], len(encodings)

    student_ids = face_index['student_ids']
    detected_student = match_faces(encodings, face_index['embeddings'], student_ids)

    return detected_student, sorted(set(student_ids)), len(encodings)
