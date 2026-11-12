"""Face models compared in this benchmark, all behind one small interface.

Every model takes an RGB image (H x W x 3, uint8) and returns one `Face` per
detected face: its bounding box and its embedding vector.

`DlibHOG` reproduces the SnapClass app pipeline exactly
(src/pipelines/face_pipeline.py): HOG detector with 1x upsampling, 68-point
landmarks, dlib ResNet descriptor with num_jitters=1.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

MODEL_CACHE = Path(__file__).resolve().parent / "cache" / "weights"


@dataclass
class Face:
    box: tuple[float, float, float, float]  # left, top, right, bottom
    embedding: np.ndarray


class FaceModel:
    name: str
    label: str
    metric: str  # "euclidean" or "cosine"
    embedding_dim: int

    def __call__(self, image_rgb: np.ndarray) -> list[Face]:
        raise NotImplementedError

    def distance(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Distance between embedding(s) `a` (n x d or d) and `b` (m x d or d). Lower = more similar."""
        a, b = np.atleast_2d(a), np.atleast_2d(b)
        if self.metric == "euclidean":
            return np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
        a = a / np.linalg.norm(a, axis=1, keepdims=True)
        b = b / np.linalg.norm(b, axis=1, keepdims=True)
        return 1.0 - a @ b.T


class DlibHOG(FaceModel):
    """Current SnapClass pipeline: dlib HOG detector + 68 landmarks + ResNet (128-D)."""
    name, label, metric, embedding_dim = "dlib_hog", "dlib HOG + ResNet (current app)", "euclidean", 128

    def __init__(self, cnn_detector: bool = False):
        import dlib
        import face_recognition_models as frm
        self.dlib = dlib
        self.cnn = cnn_detector
        if cnn_detector:
            self.name, self.label = "dlib_cnn", "dlib CNN detector + ResNet"
            self.detector = dlib.cnn_face_detection_model_v1(frm.cnn_face_detector_model_location())
        else:
            self.detector = dlib.get_frontal_face_detector()
        self.sp = dlib.shape_predictor(frm.pose_predictor_model_location())
        self.facerec = dlib.face_recognition_model_v1(frm.face_recognition_model_location())

    def __call__(self, image_rgb):
        dets = self.detector(image_rgb, 1)  # same upsampling as the app
        faces = []
        for d in dets:
            rect = d.rect if self.cnn else d
            shape = self.sp(image_rgb, rect)
            emb = np.array(self.facerec.compute_face_descriptor(image_rgb, shape, 1))
            faces.append(Face((rect.left(), rect.top(), rect.right(), rect.bottom()), emb))
        return faces


class InsightFace(FaceModel):
    """InsightFace model packs: SCRFD detector + ArcFace recognition (512-D).

    buffalo_l: SCRFD-10G + ResNet-100 ArcFace (large, most accurate)
    buffalo_s: SCRFD-500M + MobileFaceNet ArcFace (small, fast)
    """
    metric, embedding_dim = "cosine", 512

    def __init__(self, pack: str = "buffalo_l", det_size: int = 640):
        from insightface.app import FaceAnalysis
        self.name = f"insightface_{pack}"
        size = "large" if pack == "buffalo_l" else "small"
        self.label = f"InsightFace ArcFace ({size}, {pack})"
        self.app = FaceAnalysis(name=pack, root=str(MODEL_CACHE / "insightface"),
                                providers=["CPUExecutionProvider"],
                                allowed_modules=["detection", "recognition"])
        self.app.prepare(ctx_id=-1, det_size=(det_size, det_size))

    def __call__(self, image_rgb):
        faces = self.app.get(np.ascontiguousarray(image_rgb[:, :, ::-1]))  # InsightFace expects BGR
        return [Face(tuple(float(v) for v in f.bbox), np.asarray(f.normed_embedding, dtype=np.float64))
                for f in faces]


class FaceNet(FaceModel):
    """facenet-pytorch: MTCNN detector + InceptionResnetV1 trained on VGGFace2 (512-D)."""
    name, label, metric, embedding_dim = "facenet", "FaceNet (MTCNN + InceptionResnetV1)", "cosine", 512

    def __init__(self):
        import torch
        torch.hub.set_dir(str(MODEL_CACHE / "torch"))
        from facenet_pytorch import MTCNN, InceptionResnetV1
        self.torch = torch
        self.mtcnn = MTCNN(keep_all=True, device="cpu")
        self.net = InceptionResnetV1(pretrained="vggface2").eval()

    def __call__(self, image_rgb):
        from PIL import Image
        pil = Image.fromarray(image_rgb)
        boxes, _ = self.mtcnn.detect(pil)
        if boxes is None:
            return []
        crops = self.mtcnn.extract(pil, boxes, save_path=None)
        with self.torch.no_grad():
            embs = self.net(crops).numpy().astype(np.float64)
        return [Face(tuple(float(v) for v in b), e) for b, e in zip(boxes, embs)]


def get_model(name: str) -> FaceModel:
    factories = {
        "dlib_hog": lambda: DlibHOG(),
        "dlib_cnn": lambda: DlibHOG(cnn_detector=True),
        "insightface_buffalo_l": lambda: InsightFace("buffalo_l"),
        "insightface_buffalo_s": lambda: InsightFace("buffalo_s"),
        "facenet": lambda: FaceNet(),
    }
    return factories[name]()


RECOGNITION_MODELS = ["dlib_hog", "facenet", "insightface_buffalo_s", "insightface_buffalo_l"]
