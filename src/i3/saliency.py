"""Facial saliency / relevancy mask  W_PIR  (values in [floor, 1]).

Where the adversarial budget is spent:
    eyes, lips          1.00   (identity + expression – what face-swap / morph models key on)
    brows, nose         0.85-0.9
    jaw/face contour    0.80   (a band around the face oval – geometry cue for morphing)
    face interior       0.55
    hair / neck halo    0.30
    background          floor (default 0.10)
Inside the face the weight is further modulated by edge strength (Sobel gradient),
so perturbation energy concentrates on high-frequency structure instead of smooth
skin – the main trick for avoiding blotches on darker / smoother skin.

Illumination inclusivity: detection and gradients are computed on a CLAHE-normalised
copy of the image (LAB lightness), which helps Haar/MediaPipe on low-light photos and
darker skin tones (Fitzpatrick IV-VI / Monk 6-10). The *output* image is never equalised.

Back-ends (all optional, picked by "auto" in this order):
  * "mediapipe" – FaceLandmarker (478 landmarks). Needs models/face_landmarker.task. Most precise.
  * "yunet"     – OpenCV DNN face detector, 5 landmarks, 230 KB ONNX
                  (models/face_detection_yunet_2023mar.onnx). Good speed/robustness trade-off.
  * "haar"      – OpenCV Haar cascades shipped inside opencv 4.x – zero download, coarsest.
Run `python scripts/download_models.py` to fetch the model files.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

DEFAULT_WEIGHTS = {
    "eyes": 1.0,
    "lips": 1.0,
    "brows": 0.9,
    "nose": 0.85,
    "contour": 0.8,
    "face": 0.55,
    "halo": 0.3,
}


@dataclass
class Face:
    box: tuple[int, int, int, int]  # x, y, w, h
    regions: dict[str, list[np.ndarray]] = field(default_factory=dict)  # name -> polygons (N,2) int32
    source: str = "haar"


def normalize_illumination(rgb: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """CLAHE on LAB lightness. rgb uint8 HxWx3 -> uint8 HxWx3. Used for analysis only."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    lab[..., 0] = clahe.apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


# --------------------------------------------------------------------------- detection
def _ellipse_poly(cx, cy, ax, ay, n=40) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([cx + ax * np.cos(t), cy + ay * np.sin(t)], 1).astype(np.int32)


def _regions_from_points(box, eyes_xy, mouth_xy, nose_xy, eye_r) -> dict[str, list[np.ndarray]]:
    x, y, fw, fh = box
    mx, my = mouth_xy
    return {
        "eyes": [_ellipse_poly(cx, cy, eye_r * 1.3, eye_r * 0.8) for cx, cy in eyes_xy],
        "brows": [_ellipse_poly(cx, cy - 1.4 * eye_r, eye_r * 1.5, eye_r * 0.45) for cx, cy in eyes_xy],
        "nose": [_ellipse_poly(nose_xy[0], nose_xy[1], 0.12 * fw, 0.14 * fh)],
        "lips": [_ellipse_poly(mx, my, 0.2 * fw, 0.075 * fh)],
        "face": [_ellipse_poly(x + 0.5 * fw, y + 0.52 * fh, 0.5 * fw, 0.62 * fh)],
    }


def _haar_faces(rgb: np.ndarray) -> list[Face]:
    if not hasattr(cv2, "CascadeClassifier"):  # OpenCV 5 moved cascades to contrib
        log.warning("cv2.CascadeClassifier unavailable (OpenCV %s); install opencv<5 or use yunet.", cv2.__version__)
        return []
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    root = Path(cv2.data.haarcascades)
    face_cc = cv2.CascadeClassifier(str(root / "haarcascade_frontalface_default.xml"))
    eye_cc = cv2.CascadeClassifier(str(root / "haarcascade_eye.xml"))
    min_side = max(24, int(0.08 * min(h, w)))
    boxes = face_cc.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(min_side, min_side))
    faces: list[Face] = []
    for x, y, fw, fh in boxes:
        x, y, fw, fh = int(x), int(y), int(fw), int(fh)
        # geometric priors relative to the Haar box (box spans ~brows to chin)
        eye_y, eye_r = y + 0.38 * fh, 0.11 * fw
        eyes_xy = [(x + 0.31 * fw, eye_y), (x + 0.69 * fw, eye_y)]
        roi = gray[y : y + int(0.6 * fh), x : x + fw]
        found = eye_cc.detectMultiScale(roi, 1.1, 5, minSize=(max(8, fw // 10),) * 2)
        if len(found) >= 2:
            found = sorted(found, key=lambda e: -e[2] * e[3])[:2]
            eyes_xy = [(x + ex + ew / 2, y + ey + eh / 2) for ex, ey, ew, eh in found]
            eye_r = float(np.mean([ew for _, _, ew, _ in found])) * 0.55
        regions = _regions_from_points(
            (x, y, fw, fh), eyes_xy, (x + 0.5 * fw, y + 0.8 * fh), (x + 0.5 * fw, y + 0.6 * fh), eye_r
        )
        faces.append(Face(box=(x, y, fw, fh), regions=regions, source="haar"))
    return faces


def _yunet_faces(rgb: np.ndarray, model_path: Path, score: float = 0.7) -> list[Face]:
    h, w = rgb.shape[:2]
    det = cv2.FaceDetectorYN.create(str(model_path), "", (w, h), score, 0.3, 50)
    det.setInputSize((w, h))
    _, rows = det.detect(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    faces: list[Face] = []
    for r in rows if rows is not None else []:
        x, y, fw, fh = (int(v) for v in r[:4])
        re, le, nose, rm, lm = r[4:6], r[6:8], r[8:10], r[10:12], r[12:14]
        eye_r = 0.22 * float(np.linalg.norm(le - re))
        mouth = ((rm[0] + lm[0]) / 2, (rm[1] + lm[1]) / 2)
        regions = _regions_from_points((x, y, fw, fh), [tuple(re), tuple(le)], mouth, tuple(nose), eye_r)
        faces.append(Face(box=(x, y, fw, fh), regions=regions, source="yunet"))
    return faces


def _mediapipe_faces(rgb: np.ndarray, model_path: Path, max_faces: int = 4) -> list[Face]:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision

    opts = vision.FaceLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=str(model_path)), num_faces=max_faces
    )
    conn = vision.FaceLandmarksConnections
    groups = {
        "eyes": [conn.FACE_LANDMARKS_LEFT_EYE, conn.FACE_LANDMARKS_RIGHT_EYE],
        "brows": [conn.FACE_LANDMARKS_LEFT_EYEBROW, conn.FACE_LANDMARKS_RIGHT_EYEBROW],
        "lips": [conn.FACE_LANDMARKS_LIPS],
        "nose": [conn.FACE_LANDMARKS_NOSE],
        "face": [conn.FACE_LANDMARKS_FACE_OVAL],
    }
    h, w = rgb.shape[:2]
    with vision.FaceLandmarker.create_from_options(opts) as lm:
        res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)))
    faces: list[Face] = []
    for pts in res.face_landmarks:
        xy = np.array([[p.x * w, p.y * h] for p in pts], dtype=np.float32)
        regions: dict[str, list[np.ndarray]] = {}
        for name, conns in groups.items():
            polys = []
            for cset in conns:
                idx = sorted({c.start for c in cset} | {c.end for c in cset})
                polys.append(cv2.convexHull(xy[idx]).reshape(-1, 2).astype(np.int32))
            regions[name] = polys
        x0, y0 = xy.min(0)
        x1, y1 = xy.max(0)
        faces.append(Face(box=(int(x0), int(y0), int(x1 - x0), int(y1 - y0)), regions=regions, source="mediapipe"))
    return faces


MEDIAPIPE_MODEL = "face_landmarker.task"
YUNET_MODEL = "face_detection_yunet_2023mar.onnx"


def detect_faces(
    rgb: np.ndarray,
    backend: str = "auto",
    models_dir: str | Path = "models",
    illumination_normalize: bool = True,
) -> list[Face]:
    """Detect faces on an illumination-normalised copy. Returns [] if none found."""
    analysis = normalize_illumination(rgb) if illumination_normalize else rgb
    models_dir = Path(models_dir)
    order = ["mediapipe", "yunet", "haar"] if backend == "auto" else [backend]
    for name in order:
        try:
            if name == "mediapipe":
                path = models_dir / MEDIAPIPE_MODEL
                if not path.exists():
                    if backend == name:
                        raise FileNotFoundError(f"{path} missing - run scripts/download_models.py")
                    continue
                faces = _mediapipe_faces(analysis, path)
            elif name == "yunet":
                path = models_dir / YUNET_MODEL
                if not path.exists():
                    if backend == name:
                        raise FileNotFoundError(f"{path} missing - run scripts/download_models.py")
                    continue
                faces = _yunet_faces(analysis, path)
            elif name == "haar":
                faces = _haar_faces(analysis)
            else:
                raise ValueError(f"unknown face backend {name!r}")
        except FileNotFoundError:
            raise
        except Exception as e:  # pragma: no cover - depends on local install
            if backend == name:
                raise
            log.warning("%s face backend failed (%s); trying next.", name, e)
            continue
        if faces:
            return faces
    return []


# --------------------------------------------------------------------------- mask
def _edge_strength(rgb_norm: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(rgb_norm, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.GaussianBlur(np.sqrt(gx**2 + gy**2), (0, 0), 1.5)
    return np.clip(mag / (np.percentile(mag, 95) + 1e-6), 0, 1)


def build_wpir(
    rgb: np.ndarray,
    faces: list[Face],
    floor: float = 0.10,
    weights: dict[str, float] | None = None,
    edge_mix: float = 0.4,
) -> np.ndarray:
    """Return W_PIR as float32 HxW in [floor, 1]."""
    wts = {**DEFAULT_WEIGHTS, **(weights or {})}
    h, w = rgb.shape[:2]
    edges = _edge_strength(normalize_illumination(rgb))
    m = np.zeros((h, w), np.float32)

    def paint(polys, value, thickness=None):
        layer = np.zeros((h, w), np.uint8)
        for p in polys:
            if thickness:
                cv2.polylines(layer, [p], True, 255, thickness)
            else:
                cv2.fillPoly(layer, [p], 255)
        np.maximum(m, (layer > 0) * value, out=m)

    if not faces:
        log.warning("No face found - using centre-weighted + edge mask (weaker protection).")
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        centre = np.exp(-(((xx - w / 2) / (0.35 * w)) ** 2 + ((yy - h / 2) / (0.35 * h)) ** 2))
        m = 0.5 * centre
    for f in faces:
        fw = f.box[2]
        face_polys = f.regions.get("face", [])
        # halo: dilated face oval (hair line, ears, neck)
        halo = []
        for p in face_polys:
            c = p.mean(0)
            halo.append(((p - c) * 1.35 + c).astype(np.int32))
        paint(halo, wts["halo"])
        paint(face_polys, wts["face"])
        paint(face_polys, wts["contour"], thickness=max(2, int(0.06 * fw)))
        for name in ("nose", "brows", "eyes", "lips"):
            paint(f.regions.get(name, []), wts[name])

    face_area = m > 0
    m = np.where(face_area, m * ((1 - edge_mix) + edge_mix * edges), m)
    sigma = max(1.0, 0.01 * min(h, w))
    m = cv2.GaussianBlur(m, (0, 0), sigma)
    m = floor + (1 - floor) * np.clip(m / (m.max() + 1e-6), 0, 1)
    return m.astype(np.float32)
