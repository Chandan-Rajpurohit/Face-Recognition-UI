# app.py
import base64
import os
import sqlite3
import zlib
from typing import List, Optional, Tuple

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ML deps
import mediapipe as mp
from insightface.app import FaceAnalysis

# ---------------- Config ----------------
DB_PATH = "faces.db"
SIMILARITY_THRESHOLD = 0.55
MODEL_NAME = "buffalo_s"  # lighter for CPU-only laptops
TARGET_MAX_WIDTH = 640    # backend will downscale incoming frames if larger

ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

# ---------------- FastAPI ----------------
app = FastAPI(title="Local Face API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- DB ----------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS faces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            embedding BLOB NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def insert_face(name: str, embedding: np.ndarray):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    comp = zlib.compress(embedding.astype(np.float16).tobytes())
    cur.execute("INSERT INTO faces (name, embedding) VALUES (?, ?)", (name, comp))
    conn.commit()
    conn.close()

def load_faces_matrix() -> Tuple[List[str], Optional[np.ndarray]]:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT name, embedding FROM faces")
    rows = cur.fetchall()
    conn.close()
    if not rows:
        return [], None
    names, embs = [], []
    for name, blob in rows:
        em = np.frombuffer(zlib.decompress(blob), dtype=np.float16).astype(np.float32)
        n = np.linalg.norm(em)
        if n > 1e-9:
            names.append(name)
            embs.append(em / n)
    if not embs:
        return [], None
    return names, np.vstack(embs)

# ---------------- Models (loaded once) ----------------
mp_face_detection = None
face_detector = None
face_app = None
names_cache: List[str] = []
emb_matrix_cache: Optional[np.ndarray] = None

def load_models():
    global mp_face_detection, face_detector, face_app
    mp_face_detection = mp.solutions.face_detection
    face_detector = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.6)
    face_app = FaceAnalysis(name=MODEL_NAME, providers=['CPUExecutionProvider'])
    face_app.prepare(ctx_id=0, det_size=(320, 320))

def reload_cache():
    global names_cache, emb_matrix_cache
    names, embs = load_faces_matrix()
    names_cache, emb_matrix_cache = names, embs
    print(f"[CACHE] {len(names_cache)} faces loaded")

# ---------------- Utils ----------------
def decode_image_b64(data: str) -> np.ndarray:
    # supports "data:image/jpeg;base64,..." or raw base64
    if "," in data and data.strip().startswith("data:"):
        data = data.split(",", 1)[1]
    img_bytes = base64.b64decode(data)
    img_array = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Invalid image")
    h, w = img.shape[:2]
    if w > TARGET_MAX_WIDTH:
        scale = TARGET_MAX_WIDTH / float(w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img

def bbox_to_xyxy(bboxC, iw, ih) -> Tuple[int, int, int, int]:
    x1 = max(0, int(bboxC.xmin * iw))
    y1 = max(0, int(bboxC.ymin * ih))
    w = int(bboxC.width * iw)
    h = int(bboxC.height * ih)
    x2 = min(iw, x1 + w)
    y2 = min(ih, y1 + h)
    return x1, y1, x2, y2

def detect_faces(image_bgr: np.ndarray):
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    res = face_detector.process(rgb)
    return getattr(res, "detections", None)

def crop_embed(image_bgr: np.ndarray, x1, y1, x2, y2) -> Optional[np.ndarray]:
    face_crop = image_bgr[y1:y2, x1:x2]
    if face_crop.size == 0:
        return None
    try:
        faces = face_app.get(face_crop, max_num=1)
    except TypeError:
        faces = face_app.get(face_crop)
    if not faces:
        return None
    emb = getattr(faces[0], "embedding", None)
    if emb is None:
        emb = getattr(faces[0], "normed_embedding", None)
    if emb is None:
        return None
    return emb.astype(np.float32)

def match_embedding(emb: np.ndarray) -> Tuple[str, float]:
    if emb_matrix_cache is None or not len(names_cache):
        return "Unknown", 0.0
    n = np.linalg.norm(emb)
    if n < 1e-9:
        return "Unknown", 0.0
    emb_norm = emb / n
    sims = emb_matrix_cache @ emb_norm  # vectorized cosine sim
    idx = int(np.argmax(sims))
    return names_cache[idx], float(sims[idx])

# ---------------- Schemas ----------------
class EnrollRequest(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    images: List[str] = Field(..., description="Array of base64 frames (JPEG/WebP)")
    min_samples: int = 6

class EnrollResponse(BaseModel):
    ok: bool
    collected: int
    message: str

class RecognizeRequest(BaseModel):
    image: str

class BBox(BaseModel):
    x1: int; y1: int; x2: int; y2: int

class RecognizeItem(BaseModel):
    bbox: BBox
    label: str
    score: float

class RecognizeResponse(BaseModel):
    width: int
    height: int
    results: List[RecognizeItem]

# ---------------- Startup ----------------
@app.on_event("startup")
def on_startup():
    init_db()
    load_models()
    reload_cache()
    print("[STARTED] Local Face API ready.")

# ---------------- Routes ----------------
@app.post("/enroll", response_model=EnrollResponse)
def enroll(req: EnrollRequest):
    name = req.name.strip()
    if not name:
        raise HTTPException(400, "Name required")
    if len(req.images) < req.min_samples:
        raise HTTPException(400, f"Send at least {req.min_samples} frames")

    embeddings = []
    for data in req.images:
        img = decode_image_b64(data)
        ih, iw = img.shape[:2]
        detections = detect_faces(img)
        if not detections:
            continue
        # largest face in frame
        best = None; best_area = 0
        for det in detections:
            bboxC = det.location_data.relative_bounding_box
            x1, y1, x2, y2 = bbox_to_xyxy(bboxC, iw, ih)
            area = max(0, x2 - x1) * max(0, y2 - y1)
            if area > best_area:
                best_area = area; best = (x1, y1, x2, y2)
        if best is None:
            continue
        emb = crop_embed(img, *best)
        if emb is not None:
            embeddings.append(emb)

    if len(embeddings) < req.min_samples:
        return EnrollResponse(ok=False, collected=len(embeddings),
                              message=f"Only {len(embeddings)} samples. Try better lighting / closer framing.")

    avg_embedding = np.mean(embeddings, axis=0)
    insert_face(name, avg_embedding)
    reload_cache()
    return EnrollResponse(ok=True, collected=len(embeddings),
                          message=f"Enrollment complete for '{name}'")

@app.post("/recognize", response_model=RecognizeResponse)
def recognize(req: RecognizeRequest):
    img = decode_image_b64(req.image)
    ih, iw = img.shape[:2]
    detections = detect_faces(img)
    out: List[RecognizeItem] = []
    if detections:
        for det in detections:
            bboxC = det.location_data.relative_bounding_box
            x1, y1, x2, y2 = bbox_to_xyxy(bboxC, iw, ih)
            emb = crop_embed(img, x1, y1, x2, y2)
            if emb is None:
                continue
            name, score = match_embedding(emb)
            label = name if score >= SIMILARITY_THRESHOLD else "Unknown"
            out.append(RecognizeItem(bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2),
                                     label=label, score=score))
    return RecognizeResponse(width=iw, height=ih, results=out)