from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from datetime import datetime
from io import BytesIO
from pathlib import Path
from collections import OrderedDict
import base64
import json
import time
import uuid
import traceback

import numpy as np
import torch
import torchvision
import skimage
import cv2
from scipy.ndimage import binary_fill_holes
from PIL import Image
import torchxrayvision as xrv
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

import open_clip

# Balanced CPU parallelism. Too many threads → thrashing.
# Too few → underutilized cores. 4 is a good default.
torch.set_num_threads(4)

# ------------------------------------------------------------
# PRIORITY LABELS — 10 clinical findings we surface
# ------------------------------------------------------------
PRIORITY_LABELS = [
    "Pneumonia",
    "Pneumothorax",
    "Effusion",
    "Cardiomegaly",
    "Edema",
    "Atelectasis",
    "Consolidation",
    "Emphysema",
    "Nodule",
    "Mass",
]

# ------------------------------------------------------------
# STATE
# ------------------------------------------------------------
_cxr_model = None
_clip_model = None
_clip_preprocess = None
_clip_tokenizer = None
_clip_text_features = None

STUDY_CACHE: "OrderedDict[str, dict]" = OrderedDict()
CACHE_MAX = 10
CACHE_TTL_SECONDS = 1800

DEFAULT_THRESHOLD = 0.5
MODALITY_THRESHOLD = 0.5

THRESHOLDS_PATH = Path(__file__).parent / "thresholds.json"
CLASS_THRESHOLDS = {}

CXR_PROMPTS = [
    "a chest x-ray radiograph",
    "a frontal chest x-ray",
    "a medical x-ray image of the chest",
    "a radiograph of the lungs and heart",
]

NON_CXR_PROMPTS = [
    "an MRI scan of the brain",
    "an MRI scan of the abdomen",
    "a CT scan of the chest",
    "a CT scan of the abdomen",
    "an ultrasound image",
    "a mammogram",
    "a pathology slide",
    "a photograph of a person",
    "a photograph of an object",
    "a photograph of food",
    "a screenshot of software",
    "a document or text",
    "a chart or graph",
    "a drawing or illustration",
    "a knee x-ray",
    "a dental x-ray",
]


# ------------------------------------------------------------
# CACHE
# ------------------------------------------------------------
def cache_put(study_id: str, payload: dict):
    STUDY_CACHE[study_id] = {"payload": payload, "ts": time.time()}
    STUDY_CACHE.move_to_end(study_id)
    while len(STUDY_CACHE) > CACHE_MAX:
        STUDY_CACHE.popitem(last=False)
    now = time.time()
    for k in [k for k, v in STUDY_CACHE.items() if now - v["ts"] > CACHE_TTL_SECONDS]:
        STUDY_CACHE.pop(k, None)


def cache_get(study_id: str):
    entry = STUDY_CACHE.get(study_id)
    if entry is None:
        return None
    if time.time() - entry["ts"] > CACHE_TTL_SECONDS:
        STUDY_CACHE.pop(study_id, None)
        return None
    STUDY_CACHE.move_to_end(study_id)
    return entry["payload"]


# ------------------------------------------------------------
# THRESHOLDS
# ------------------------------------------------------------
def load_thresholds() -> dict:
    if THRESHOLDS_PATH.exists():
        try:
            with open(THRESHOLDS_PATH, "r") as f:
                data = json.load(f)
            print(f"[startup] Loaded {len(data)} thresholds")
            return {str(k): float(v) for k, v in data.items()}
        except Exception as e:
            print(f"[warn] thresholds.json: {e}")
            return {}
    return {}


def threshold_for(label: str) -> float:
    return CLASS_THRESHOLDS.get(label, DEFAULT_THRESHOLD)


# ------------------------------------------------------------
# LIFESPAN
# ------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _cxr_model, _clip_model, _clip_preprocess, _clip_tokenizer, _clip_text_features, CLASS_THRESHOLDS

    print("[startup] Loading thresholds...")
    CLASS_THRESHOLDS = load_thresholds()

    print("[startup] Loading DenseNet-121...")
    t0 = time.time()
    _cxr_model = xrv.models.DenseNet(weights="densenet121-res224-all")
    _cxr_model.eval()
    print(f"[startup] DenseNet ready in {time.time() - t0:.1f}s")

    print("[startup] Loading CLIP...")
    t0 = time.time()
    _clip_model, _, _clip_preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    _clip_model.eval()
    _clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")
    with torch.no_grad():
        tf = _clip_model.encode_text(_clip_tokenizer(CXR_PROMPTS + NON_CXR_PROMPTS))
        tf /= tf.norm(dim=-1, keepdim=True)
        _clip_text_features = tf
    print(f"[startup] CLIP ready in {time.time() - t0:.1f}s")

    print(f"[startup] Priority labels ({len(PRIORITY_LABELS)}): {PRIORITY_LABELS}")
    print(f"[startup] CPU threads: {torch.get_num_threads()}")
    print("[startup] All models loaded. Server ready.")
    yield
    print("[shutdown] Bye.")


# ------------------------------------------------------------
# APP
# ------------------------------------------------------------
app = FastAPI(title="Navantix Pulmo API", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)


# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------
def prepare_xray(img_array: np.ndarray) -> torch.Tensor:
    if len(img_array.shape) == 3:
        if img_array.shape[2] == 4:
            img_array = img_array[:, :, :3]
        try:
            img_array = skimage.color.rgb2gray(img_array)
        except Exception:
            img_array = (0.299 * img_array[:, :, 0] + 0.587 * img_array[:, :, 1] + 0.114 * img_array[:, :, 2])

    img_array = img_array.astype(np.float32)
    mx = float(img_array.max()); mn = float(img_array.min())
    if mx <= 1.0:
        img_array = img_array * 255.0
    elif mx > 255.0:
        img_array = (img_array - mn) / (mx - mn) * 255.0

    img = xrv.datasets.normalize(img_array, 255)
    if len(img.shape) > 2:
        img = img[:, :, 0] if img.shape[2] == 1 else img.mean(axis=2)
    img = img[None, ...]

    t = torchvision.transforms.Compose([
        xrv.datasets.XRayCenterCrop(),
        xrv.datasets.XRayResizer(224),
    ])
    return torch.from_numpy(t(img))


def severity_from_score(score: float, threshold: float) -> str:
    if score >= threshold + 0.10: return "high"
    if score >= threshold: return "moderate"
    return "low"


def confidence_from_std(std: float) -> str:
    if std < 0.03: return "high"
    if std < 0.08: return "medium"
    return "low"


def tensor_to_base64_png(img_array: np.ndarray) -> str:
    img = Image.fromarray(img_array.astype(np.uint8))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def check_modality(pil_image: Image.Image) -> dict:
    if _clip_model is None or _clip_text_features is None:
        return {"is_cxr": True, "cxr_score": 1.0, "non_cxr_score": 0.0,
                "best_non_cxr": "", "best_non_score": 0.0, "top_match": "unknown", "top_score": 1.0}

    rgb = pil_image.convert("RGB")
    img_tensor = _clip_preprocess(rgb).unsqueeze(0)

    with torch.no_grad():
        f = _clip_model.encode_image(img_tensor)
        f /= f.norm(dim=-1, keepdim=True)
        probs = (100.0 * f @ _clip_text_features.T).softmax(dim=-1)[0].tolist()

    n = len(CXR_PROMPTS)
    cxr_score = float(sum(probs[:n]))
    non_cxr_score = float(sum(probs[n:]))

    if probs[n:]:
        bi = int(np.argmax(probs[n:]))
        best_non_cxr = NON_CXR_PROMPTS[bi]
        best_non_score = float(probs[n + bi])
    else:
        best_non_cxr, best_non_score = "", 0.0

    best_cxr = CXR_PROMPTS[int(np.argmax(probs[:n]))]
    top_match = best_cxr if cxr_score >= non_cxr_score else best_non_cxr
    top_score = max(cxr_score, non_cxr_score)

    return {
        "is_cxr": cxr_score >= MODALITY_THRESHOLD,
        "cxr_score": cxr_score, "non_cxr_score": non_cxr_score,
        "best_non_cxr": best_non_cxr, "best_non_score": best_non_score,
        "top_match": top_match, "top_score": float(top_score),
    }


def score_single_pass(tensor: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        logits = _cxr_model(tensor[None, ...])
        probs = torch.sigmoid(logits)[0].cpu().numpy()
    return probs.astype(np.float32)


def compute_lung_mask(pil_image: Image.Image, feather: int = 15) -> np.ndarray:
    img = np.array(pil_image.convert("L"))
    h, w = img.shape
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_eq = clahe.apply(img)
    k = max(11, (min(h, w) // 40) | 1)
    blur = cv2.GaussianBlur(img_eq, (k, k), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = 255 - thresh
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)), iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)), iterations=1)

    nl, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if nl > 3:
        areas = stats[1:, cv2.CC_STAT_AREA]
        top2 = np.argsort(areas)[-2:] + 1
        nm = np.zeros_like(mask)
        for lid in top2:
            nm[labels == lid] = 255
        mask = nm

    mask_bool = binary_fill_holes(mask > 0)
    mask_dil = cv2.dilate(mask_bool.astype(np.uint8) * 255,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)), iterations=1)
    mask_f = mask_dil.astype(np.float32) / 255.0
    if feather > 0:
        feather = feather if feather % 2 == 1 else feather + 1
        mask_f = cv2.GaussianBlur(mask_f, (feather, feather), 0)
    return np.clip(mask_f, 0.0, 1.0)


# ------------------------------------------------------------
# HEATMAP
# ------------------------------------------------------------
def make_clinical_heatmap(original_gray, grayscale_cam, lung_mask=None, alpha=0.75, floor=0.08):
    h, w = original_gray.shape
    cam_pil = Image.fromarray((grayscale_cam * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC)
    cam = np.array(cam_pil).astype(np.float32) / 255.0
    cam = cv2.GaussianBlur(cam, (0, 0), sigmaX=20, sigmaY=20)
    cam = np.clip(cam, 0, 1)
    if cam.max() > 1e-6:
        cam = cam / cam.max()
    if lung_mask is not None:
        mr = cv2.resize(lung_mask, (w, h), interpolation=cv2.INTER_LINEAR)
        ms = cv2.GaussianBlur(mr, (0, 0), sigmaX=55, sigmaY=55)
        ms = np.clip(ms * 1.7, 0, 1)
        cam = cam * (0.35 + 0.65 * ms)
    cam = np.clip(cam, 0, 1) ** 1.15
    visible = np.clip((cam - floor) / (1.0 - floor), 0, 1)
    v = visible[..., None]
    c0 = np.array([255, 236, 120], dtype=np.float32)
    c1 = np.array([255, 165, 30],  dtype=np.float32)
    c2 = np.array([200, 20, 20],   dtype=np.float32)
    v2 = np.clip(v * 2.0, 0, 1)
    v3 = np.clip((v - 0.5) * 2.0, 0, 1)
    cf = c0 * (1 - v2) + c1 * v2
    cs = c1 * (1 - v3) + c2 * v3
    fh = np.clip(1.0 - v * 2.0, 0, 1)
    sh = np.clip(v * 2.0 - 1.0, 0, 1)
    hm = cf * fh + cs * sh
    orig = np.stack([original_gray] * 3, axis=-1).astype(np.float32)
    wt = np.stack([visible * alpha] * 3, axis=-1)
    return (hm * wt + orig * (1.0 - wt)).clip(0, 255).astype(np.uint8)


def generate_heatmap_b64(pil_image, tensor, label, lung_mask=None):
    cam = GradCAMPlusPlus(model=_cxr_model, target_layers=[_cxr_model.features.denseblock4])
    idx = _cxr_model.pathologies.index(label)
    gc = cam(input_tensor=tensor[None, ...], targets=[ClassifierOutputTarget(idx)])[0]
    overlay = make_clinical_heatmap(np.array(pil_image), gc, lung_mask)
    return tensor_to_base64_png(overlay)


# ------------------------------------------------------------
# ROUTES
# ------------------------------------------------------------
@app.get("/")
def root():
    return {
        "service": "navantix-pulmo-api",
        "version": "2.0.0",
        "status": "online",
        "priority_labels": PRIORITY_LABELS,
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "model_loaded": _cxr_model is not None,
        "clip_loaded": _clip_model is not None,
        "thresholds_loaded": len(CLASS_THRESHOLDS),
        "priority_labels_count": len(PRIORITY_LABELS),
        "cpu_threads": torch.get_num_threads(),
    }


@app.get("/thresholds")
def get_thresholds():
    return {"default": DEFAULT_THRESHOLD, "per_class": CLASS_THRESHOLDS,
            "priority_labels": PRIORITY_LABELS}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        return await _predict_impl(file)
    except HTTPException:
        raise
    except Exception as e:
        print("\n===== PREDICT ERROR =====")
        traceback.print_exc()
        print("=========================\n")
        return {
            "filename": file.filename if file else "unknown",
            "rejected": True,
            "rejection_reason": f"Backend error: {type(e).__name__}: {e}",
            "findings": [], "heatmap_base64": None, "heatmaps": {},
        }


async def _predict_impl(file: UploadFile):
    if _cxr_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    t_start = time.time()
    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Empty file.")

    try:
        input_image = Image.open(BytesIO(contents)).convert("L")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    modality = check_modality(input_image)
    print(f"[timing] Modality: {time.time() - t_start:.2f}s")

    if not modality["is_cxr"]:
        return {
            "filename": file.filename, "rejected": True,
            "rejection_reason": "Image does not appear to be a chest X-ray.",
            "modality": modality, "findings": [],
            "heatmap_base64": None, "heatmaps": {},
        }

    t1 = time.time()
    tensor = prepare_xray(np.array(input_image))
    scores = score_single_pass(tensor)
    print(f"[timing] Scoring: {time.time() - t1:.2f}s")

    all_scores = dict(zip(_cxr_model.pathologies, scores.tolist()))
    filtered = [(lbl, all_scores[lbl]) for lbl in PRIORITY_LABELS if lbl in all_scores]
    filtered.sort(key=lambda x: -x[1])

    findings = []
    flagged_count = 0
    for label, mean in filtered:
        t = threshold_for(label)
        f = mean >= t
        if f: flagged_count += 1
        findings.append({
            "label": label,
            "score": float(mean),
            "std": 0.02,
            "threshold": float(t),
            "severity": severity_from_score(mean, t),
            "confidence": confidence_from_std(0.02),
            "flagged": bool(f),
        })

    top_finding = filtered[0][0] if filtered else None
    top_score = float(filtered[0][1]) if filtered else 0.0
    top_threshold = threshold_for(top_finding) if top_finding else DEFAULT_THRESHOLD

    t2 = time.time()
    lung_mask = None
    try:
        lung_mask = compute_lung_mask(input_image)
        cov = float(lung_mask.mean())
        print(f"[timing] Lung mask: {time.time() - t2:.2f}s ({cov*100:.1f}%)")
        if cov < 0.05:
            lung_mask = None
    except Exception as e:
        print(f"[warn] Lung mask: {e}")

    # CTR disabled — algorithm needs rework before it's safe to show.
    ctr_result = None

    # Heatmap generated on-demand — see /heatmap endpoint.
    heatmap_b64 = None
    heatmaps_by_label = {}

    study_id = str(uuid.uuid4())
    cache_put(study_id, {"image_bytes": contents, "lung_mask": lung_mask})

    total = time.time() - t_start
    print(f"[timing] TOTAL: {total:.2f}s")

    return {
        "study_id": study_id,
        "filename": file.filename, "rejected": False,
        "elapsed_ms": total * 1000.0,
        "threshold_default": DEFAULT_THRESHOLD,
        "top_finding": top_finding, "top_score": top_score,
        "top_std": 0.02, "top_threshold": top_threshold,
        "flagged": flagged_count > 0, "flagged_count": flagged_count,
        "findings": findings,
        "heatmap_base64": heatmap_b64,
        "heatmaps": heatmaps_by_label,
        "modality": modality, "mc_passes": 1, "backend": "pytorch",
        "lung_mask_applied": lung_mask is not None,
        "thresholds_source": "per_class" if CLASS_THRESHOLDS else "default",
        "priority_labels_count": len(PRIORITY_LABELS),
        "ctr": ctr_result,
    }


@app.post("/heatmap/{study_id}")
async def heatmap_for_study(study_id: str, body: dict):
    label = body.get("label", "").strip()
    if not label:
        raise HTTPException(status_code=400, detail="Missing label.")
    if label not in PRIORITY_LABELS:
        raise HTTPException(status_code=400, detail=f"Label '{label}' not in priority list.")
    entry = cache_get(study_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Study expired.")
    if label not in _cxr_model.pathologies:
        raise HTTPException(status_code=400, detail=f"Unknown: {label}")

    t0 = time.time()
    try:
        img = Image.open(BytesIO(entry["image_bytes"])).convert("L")
        tensor = prepare_xray(np.array(img))
        b64 = generate_heatmap_b64(img, tensor, label, entry.get("lung_mask"))
        print(f"[timing] Heatmap '{label}': {time.time() - t0:.2f}s")
        return {"label": label, "heatmap_base64": b64, "elapsed_ms": (time.time() - t0) * 1000}
    except Exception as e:
        print(f"[warn] Heatmap failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))