from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from datetime import datetime
from io import BytesIO
import base64
import time
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
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

import open_clip

# ------------------------------------------------------------
# GLOBAL STATE
# ------------------------------------------------------------
_cxr_model = None
_clip_model = None
_clip_preprocess = None
_clip_tokenizer = None
_clip_text_features = None

CLINICAL_THRESHOLD = 0.5
MODALITY_THRESHOLD = 0.5
HEATMAP_TOP_N = 5
MC_PASSES = 10
MC_DROPOUT_P = 0.3

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
# LIFESPAN
# ------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _cxr_model, _clip_model, _clip_preprocess, _clip_tokenizer, _clip_text_features

    print("[startup] Loading Navantix Pulmo CXR model...")
    t0 = time.time()
    _cxr_model = xrv.models.DenseNet(weights="densenet121-res224-all")
    _cxr_model.eval()
    print(f"[startup] CXR model ready in {time.time() - t0:.1f}s")

    print("[startup] Loading CLIP modality checker...")
    t0 = time.time()
    _clip_model, _, _clip_preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    _clip_model.eval()
    _clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")

    all_prompts = CXR_PROMPTS + NON_CXR_PROMPTS
    with torch.no_grad():
        text_tokens = _clip_tokenizer(all_prompts)
        tf = _clip_model.encode_text(text_tokens)
        tf /= tf.norm(dim=-1, keepdim=True)
        _clip_text_features = tf
    print(f"[startup] CLIP ready in {time.time() - t0:.1f}s")

    print("[startup] All models loaded. Server ready.")
    yield
    print("[shutdown] Bye.")


# ------------------------------------------------------------
# APP
# ------------------------------------------------------------
app = FastAPI(
    title="Navantix Pulmo API",
    version="1.0.0",
    description="AI-assisted chest radiograph analysis — research prototype.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------
# HELPERS — IMAGE PREPROCESSING
# ------------------------------------------------------------
def prepare_xray(img_array: np.ndarray) -> torch.Tensor:
    if len(img_array.shape) == 3:
        if img_array.shape[2] == 4:
            img_array = img_array[:, :, :3]
        try:
            img_array = skimage.color.rgb2gray(img_array)
        except Exception:
            img_array = (0.299 * img_array[:, :, 0]
                         + 0.587 * img_array[:, :, 1]
                         + 0.114 * img_array[:, :, 2])

    img_array = img_array.astype(np.float32)
    mx = float(img_array.max())
    mn = float(img_array.min())
    if mx <= 1.0:
        img_array = img_array * 255.0
    elif mx > 255.0:
        img_array = (img_array - mn) / (mx - mn) * 255.0

    img = xrv.datasets.normalize(img_array, 255)
    if len(img.shape) > 2:
        img = img[:, :, 0] if img.shape[2] == 1 else img.mean(axis=2)
    img = img[None, ...]

    transform = torchvision.transforms.Compose([
        xrv.datasets.XRayCenterCrop(),
        xrv.datasets.XRayResizer(224),
    ])
    return torch.from_numpy(transform(img))


def severity_from_score(score: float) -> str:
    if score >= CLINICAL_THRESHOLD + 0.1:
        return "high"
    if score >= CLINICAL_THRESHOLD:
        return "moderate"
    return "low"


def confidence_from_std(std: float) -> str:
    if std < 0.03:
        return "high"
    if std < 0.08:
        return "medium"
    return "low"


def tensor_to_base64_png(img_array: np.ndarray) -> str:
    img = Image.fromarray(img_array.astype(np.uint8))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ------------------------------------------------------------
# HELPERS — MODALITY CHECK (CLIP)
# ------------------------------------------------------------
def check_modality(pil_image: Image.Image) -> dict:
    if _clip_model is None or _clip_text_features is None:
        return {
            "is_cxr": True,
            "cxr_score": 1.0,
            "non_cxr_score": 0.0,
            "best_non_cxr": "",
            "best_non_score": 0.0,
            "top_match": "unknown",
            "top_score": 1.0,
        }

    rgb = pil_image.convert("RGB")
    img_tensor = _clip_preprocess(rgb).unsqueeze(0)

    with torch.no_grad():
        image_features = _clip_model.encode_image(img_tensor)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        logits = (100.0 * image_features @ _clip_text_features.T)
        probs = logits.softmax(dim=-1)[0].tolist()

    n_cxr = len(CXR_PROMPTS)
    cxr_probs = probs[:n_cxr]
    non_cxr_probs = probs[n_cxr:]

    cxr_score = float(sum(cxr_probs))
    non_cxr_score = float(sum(non_cxr_probs))

    if non_cxr_probs:
        best_non_idx = int(np.argmax(non_cxr_probs))
        best_non_cxr = NON_CXR_PROMPTS[best_non_idx]
        best_non_score = float(non_cxr_probs[best_non_idx])
    else:
        best_non_cxr = ""
        best_non_score = 0.0

    best_cxr_idx = int(np.argmax(cxr_probs))
    best_cxr = CXR_PROMPTS[best_cxr_idx]

    top_match = best_cxr if cxr_score >= non_cxr_score else best_non_cxr
    top_score = max(cxr_score, non_cxr_score)

    return {
        "is_cxr": cxr_score >= MODALITY_THRESHOLD,
        "cxr_score": cxr_score,
        "non_cxr_score": non_cxr_score,
        "best_non_cxr": best_non_cxr,
        "best_non_score": best_non_score,
        "top_match": top_match,
        "top_score": float(top_score),
    }


# ------------------------------------------------------------
# HELPERS — MC DROPOUT
# ------------------------------------------------------------
def mc_dropout_predict(model, tensor, n_passes: int = MC_PASSES, p: float = MC_DROPOUT_P):
    import torch.nn.functional as F

    predictions = []

    def _dropout_hook(module, inputs):
        x = inputs[0]
        return (F.dropout(x, p=p, training=True),)

    handle = model.classifier.register_forward_pre_hook(_dropout_hook)
    try:
        with torch.no_grad():
            for _ in range(n_passes):
                out = model(tensor[None, ...])
                predictions.append(torch.sigmoid(out).cpu().numpy()[0])
    finally:
        handle.remove()

    predictions = np.stack(predictions)
    return predictions.mean(axis=0), predictions.std(axis=0)


# ------------------------------------------------------------
# HELPERS — LUNG SEGMENTATION
# ------------------------------------------------------------
def compute_lung_mask(pil_image: Image.Image, feather: int = 15) -> np.ndarray:
    img = np.array(pil_image.convert("L"))
    h, w = img.shape

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_eq = clahe.apply(img)

    blur_kernel = max(11, (min(h, w) // 40) | 1)
    img_blur = cv2.GaussianBlur(img_eq, (blur_kernel, blur_kernel), 0)

    _, thresh = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = 255 - thresh

    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close, iterations=2)

    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    if num_labels > 3:
        areas = stats[1:, cv2.CC_STAT_AREA]
        top_two = np.argsort(areas)[-2:] + 1
        new_mask = np.zeros_like(mask)
        for lid in top_two:
            new_mask[labels == lid] = 255
        mask = new_mask

    mask_bool = mask > 0
    mask_bool = binary_fill_holes(mask_bool)

    k_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask_dilated = cv2.dilate(mask_bool.astype(np.uint8) * 255, k_dilate, iterations=1)

    mask_float = mask_dilated.astype(np.float32) / 255.0
    if feather > 0:
        feather = feather if feather % 2 == 1 else feather + 1
        mask_float = cv2.GaussianBlur(mask_float, (feather, feather), 0)

    return np.clip(mask_float, 0.0, 1.0)


def blend_heatmap_with_mask(
    heatmap_rgb: np.ndarray,
    original_rgb: np.ndarray,
    lung_mask: np.ndarray,
) -> np.ndarray:
    """
    Show heatmap inside the lung mask; keep the original radiograph outside.
    Smooth blend at mask edges.
    """
    h, w = heatmap_rgb.shape[:2]

    if lung_mask.shape[:2] != (h, w):
        lung_mask = cv2.resize(lung_mask, (w, h), interpolation=cv2.INTER_LINEAR)

    if original_rgb.shape[:2] != (h, w):
        original_rgb = cv2.resize(original_rgb, (w, h), interpolation=cv2.INTER_LINEAR)

    mask3 = np.stack([lung_mask] * 3, axis=-1).astype(np.float32)

    blended = (
        heatmap_rgb.astype(np.float32) * mask3
        + original_rgb.astype(np.float32) * (1.0 - mask3)
    )
    return np.clip(blended, 0, 255).astype(np.uint8)


# ------------------------------------------------------------
# ROUTES
# ------------------------------------------------------------
@app.get("/")
def root():
    return {
        "service": "navantix-pulmo-api",
        "version": "1.0.0",
        "status": "online",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "model": "densenet121-res224-all",
        "model_loaded": _cxr_model is not None,
        "modality_check": _clip_model is not None,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        return await _predict_impl(file)
    except HTTPException:
        raise
    except Exception as e:
        print("\n========== PREDICT ERROR ==========")
        traceback.print_exc()
        print("===================================\n")
        return {
            "filename": file.filename if file else "unknown",
            "rejected": True,
            "rejection_reason": f"Backend error: {type(e).__name__}: {e}",
            "findings": [],
            "heatmap_base64": None,
            "heatmaps": {},
        }


async def _predict_impl(file: UploadFile):
    if _cxr_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Empty file.")

    try:
        input_image = Image.open(BytesIO(contents)).convert("L")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    # -------- Step 1: Modality check --------
    modality = check_modality(input_image)
    if not modality["is_cxr"]:
        return {
            "filename": file.filename,
            "rejected": True,
            "rejection_reason": "Image does not appear to be a chest X-ray.",
            "modality": modality,
            "findings": [],
            "heatmap_base64": None,
            "heatmaps": {},
        }

    # -------- Step 2: Inference with MC Dropout --------
    t0 = time.time()
    img_array = np.array(input_image)
    tensor = prepare_xray(img_array)

    mean_scores, std_scores = mc_dropout_predict(
        _cxr_model, tensor, n_passes=MC_PASSES, p=MC_DROPOUT_P
    )

    indexed = list(zip(_cxr_model.pathologies, mean_scores, std_scores))
    indexed.sort(key=lambda x: -x[1])
    pairs = [(label, float(mean), float(std)) for label, mean, std in indexed]

    findings = []
    for label, mean, std in pairs:
        findings.append({
            "label": label,
            "score": float(mean),
            "std": float(std),
            "severity": severity_from_score(float(mean)),
            "confidence": confidence_from_std(float(std)),
        })

    top_finding = pairs[0][0]
    top_score = pairs[0][1]
    top_std = pairs[0][2]
    flagged = top_score >= CLINICAL_THRESHOLD

    # -------- Step 3: Lung mask --------
    lung_mask = None
    try:
        lung_mask = compute_lung_mask(input_image)
        mask_coverage = float(lung_mask.mean())
        print(f"[info] Lung mask coverage: {mask_coverage * 100:.1f}%")
        if mask_coverage < 0.05:
            print("[warn] Lung mask too small — skipping masking")
            lung_mask = None
    except Exception as e:
        print(f"[warn] Lung mask skipped: {e}")
        lung_mask = None

    # -------- Step 4: Grad-CAM++ for top N findings --------
    heatmap_b64 = None
    heatmaps_by_label = {}

    try:
        target_layers = [_cxr_model.features.denseblock4]
        cam = GradCAMPlusPlus(model=_cxr_model, target_layers=target_layers)

        original = np.array(input_image)
        h, w = original.shape
        rgb_float = np.stack([original / 255.0] * 3, axis=-1).astype(np.float32)
        rgb_uint8 = (rgb_float * 255.0).astype(np.uint8)

        for label, _mean, _std in pairs[:HEATMAP_TOP_N]:
            try:
                idx = _cxr_model.pathologies.index(label)
                grayscale_cam = cam(
                    input_tensor=tensor[None, ...],
                    targets=[ClassifierOutputTarget(idx)],
                )[0]

                cam_pil = Image.fromarray((grayscale_cam * 255).astype(np.uint8))
                cam_pil = cam_pil.resize((w, h), Image.BILINEAR)
                cam_full = np.array(cam_pil).astype(np.float32) / 255.0

                overlay = show_cam_on_image(rgb_float, cam_full, use_rgb=True)

                if lung_mask is not None:
                    overlay = blend_heatmap_with_mask(
                        overlay, rgb_uint8, lung_mask
                    )

                heatmaps_by_label[label] = tensor_to_base64_png(overlay)

                if label == top_finding:
                    heatmap_b64 = heatmaps_by_label[label]
            except Exception as e:
                print(f"[warn] Grad-CAM failed for {label}: {e}")
    except Exception as e:
        print(f"[warn] Grad-CAM setup skipped: {e}")

    elapsed_ms = (time.time() - t0) * 1000.0

    return {
        "filename": file.filename,
        "rejected": False,
        "elapsed_ms": round(elapsed_ms, 1),
        "threshold": CLINICAL_THRESHOLD,
        "top_finding": top_finding,
        "top_score": top_score,
        "top_std": top_std,
        "flagged": flagged,
        "findings": findings,
        "heatmap_base64": heatmap_b64,
        "heatmaps": heatmaps_by_label,
        "modality": modality,
        "mc_passes": MC_PASSES,
        "lung_mask_applied": lung_mask is not None,
    }