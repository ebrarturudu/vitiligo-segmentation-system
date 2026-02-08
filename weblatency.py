# app.py (Flask) - düzeltilmiş çekirdek
import io, time, csv, os, torch
from datetime import datetime
from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
from torchvision import transforms

app = Flask(__name__)

# --- MODELLERİ BİR KEZ YÜKLE ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Sınıflandırma: EfficientNet-B0 + tek nöronlu head (sigmoid)
from efficientnet_pytorch import EfficientNet
import torch.nn as nn
clf_model = EfficientNet.from_name('efficientnet-b0')
clf_model._fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(clf_model._fc.in_features, 1))
clf_model.load_state_dict(torch.load(r"C:\Users\ebrar\OneDrive\Desktop\VSCODE PYTHON\vitiligo_app\model\models\siniflandirma_model.pth", map_location=device))
clf_model.eval().to(device)  # :contentReference[oaicite:4]{index=4}

# Segmentasyon: SAM_UNet + kendi ağırlıkların
# (SAM backbone için sam_vit_b.pth bekleniyor; SAM_UNet içinde kullanılıyor)  :contentReference[oaicite:5]{index=5}
from model.sam_unet import SAM_UNet  # senin dosyanın içindeki sınıf  :contentReference[oaicite:6]{index=6}
seg_model = SAM_UNet().to(device)
seg_model.load_state_dict(torch.load(r"C:\Users\ebrar\OneDrive\Desktop\VSCODE PYTHON\vitiligo_app\model\models\best_model1.pth", map_location=device))
seg_model.eval()  # :contentReference[oaicite:7]{index=7}

# Transformlar (segment.py ile uyumlu)  :contentReference[oaicite:8]{index=8}
cls_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])
seg_tf = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor()
])

def run_classification(img_pil):
    x = cls_tf(img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        prob = torch.sigmoid(clf_model(x)).item()  # 0..1
    return {"prob_vitiligo": prob, "is_vitiligo": prob >= 0.5}

def run_segmentation(img_pil):
    x = seg_tf(img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = seg_model(x)             # [1,1,H,W]
        mask = torch.sigmoid(logits)[0,0].cpu().numpy()  # 0..1
    # (Opsiyonel) Dice hesaplaması için binarize:
    mask_bin = (mask > 0.5).astype(np.uint8)
    return {"mask": mask_bin, "dice": None}  # Dice yoksa None döndürüyoruz

def compute_severity(mask_bin):
    # lesion/total ratio
    ratio = float(mask_bin.sum()) / float(mask_bin.size)
    return {"lesion_ratio": round(ratio, 4)}

# --- LOG ---
LOG_PATH = "latency_log.csv"
if not os.path.exists(LOG_PATH):
    with open(LOG_PATH, "w", newline="") as f:
        csv.writer(f).writerow(["ts","img_id","t_total_ms","t_clf_ms","t_seg_ms","t_sev_ms","clf_positive"])

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("image")
    if file is None:
        return jsonify({"error": "no image"}), 400

    img_id = request.form.get("img_id", file.filename or "image")
    img = Image.open(io.BytesIO(file.read())).convert("RGB")

    t0 = time.perf_counter()

    # 1) classification
    tc0 = time.perf_counter()
    clf_out = run_classification(img)
    t_clf = (time.perf_counter() - tc0) * 1000.0

    # 2) segmentation (yalnızca pozitifse)
    t_seg = 0.0
    seg_out = {}
    if clf_out["is_vitiligo"]:
        ts0 = time.perf_counter()
        seg_out = run_segmentation(img)
        t_seg = (time.perf_counter() - ts0) * 1000.0

        # 3) severity
        tv0 = time.perf_counter()
        sev_out = compute_severity(seg_out["mask"])
        t_sev = (time.perf_counter() - tv0) * 1000.0
    else:
        t_sev = 0.0
        sev_out = {"lesion_ratio": 0.0}

    t_total = (time.perf_counter() - t0) * 1000.0

    # CSV log
    with open(LOG_PATH, "a", newline="") as f:
        csv.writer(f).writerow([
            datetime.utcnow().isoformat(),
            img_id, f"{t_total:.1f}", f"{t_clf:.1f}", f"{t_seg:.1f}", f"{t_sev:.1f}",
            int(clf_out["is_vitiligo"])
        ])

    return jsonify({
        "img_id": img_id,
        "classification": clf_out,  # {"prob_vitiligo":..., "is_vitiligo":...}
        "segmentation": {"dice": seg_out.get("dice", None)},
        "severity": sev_out,
        "timings_ms": {
            "total": round(t_total,1),
            "classification": round(t_clf,1),
            "segmentation": round(t_seg,1),
            "severity": round(t_sev,1)
        }
    })