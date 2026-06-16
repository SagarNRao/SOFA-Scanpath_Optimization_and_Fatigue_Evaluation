import os
import io
import base64
import random
import urllib.request

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ─────────────────────────────────────────────
#  REPRODUCIBILITY
# ─────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
])

# ─────────────────────────────────────────────
#  MODEL DEFINITIONS (must match training — unchanged from your app.py,
#  except weights=None below: your checkpoint overwrites every weight
#  anyway, so there's no reason to pull the ImageNet weights from the
#  internet on every cold start)
# ─────────────────────────────────────────────
class ScanpathPredictor(nn.Module):
    def __init__(self, max_fixations=30, hidden_dim=512, num_layers=2):
        super().__init__()
        self.max_fixations = max_fixations
        self.hidden_dim = hidden_dim
        resnet = resnet50(weights=None)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        for param in list(self.feature_extractor.parameters())[:-20]:
            param.requires_grad = False
        self.feature_dim = 2048
        self.lstm = nn.LSTM(
            input_size=self.feature_dim + 2,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_dim, 256), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(256, 2), nn.Sigmoid()
        )
        self.eos_predictor = nn.Sequential(
            nn.Linear(hidden_dim, 128), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(128, 1), nn.Sigmoid()
        )

    def forward(self, images, teacher_forcing_ratio=0.5, target_scanpath=None,
                dynamic_length=False, min_fixations=10, max_generation_length=None):
        batch_size = images.size(0)
        generation_length = max_generation_length or self.max_fixations
        img_features = self.feature_extractor(images).view(batch_size, -1)
        h0 = torch.zeros(2, batch_size, self.hidden_dim).to(images.device)
        c0 = torch.zeros(2, batch_size, self.hidden_dim).to(images.device)
        outputs, eos_outputs = [], []
        prev_coords = torch.zeros(batch_size, 2).to(images.device)
        for t in range(generation_length):
            lstm_input = torch.cat([img_features, prev_coords], dim=1).unsqueeze(1)
            lstm_out, (h0, c0) = self.lstm(lstm_input, (h0, c0))
            lstm_out = lstm_out.squeeze(1)
            coords = self.fc_out(lstm_out)
            eos_prob = self.eos_predictor(lstm_out)
            outputs.append(coords)
            eos_outputs.append(eos_prob)
            if dynamic_length and not self.training and t >= min_fixations:
                if torch.all(eos_prob > 0.5):
                    break
            if (self.training and target_scanpath is not None
                    and t < target_scanpath.size(1)
                    and random.random() < teacher_forcing_ratio):
                prev_coords = target_scanpath[:, t, :]
            else:
                prev_coords = coords
        return torch.stack(outputs, dim=1), torch.stack(eos_outputs, dim=1)


class FatiguePredictor(nn.Module):
    def __init__(self, hidden_dim=512, num_layers=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        resnet = resnet50(weights=None)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.feature_dim = 2048
        for param in list(self.feature_extractor.parameters())[:-20]:
            param.requires_grad = False
        self.lstm = nn.LSTM(
            input_size=self.feature_dim + 2,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_dim, 256), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(256, 1), nn.Sigmoid()
        )

    def forward(self, images, scanpaths):
        batch_size = images.size(0)
        img_features = self.feature_extractor(images).view(batch_size, -1)
        seq_len = scanpaths.size(1)
        img_exp = img_features.unsqueeze(1).repeat(1, seq_len, 1)
        lstm_input = torch.cat([img_exp, scanpaths], dim=2)
        lstm_out, _ = self.lstm(lstm_input)
        return self.fc_out(lstm_out)


# ─────────────────────────────────────────────
#  WEIGHT LOADING — download from a URL if not present locally.
#  Commit the .pth files directly only if you're using Git LFS;
#  otherwise host them somewhere (e.g. a Hugging Face model repo)
#  and set SCANPATH_MODEL_URL / FATIGUE_MODEL_URL as env vars on Render.
# ─────────────────────────────────────────────
MODEL_DIR = os.environ.get("MODEL_DIR", os.path.dirname(os.path.abspath(__file__)))
SCANPATH_PATH = os.path.join(MODEL_DIR, "best_scanpath_model_enhanced.pth")
FATIGUE_PATH = os.path.join(MODEL_DIR, "best_fatigue_model.pth")
SCANPATH_URL = os.environ.get("SCANPATH_MODEL_URL")
FATIGUE_URL = os.environ.get("FATIGUE_MODEL_URL")


def _download(url: str, dest: str):
    print(f"Downloading {os.path.basename(dest)} from {url} ...")
    urllib.request.urlretrieve(url, dest)
    print(f"Saved to {dest} ({os.path.getsize(dest) / 1e6:.1f} MB)")


def ensure_weights():
    if not os.path.exists(SCANPATH_PATH):
        if not SCANPATH_URL:
            raise FileNotFoundError(
                "best_scanpath_model_enhanced.pth not found and SCANPATH_MODEL_URL is not set."
            )
        _download(SCANPATH_URL, SCANPATH_PATH)
    if not os.path.exists(FATIGUE_PATH):
        if not FATIGUE_URL:
            raise FileNotFoundError(
                "best_fatigue_model.pth not found and FATIGUE_MODEL_URL is not set."
            )
        _download(FATIGUE_URL, FATIGUE_PATH)


def load_models():
    ensure_weights()
    sp = ScanpathPredictor(max_fixations=30, hidden_dim=512, num_layers=2).to(device)
    fp = FatiguePredictor(hidden_dim=512, num_layers=2).to(device)

    sp.load_state_dict(torch.load(SCANPATH_PATH, map_location=device))
    sp.eval()
    print("Scanpath model loaded")

    fp.load_state_dict(torch.load(FATIGUE_PATH, map_location=device))
    fp.eval()
    print("Fatigue model loaded")

    return sp, fp


scanpath_model, fatigue_model = load_models()

# ─────────────────────────────────────────────
#  INFERENCE  (same logic as your Gradio app)
# ─────────────────────────────────────────────
def predict(image: Image.Image, min_fix: int, max_fix: int, mode: str):
    orig_w, orig_h = image.size
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        if mode == "variable":
            pred_sp, eos_probs = scanpath_model(
                img_tensor,
                teacher_forcing_ratio=0.0,
                dynamic_length=True,
                min_fixations=min_fix,
                max_generation_length=max_fix
            )
            pred_sp = pred_sp.squeeze(0).cpu().numpy()
            eos_probs = eos_probs.squeeze(0).cpu().numpy()
            actual_len = max_fix
            for i in range(min_fix, len(eos_probs)):
                if eos_probs[i] > 0.5:
                    actual_len = i + 1
                    break
            pred_sp = pred_sp[:actual_len]
        else:
            pred_sp, _ = scanpath_model(
                img_tensor,
                teacher_forcing_ratio=0.0,
                max_generation_length=max_fix
            )
            pred_sp = pred_sp.squeeze(0).cpu().numpy()

        denorm, norm_for_fatigue = [], []
        for coords in pred_sp:
            x, y = float(coords[0] * orig_w), float(coords[1] * orig_h)
            if mode == "variable" and x < 10 and y < 10:
                continue
            denorm.append([x, y])
            norm_for_fatigue.append([float(coords[0]), float(coords[1])])

        fatigue_labels = []
        if norm_for_fatigue:
            sp_tensor = torch.tensor([norm_for_fatigue], dtype=torch.float32).to(device)
            fat_preds = fatigue_model(img_tensor, sp_tensor)
            fat_preds = fat_preds.squeeze(0).squeeze(-1).cpu().numpy()
            fatigue_labels = (fat_preds > 0.5).astype(int).tolist()

    return denorm, fatigue_labels


def build_stats(scanpath, fatigue_labels, image: Image.Image):
    if not scanpath:
        return {"message": "No fixations generated."}

    n = len(scanpath)
    n_fat = sum(fatigue_labels) if fatigue_labels else 0
    w, h = image.size
    sp = np.array(scanpath)

    dists = [float(np.linalg.norm(sp[i + 1] - sp[i])) for i in range(len(sp) - 1)]
    path_len = sum(dists)

    x_range = sp[:, 0].max() - sp[:, 0].min()
    y_range = sp[:, 1].max() - sp[:, 1].min()
    coverage = float((x_range * y_range) / (w * h) * 100)

    first_fat = next((i for i, f in enumerate(fatigue_labels) if f), None)

    return {
        "total_fixations": n,
        "fatigue_fixations": n_fat,
        "fatigue_pct": round(n_fat / n * 100, 1) if n else 0,
        "normal_fixations": n - n_fat,
        "first_fatigue_index": first_fat,
        "total_path_length_px": round(path_len, 1),
        "avg_saccade_length_px": round(float(np.mean(dists)), 1) if dists else 0,
        "coverage_pct": round(coverage, 1),
        "image_size": [w, h],
    }


def build_figure(image: Image.Image, scanpath, fatigue_labels):
    img_np = np.array(image)
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#0f172a")
    ax.imshow(img_np)

    if scanpath:
        sp = np.array(scanpath)

        if len(sp) > 1:
            for i in range(len(sp) - 1):
                cf = fatigue_labels[i] if i < len(fatigue_labels) else 0
                nf = fatigue_labels[i + 1] if i + 1 < len(fatigue_labels) else 0
                color = "#ef4444" if (cf or nf) else "#3b82f6"
                lw = 3 if (cf or nf) else 2
                ax.plot([sp[i][0], sp[i + 1][0]], [sp[i][1], sp[i + 1][1]],
                        color=color, linewidth=lw, alpha=0.85, zorder=2)

        for i, (x, y) in enumerate(sp):
            fat = fatigue_labels[i] if i < len(fatigue_labels) else 0
            if fat:
                ax.plot(x, y, "o", markersize=13, markeredgewidth=2.5,
                        markeredgecolor="#7f1d1d", markerfacecolor="#ef4444",
                        alpha=0.92, zorder=3)
                ax.text(x, y, str(i + 1), color="white", fontsize=9,
                        fontweight="bold", ha="center", va="center", zorder=4)
            else:
                ax.plot(x, y, "o", markersize=11, markeredgewidth=2,
                        markeredgecolor="white", markerfacecolor="#22c55e",
                        alpha=0.85, zorder=3)
                ax.text(x, y, str(i + 1), color="#0f172a", fontsize=8,
                        fontweight="bold", ha="center", va="center", zorder=4)

        n_fat = sum(fatigue_labels) if fatigue_labels else 0
        legend_elements = [
            Patch(facecolor="#22c55e", edgecolor="white", label="Normal fixation"),
            Patch(facecolor="#ef4444", edgecolor="#7f1d1d", label="Fatigue fixation"),
            plt.Line2D([0], [0], color="#3b82f6", lw=2, label="Normal path"),
            plt.Line2D([0], [0], color="#ef4444", lw=3, label="Fatigue path"),
        ]
        ax.legend(handles=legend_elements, loc="upper right",
                  facecolor="#1e293b", labelcolor="#e2e8f0",
                  framealpha=0.9, fontsize=10)

        title = (f"Scanpath — {len(scanpath)} fixations  |  "
                 f"{n_fat} fatigue points  ({n_fat / len(scanpath) * 100:.0f}%)")
    else:
        title = "No fixations generated"

    ax.set_title(title, color="#f1f5f9", fontsize=13, fontweight="bold", pad=12)
    ax.axis("off")
    plt.tight_layout()
    return fig


def fig_to_base64_png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# ─────────────────────────────────────────────
#  FASTAPI APP
# ─────────────────────────────────────────────
app = FastAPI(title="ScanpathAI API", version="1.0.0")

allowed_origins = os.environ.get("ALLOWED_ORIGINS", "*")
origins = [o.strip() for o in allowed_origins.split(",")] if allowed_origins != "*" else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "device": str(device)}


@app.post("/predict")
async def predict_endpoint(
    file: UploadFile = File(...),
    min_fix: int = Form(10),
    max_fix: int = Form(25),
    mode: str = Form("variable"),          # "variable" | "fixed"
    return_image: bool = Form(False),
):
    if mode not in ("variable", "fixed"):
        raise HTTPException(status_code=400, detail="mode must be 'variable' or 'fixed'")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read uploaded file as an image")

    try:
        scanpath, fatigue_labels = predict(image, min_fix, max_fix, mode)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    response = {
        "scanpath": scanpath,                 # [[x, y], ...] in original image pixel coords
        "fatigue_labels": fatigue_labels,      # [0/1, ...] aligned with scanpath
        "stats": build_stats(scanpath, fatigue_labels, image),
    }

    if return_image:
        fig = build_figure(image, scanpath, fatigue_labels)
        response["visualization_png_base64"] = fig_to_base64_png(fig)

    return JSONResponse(response)


@app.get("/")
def root():
    return {
        "message": "ScanpathAI API is running.",
        "endpoints": {"health": "/health", "predict": "POST /predict", "docs": "/docs"},
    }