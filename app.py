import os
import random
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import gradio as gr

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
#  MODEL DEFINITIONS (must match training)
# ─────────────────────────────────────────────
class ScanpathPredictor(nn.Module):
    def __init__(self, max_fixations=30, hidden_dim=512, num_layers=2):
        super().__init__()
        self.max_fixations = max_fixations
        self.hidden_dim = hidden_dim
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
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
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
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
#  LOAD MODELS
# ─────────────────────────────────────────────
def load_models():
    sp = ScanpathPredictor(max_fixations=30, hidden_dim=512, num_layers=2).to(device)
    fp = FatiguePredictor(hidden_dim=512, num_layers=2).to(device)

    sp_path = "best_scanpath_model_enhanced.pth"
    fp_path = "best_fatigue_model.pth"

    if os.path.exists(sp_path):
        sp.load_state_dict(torch.load(sp_path, map_location=device))
        sp.eval()
        print(f"✅ Scanpath model loaded")
    else:
        raise FileNotFoundError(f"Scanpath model not found: {sp_path}")

    if os.path.exists(fp_path):
        fp.load_state_dict(torch.load(fp_path, map_location=device))
        fp.eval()
        print(f"✅ Fatigue model loaded")
    else:
        raise FileNotFoundError(f"Fatigue model not found: {fp_path}")

    return sp, fp

scanpath_model, fatigue_model = load_models()

# ─────────────────────────────────────────────
#  INFERENCE
# ─────────────────────────────────────────────
def predict(image: Image.Image, min_fix: int, max_fix: int, mode: str):
    orig_w, orig_h = image.size
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        if mode == "Variable length (EOS-based)":
            pred_sp, eos_probs = scanpath_model(
                img_tensor,
                teacher_forcing_ratio=0.0,
                dynamic_length=True,
                min_fixations=min_fix,
                max_generation_length=max_fix
            )
            pred_sp   = pred_sp.squeeze(0).cpu().numpy()
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

        # Denormalise & filter
        denorm, norm_for_fatigue = [], []
        for coords in pred_sp:
            x, y = coords[0] * orig_w, coords[1] * orig_h
            if mode == "Variable length (EOS-based)" and x < 10 and y < 10:
                continue
            denorm.append([x, y])
            norm_for_fatigue.append([coords[0], coords[1]])

        fatigue_labels = []
        if norm_for_fatigue:
            sp_tensor = torch.tensor([norm_for_fatigue], dtype=torch.float32).to(device)
            fat_preds = fatigue_model(img_tensor, sp_tensor)
            fat_preds = fat_preds.squeeze(0).squeeze(-1).cpu().numpy()
            fatigue_labels = (fat_preds > 0.5).astype(int).tolist()

    return denorm, fatigue_labels


def build_figure(image: Image.Image, scanpath, fatigue_labels):
    img_np = np.array(image)
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#0f172a")
    ax.imshow(img_np)

    if scanpath:
        sp = np.array(scanpath)

        # Connecting lines
        if len(sp) > 1:
            for i in range(len(sp) - 1):
                cf = fatigue_labels[i]   if i   < len(fatigue_labels) else 0
                nf = fatigue_labels[i+1] if i+1 < len(fatigue_labels) else 0
                color = "#ef4444" if (cf or nf) else "#3b82f6"
                lw    = 3         if (cf or nf) else 2
                ax.plot([sp[i][0], sp[i+1][0]], [sp[i][1], sp[i+1][1]],
                        color=color, linewidth=lw, alpha=0.85, zorder=2)

        # Fixation dots
        for i, (x, y) in enumerate(sp):
            fat = fatigue_labels[i] if i < len(fatigue_labels) else 0
            if fat:
                ax.plot(x, y, "o", markersize=13, markeredgewidth=2.5,
                        markeredgecolor="#7f1d1d", markerfacecolor="#ef4444",
                        alpha=0.92, zorder=3)
                ax.text(x, y, str(i+1), color="white", fontsize=9,
                        fontweight="bold", ha="center", va="center", zorder=4)
            else:
                ax.plot(x, y, "o", markersize=11, markeredgewidth=2,
                        markeredgecolor="white", markerfacecolor="#22c55e",
                        alpha=0.85, zorder=3)
                ax.text(x, y, str(i+1), color="#0f172a", fontsize=8,
                        fontweight="bold", ha="center", va="center", zorder=4)

        n_fat = sum(fatigue_labels) if fatigue_labels else 0
        legend_elements = [
            Patch(facecolor="#22c55e", edgecolor="white",  label="Normal fixation"),
            Patch(facecolor="#ef4444", edgecolor="#7f1d1d", label="Fatigue fixation"),
            plt.Line2D([0],[0], color="#3b82f6", lw=2, label="Normal path"),
            plt.Line2D([0],[0], color="#ef4444", lw=3, label="Fatigue path"),
        ]
        ax.legend(handles=legend_elements, loc="upper right",
                  facecolor="#1e293b", labelcolor="#e2e8f0",
                  framealpha=0.9, fontsize=10)

        title = (f"Scanpath — {len(scanpath)} fixations  |  "
                 f"{n_fat} fatigue points  ({n_fat/len(scanpath)*100:.0f}%)")
    else:
        title = "No fixations generated"

    ax.set_title(title, color="#f1f5f9", fontsize=13, fontweight="bold", pad=12)
    ax.axis("off")
    plt.tight_layout()
    return fig


def build_stats_md(scanpath, fatigue_labels, image: Image.Image):
    if not scanpath:
        return "No fixations generated."
    n = len(scanpath)
    n_fat = sum(fatigue_labels) if fatigue_labels else 0
    w, h = image.size
    sp = np.array(scanpath)

    # Path length
    dists = [np.linalg.norm(sp[i+1] - sp[i]) for i in range(len(sp)-1)]
    path_len = sum(dists)

    # Coverage: bounding box area as % of image
    x_range = sp[:,0].max() - sp[:,0].min()
    y_range = sp[:,1].max() - sp[:,1].min()
    coverage = (x_range * y_range) / (w * h) * 100

    # First fatigue index
    first_fat = next((i for i, f in enumerate(fatigue_labels) if f), None)
    first_fat_str = f"Fixation #{first_fat+1}" if first_fat is not None else "None"

    return f"""
### 📊 Prediction Summary

| Metric | Value |
|--------|-------|
| Total fixations | **{n}** |
| Fatigue fixations | **{n_fat}** ({n_fat/n*100:.0f}%) |
| Normal fixations | **{n - n_fat}** |
| First fatigue at | **{first_fat_str}** |
| Total path length | **{path_len:.0f} px** |
| Avg saccade length | **{np.mean(dists):.0f} px** |
| Gaze coverage area | **{coverage:.1f}%** of image |
| Image size | {w} × {h} px |
"""


def run(image, min_fix, max_fix, mode):
    if image is None:
        return None, "⚠️ Please upload an image."

    img = Image.fromarray(image).convert("RGB") if not isinstance(image, Image.Image) else image

    try:
        scanpath, fatigue_labels = predict(img, int(min_fix), int(max_fix), mode)
        fig   = build_figure(img, scanpath, fatigue_labels)
        stats = build_stats_md(scanpath, fatigue_labels, img)
        return fig, stats
    except Exception as e:
        return None, f"❌ Error: {e}"


# ─────────────────────────────────────────────
#  GRADIO UI
# ─────────────────────────────────────────────
CSS = """
body, .gradio-container { background: #0f172a !important; color: #e2e8f0 !important; }
.gr-panel, .gr-box, .gr-form { background: #1e293b !important; border: 1px solid #334155 !important; border-radius: 10px !important; }
label { color: #94a3b8 !important; font-size: 13px !important; }
.gr-button-primary { background: linear-gradient(135deg,#6366f1,#8b5cf6) !important; border: none !important; color: white !important; font-weight: 600 !important; }
.gr-button { background: #334155 !important; color: #e2e8f0 !important; border: 1px solid #475569 !important; }
h1, h2, h3 { color: #f1f5f9 !important; }
"""

HEADER = """
<div style="text-align:center;padding:20px 0 10px">
  <h1 style="font-size:2.2rem;font-weight:800;
             background:linear-gradient(135deg,#6366f1,#22c55e);
             -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin:0">
    ScanpathAI
  </h1>
  <p style="color:#94a3b8;margin-top:8px;font-size:1rem">
    Upload an image — the model predicts where a human eye would look,<br>
    then flags which fixations show signs of <span style="color:#ef4444">visual fatigue</span>.
    <span>DISCLAIMER: This model is trained on a very small dataset and created as proof of concept, therefore it might not work for all designs</span>
  </p>
</div>
"""

with gr.Blocks(css=CSS, title="ScanpathAI") as demo:
    gr.HTML(HEADER)

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(label="Upload Image", type="numpy", height=300)

            gr.Markdown("### ⚙️ Settings")
            mode_input = gr.Radio(
                choices=["Variable length (EOS-based)", "Fixed length"],
                value="Variable length (EOS-based)",
                label="Prediction mode"
            )
            min_fix_input = gr.Slider(5, 20, value=10, step=1,
                                      label="Min fixations")
            max_fix_input = gr.Slider(10, 30, value=25, step=1,
                                      label="Max fixations")

            run_btn = gr.Button("🔍 Generate Scanpath", variant="primary", size="lg")

        with gr.Column(scale=2):
            plot_output  = gr.Plot(label="Predicted Scanpath")
            stats_output = gr.Markdown()

    gr.Markdown("""
---
**Legend** &nbsp;🟢 Green = normal fixation &nbsp;|&nbsp; 🔴 Red = fatigue fixation &nbsp;|&nbsp;
🔵 Blue line = normal saccade &nbsp;|&nbsp; 🔴 Red line = fatigue saccade
&nbsp;&nbsp; Numbers indicate fixation order.
""")

    run_btn.click(
        fn=run,
        inputs=[image_input, min_fix_input, max_fix_input, mode_input],
        outputs=[plot_output, stats_output]
    )

if __name__ == "__main__":
    demo.launch(share=True)
