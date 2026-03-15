import gradio as gr
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
from pathlib import Path
import torchvision.transforms as transforms
from huggingface_hub import hf_hub_download

# Make sure the 'models' directory is in the python path
import sys
sys.path.append(str(Path(__file__).parent))

from models.model_tracking import TrackingTransformer

# --- Configuration ---
# Use a global variable to load the model only once.
MODEL_CACHE = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Helper Functions ---
def load_model_and_config():
    """Loads the model and configuration."""
    global MODEL_CACHE
    if MODEL_CACHE is not None:
        return MODEL_CACHE

    # --- Paths ---
    # Assumes the script is run from the root of the repository.
    config_path = 'configs/Tracking.yaml'
    # The checkpoint should be downloaded or placed in the repo.
    # The README for the original repo points to a Google Drive link.
    # For a HF Space, it's best to upload it to the repo.
    CHECKPOINT_PATH = "checkpoint_18.pth" # Assumes it's in the root.

    # --- Load Config ---
    try:
        config = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Config file not found at '{config_path}'. "
            "Please make sure 'configs/Tracking.yaml' is in your repository."
        )

    # --- Load Model ---
    model = TrackingTransformer(config=config, init_deit=True)

    # --- Load Checkpoint ---
    if not os.path.exists(CHECKPOINT_PATH):
        # As an example, you could host your model on the hub and download it.
        # HF_REPO_ID = "your-hf-username/your-eyeformer-repo" # CHANGE THIS
        # print(f"'{CHECKPOINT_PATH}' not found locally. Trying to download from Hugging Face Hub...")
        # try:
        #     CHECKPOINT_PATH = hf_hub_download(repo_id=HF_REPO_ID, filename=CHECKPOINT_PATH)
        #     print(f"Downloaded checkpoint to {CHECKPOINT_PATH}")
        # except Exception as e:
        raise FileNotFoundError(
            f"Checkpoint file '{CHECKPOINT_PATH}' not found in the repository root. "
            f"Please upload your 'checkpoint_18.pth' to the repository."
        )
    
    print(f"Loading checkpoint from: {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')
    state_dict = checkpoint['model']
    
    msg = model.load_state_dict(state_dict, strict=False)
    print("Model state loaded. Missing keys:", msg.missing_keys)
    
    model.to(DEVICE)
    model.eval()
    print("Model loaded successfully.")
    
    MODEL_CACHE = (model, config)
    return MODEL_CACHE

def get_transform(image_size):
    """Returns the image transformation pipeline."""
    # These normalization stats are from CLIP, used in similar ViT-based models.
    # It's a good starting point if the original stats are not available.
    return transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])

def draw_scanpath(image, scanpath_points):
    """Draws the scanpath on the image."""
    draw = ImageDraw.Draw(image)
    
    # Use a font for numbering fixations.
    try:
        font = ImageFont.truetype("arial.ttf", size=max(15, int(image.width / 50)))
    except IOError:
        font = ImageFont.load_default()

    # Draw lines connecting fixations
    if len(scanpath_points) > 1:
        draw.line(scanpath_points, fill="red", width=3)
        
    # Draw fixation points
    radius = max(5, int(image.width / 100))
    for i, (x, y) in enumerate(scanpath_points):
        # Circle for fixation
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill="yellow", outline="red", width=2)
        # Number for fixation order
        text_bbox = draw.textbbox((0, 0), str(i + 1), font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        draw.text((x - text_w / 2, y - text_h / 2), str(i + 1), fill="black", font=font)
        
    return image

# --- Main Prediction Function for Gradio ---
def predict(input_image_pil):
    """
    Takes a PIL image, predicts a scanpath, and returns the image with the scanpath drawn on it.
    """
    if input_image_pil is None:
        return None, "Please upload an image."

    model, config = load_model_and_config()
    
    original_width, original_height = input_image_pil.size
    
    # Preprocess
    transform = get_transform(config['image_res'])
    image_tensor = transform(input_image_pil).unsqueeze(0).to(DEVICE)
    
    # Predict
    with torch.no_grad():
        # The model predicts a sequence of normalized (x, y) coordinates
        pred_seq = model(image_tensor, greedy=True)
        
    pred_seq = pred_seq.squeeze(0).cpu().numpy() # Shape: (seq_len, 2)
    
    # Post-process: denormalize coordinates
    scanpath_points = []
    for (x, y) in pred_seq:
        # The model output is normalized to [0, 1]. Scale to original image dimensions.
        abs_x = np.clip(x, 0, 1) * original_width
        abs_y = np.clip(y, 0, 1) * original_height
        scanpath_points.append((abs_x, abs_y))
        
    # Visualize
    output_image = draw_scanpath(input_image_pil.copy(), scanpath_points)
        
    return output_image

examples = [['example.jpg']]

iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload Image"),
    outputs=gr.Image(type="pil", label="Predicted Scanpath"),
    title="EyeFormer: Scanpath Prediction",
    description="Upload an image to predict a human-like scanpath. The model will generate a sequence of fixations (where a person might look). The predicted scanpath is drawn over the image, with numbered circles indicating the order of fixations.",
    article="<p style='text-align: center'><a href='https://github.com/salesforce/EyeFormer' target='_blank'>GitHub Repository</a></p>",
    examples=examples,
    allow_flagging='never'
)

if __name__ == "__main__":
    iface.launch()
