import os
import glob
import random
import numpy as np
import cv2
import torch
import gradio as gr

# Import everything needed from your final solution script
# (Ensure your previous script is saved as 'leaf_counting_final.py' in the same folder)
from leaf_counting_final import (
    LeafCountNet, 
    generate_mask_union, 
    normalise_and_to_tensor, 
    build_tta_transforms, 
    CFG
)

# =============================================================================
# INITIALISATION
# =============================================================================

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Model
model = LeafCountNet(CFG).to(device)
checkpoint_path = CFG["checkpoint"]

if os.path.exists(checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f"Loaded weights from {checkpoint_path}")
else:
    print(f"WARNING: No checkpoint found at {checkpoint_path}. Using random weights.")
model.eval()

# Prepare TTA transforms (reused from your test inference logic)
tta_transforms = build_tta_transforms(CFG["img_size"])


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def predict_leaves(image_rgb: np.ndarray):
    """Handles inference for the uploaded image using Union Mask and TTA."""
    if image_rgb is None:
        return None, "Please upload an image."

    # 1. Generate Union Mask
    mask = generate_mask_union(image_rgb, CFG)

    # 2. Create the visual masked image for the UI
    mask_3ch = np.stack([mask] * 3, axis=-1)
    masked_vis = cv2.bitwise_and(image_rgb, mask_3ch)

    # 3. Stack into 4 channels (RGB + Mask)
    img_4ch = np.concatenate([image_rgb, mask[:, :, np.newaxis]], axis=-1)

    # 4. TTA Inference
    preds = []
    with torch.no_grad():
        for aug in tta_transforms:
            aug_img = aug(image=img_4ch)["image"]
            tensor = normalise_and_to_tensor(aug_img).unsqueeze(0).to(device)
            pred, _ = model(tensor)
            preds.append(pred.item())

    # Apply your standard deviation threshold logic
    std = np.std(preds)
    if std >= CFG["tta_std_threshold"]:
        raw = np.median(preds)
    else:
        raw = np.mean(preds)

    final_count = max(1, round(raw))

    return masked_vis, f"{final_count}"


def get_gallery_images(dataset_root="./leaf_estimation_dataset"):
    """Fetches 10 random images for each crop and shuffles them."""
    crops = ["mustard", "okra", "radish", "wheat"]
    gallery_items = []
    
    for crop in crops:
        search_path = os.path.join(dataset_root, crop, "**", "*.png")
        images = glob.glob(search_path, recursive=True)
        print(search_path)
        if images:
            sampled = random.sample(images, min(10, len(images)))
            # Convert to absolute paths so Gradio doesn't get confused by relative "../" paths
            gallery_items.extend([(os.path.abspath(img), crop.capitalize()) for img in sampled])
            
    print(f"[Gallery] Found {len(gallery_items)} images to display.")
    random.shuffle(gallery_items)
    return gallery_items
# =============================================================================
# GRADIO UI LAYOUT
# =============================================================================

# Fetch images once on startup
initial_gallery = get_gallery_images(os.path.join(CFG["root"], "train"))

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🌿 Leaf Counting Dashboard")
    gr.Markdown("Upload an image of a plant to generate a unified segmentation mask and predict the total number of leaves using the EfficientNetV2-S 4-Channel model.")

    with gr.Row():
        # Left Column: Input
        with gr.Column(scale=1):
            input_image = gr.Image(label="Drop or Click to Upload Image", type="numpy")
            analyze_btn = gr.Button("Count Leaves", variant="primary")
            
        # Right Column: Outputs
        with gr.Column(scale=1):
            output_mask = gr.Image(label="Masked Image (Union Mask)", interactive=False)
            output_count = gr.Textbox(label="Predicted Leaf Count", text_align="center", scale=0)

    gr.Markdown("---")
    gr.Markdown("### 📸 Training Data Gallery")
    gr.Markdown("A shuffled selection of 10 samples from each crop type (Mustard, Okra, Radish, Wheat).")
    
    # Gallery Section
    
    gallery = gr.Gallery(
        value=initial_gallery,
        label="Dataset Samples",
        show_label=False,
        interactive=False,  
        elem_id="gallery",
        columns=8,           # <--- Forces 8 images per row
        rows=5,              # <--- 40 total images / 8 columns = 5 rows
        object_fit="cover",  # <--- 'cover' makes small thumbnails look uniform and neat
        height="auto"        
    )

    # Button click triggers the prediction
    analyze_btn.click(
        fn=predict_leaves,
        inputs=[input_image],
        outputs=[output_mask, output_count]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=True)