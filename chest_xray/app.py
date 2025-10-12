import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import torch.nn.functional as F
import json
from pathlib import Path

# Load model
@st.cache_resource
def load_model():
    # CHANGE 1: point to your VGG16 weights-only file + labels
    weights_path = Path(r"C:\Users\rishab\chest_xray_classification\weights\vgg16_weights.pt")
    labels_path  = Path(r"C:\Users\rishab\chest_xray_classification\weights\idx_to_class.json")

    # Build VGG16 head (CHANGE 2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(labels_path, "r") as f:
        idx_to_class = json.load(f)
    num_classes = len(idx_to_class)

    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    for p in model.features.parameters():
        p.requires_grad = False
    in_feats = model.classifier[6].in_features
    model.classifier[6] = nn.Sequential(
        nn.Linear(in_feats, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, num_classes),
        nn.LogSoftmax(dim=1),
    )

    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    model.to(device)

    return model, idx_to_class

# Preprocessing function (CHANGE 3: RGB + ImageNet normalization)
def preprocess_image(image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)  # Shape: (1, 3, 224, 224)

# Generate Grad-CAM for VGG16 (CHANGE 4: hook last conv in features)
def generate_gradcam(model, input_tensor, target_class=None):
    features = []
    gradients = []

    # last layer in vgg.features is ReLU; use previous as the conv layer
    target_layer = model.features[-2] if isinstance(model.features[-1], nn.ReLU) else model.features[-1]

    def forward_hook(module, inp, out):
        features.append(out)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    h1 = target_layer.register_forward_hook(forward_hook)
    h2 = target_layer.register_full_backward_hook(backward_hook)

    # Forward pass
    output = model(input_tensor)
    pred_class = output.argmax(dim=1).item() if target_class is None else target_class

    # Backward pass
    model.zero_grad()
    output[0, pred_class].backward()

    # Extract hooked data
    # (B,C,H,W); we have batch size 1
    grads_val = gradients[0].detach().cpu().numpy()[0]
    fmap = features[0].detach().cpu().numpy()[0]

    # Compute Grad-CAM
    weights = np.mean(grads_val, axis=(1, 2))
    cam = np.zeros(fmap.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * fmap[i]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam -= cam.min()
    cam /= (cam.max() + 1e-8)

    h1.remove()
    h2.remove()
    return cam

# Overlay Grad-CAM on original image (unchanged; now using RGB input)
def overlay_cam_on_image(img, cam):
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    img_np = np.array(img.convert("RGB").resize((224, 224)))
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    result = (0.4 * heatmap + 0.6 * img_bgr).astype(np.uint8)
    return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

# Streamlit App UI (unchanged)
st.title("🩻 Chest X-Ray Pneumonia Detection")
st.write("Upload a chest X-ray image to classify it as **Normal** or **Pneumonia**, with Grad-CAM visualization.")

uploaded_file = st.file_uploader("📤 Upload an X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Load model and preprocess image
    model, idx_to_class = load_model()
    input_tensor = preprocess_image(image)

    # Prediction
    device = next(model.parameters()).device
    # Prediction only
    with torch.no_grad():
        outputs = model(input_tensor.to(device))
        _, predicted = torch.max(outputs, 1)
        prediction = idx_to_class[str(predicted.item())]
        st.markdown(f"### 🧠 Prediction: **{prediction}**")

    # Generate Grad-CAM
    cam = generate_gradcam(model, input_tensor.to(device))
    cam_image = overlay_cam_on_image(image, cam)
    st.image(cam_image, caption="🔍 Grad-CAM Visualization", use_column_width=True)
