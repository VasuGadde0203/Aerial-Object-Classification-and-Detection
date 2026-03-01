import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)

    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, 1)
    )

    model.load_state_dict(torch.load("models/bird_drone_resnet18.pth", map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model()

# -----------------------------
# Image Transform
# -----------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

# -----------------------------
# UI
# -----------------------------
st.title("🛩️ Aerial Object Classification")
st.write("Upload an image to classify as Bird or Drone")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probability = torch.sigmoid(output).item()

    if probability > 0.5:
        prediction = "Drone"
        confidence = probability
    else:
        prediction = "Bird"
        confidence = 1 - probability

    st.subheader(f"Prediction: {prediction}")
    st.write(f"Confidence: {confidence:.4f}")