import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# --------------------------------------------------
# Config
# --------------------------------------------------
DEVICE = torch.device("cpu")
MODEL_PATH = "best_resnet18_deepfake_v2.pth"

# --------------------------------------------------
# Model definition (must match training)
# --------------------------------------------------
class ResNetDeepfake(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet18(weights=None)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.backbone(x)

# --------------------------------------------------
# Load model (cached)
# --------------------------------------------------
@st.cache_resource
def load_model():
    model = ResNetDeepfake()
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=DEVICE)
    )
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

# --------------------------------------------------
# Preprocessing (same as training)
# --------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# --------------------------------------------------
# Streamlit UI
# --------------------------------------------------
st.title("Deepfake Image Detector")

file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if file:
    image = Image.open(file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    x = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logit = model(x)
        prob_fake = torch.sigmoid(logit).item()

    label = "Fake" if prob_fake > 0.5 else "Real"
    confidence = prob_fake if label == "Fake" else 1 - prob_fake

    st.subheader(f"Prediction: {label}")
    st.write(f"Confidence: {confidence * 100:.2f}%")
