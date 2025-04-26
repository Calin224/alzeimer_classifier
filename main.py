import torch
from torch import nn
import torchvision
from PIL import Image
from torchvision import transforms
import os
import streamlit as st
import urllib.request
from typing import List, Tuple

device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_URL = "https://huggingface.co/Calin224/alzeimer-resnet50/resolve/main/resnet50_model.pth"
MODEL_PATH = "models/resnet50_model.pth"

class_names = ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]

weights = None
model = torchvision.models.resnet50(weights=weights)
model.fc = torch.nn.Sequential(
    nn.Dropout(p=0.2),
    nn.Linear(in_features=2048, out_features=len(class_names))
).to(device)

if not os.path.exists(MODEL_PATH):
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with st.spinner("Downloading model from HuggingFace..."):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        st.success("Model downloaded successfully!")

model.load_state_dict(torch.load(MODEL_PATH, map_location=device), strict=False)
model.eval()

def pred_image(model: torch.nn.Module,
               img: Image.Image,
               class_names: list,
               image_size: Tuple[int, int] = (232, 232),
               transform: torchvision.transforms = None,
               device: torch.device = device):

    if transform is not None:
        img_trans = transform
    else:
        img_trans = transforms.Compose([
            transforms.Resize(size=(image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    with torch.inference_mode():
        transformed_img = img_trans(img).unsqueeze(0).to(device)
        output = model(transformed_img)
        probs = torch.softmax(output, dim=1)
        pred_class = torch.argmax(probs, dim=1)

    return class_names[pred_class.item()]

st.title("Medical Image Classification")
st.write("Upload an image and see the model's prediction.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Imagine încărcată", use_container_width=True)

    if st.button("Predict"):
        with st.spinner("Se procesează imaginea..."):
            pred_class = pred_image(model=model, img=img, class_names=class_names)
            if pred_class == "ModerateDemented":
                pred_class = "Alzheimer Mediu"
        st.success(f"Prediction: {pred_class}")
