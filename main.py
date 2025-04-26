import torch
from torch import nn
import torchvision
from PIL import Image
from torchvision import transforms
import os
import streamlit as st
from typing import List, Tuple, Dict

device = "cuda" if torch.cuda.is_available() else "cpu"

# class_names = sorted(next(os.walk("data/train"))[1])
class_names = ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]

weights = torchvision.models.ResNet50_Weights.DEFAULT
model = torchvision.models.resnet50(weights=weights)
model.fc = torch.nn.Sequential(
    nn.Dropout(p=0.2),
    nn.Linear(in_features=2048, out_features=len(class_names))
).to(device)

transform = weights.transforms()

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

    model.to(device)

    model.eval()

    with torch.inference_mode():
        transformed_img = img_trans(img).unsqueeze(0).to(device)
        target_image_pred = model(transformed_img)
        target_image_pred_probs = torch.softmax(target_image_pred, dim=1)
        target_image_pred_class = torch.argmax(target_image_pred_probs, dim=1)

    return class_names[target_image_pred_class.item()]


# print(pred_image(model=model,
#            image_path="data/val/MildDemented/mildDem665.jpg",
#            class_names=class_names,
#            image_size=(232, 232),
#            device=device))

st.title("Medical Image Classification")
st.write("Upload an image and see the model's prediction.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Imagine incarcata", use_container_width=True)

    if st.button("Predict"):
        with st.spinner("Se proceseaza imaginea..."):
            pred_class = pred_image(model=model, img=img, class_names=class_names, transform=transform)
        st.success(f"Prediction: {pred_class}")