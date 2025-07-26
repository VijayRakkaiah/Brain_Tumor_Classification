import streamlit as st
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image

# Page config
st.set_page_config(
    page_title="Brain Tumor Classifier",
    page_icon="assets/brian_tumor.png",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load trained model
def load_best_model(model_path, num_classes):
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

# Predict function
def predict_image(image, model, class_names):
    img = Image.open(image).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)

    predicted_class = class_names[predicted.item()]
    return predicted_class

# Load model
model_path = "./resnet50_best.pt"
class_names = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
model = load_best_model(model_path, len(class_names)).to(device)

# Title
st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@700&display=swap" rel="stylesheet">
    <style>
    .custom-title {
        font-family: 'Poppins', sans-serif;
        font-size: 42px;
        color: #008080;
        text-align: center;
        margin-top: 20px;
        margin-bottom: 20px;
    }
    </style>
    <div class="custom-title">🧠 Brain Tumor Classifier</div>
""", unsafe_allow_html=True)


# Two-column layout
col1, col2 = st.columns(2)

# File upload (left)
with col1:
    uploaded_file = st.file_uploader("📤 Upload MRI Image", type=["jpg", "jpeg", "png"])

    # Predict button
    st.markdown("""
        <style>
            div.stButton > button {
                width: 100%;
                height: 45px;
                font-size: 18px;
                font-weight: bold;
                background-color: #20B2AA;
                color: white;
            }
        </style>
    """, unsafe_allow_html=True)
    prediction = None
    if uploaded_file is not None:
        if st.button("🔍 Predict"):
            with st.spinner("Classifying..."):
                prediction = predict_image(uploaded_file, model, class_names)

    # Show prediction
    if prediction:
        st.success(f"✅ **Predicted Class:** {prediction}")

# Preview image (right)
with col2:
    if uploaded_file is not None:
        try:
            img = Image.open(uploaded_file).convert("RGB")
            st.image(img, caption="MRI Image Preview", use_container_width=True)
        except:
            st.error("⚠️ Error displaying image.")

# About section
with st.expander("ℹ️ About"):
    st.markdown("""
        This app uses a **ResNet-50** deep learning model to classify MRI brain scans into:
        - Glioma Tumor
        - Meningioma Tumor
        - No Tumor
        - Pituitary Tumor
    """)