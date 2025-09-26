import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image

# using gpu if available.
if torch.cuda.is_available():
    device = "cuda"
else: 
    device = "cpu"
    
# ==============================
# 1️⃣ App Configuration
# ==============================
st.set_page_config(
    page_title="Cat vs Dog Classifier 🐱🐶",
    page_icon="🐶",
    layout="centered",
)

st.title("🐱🐶 Cat vs Dog Image Classifier | Philip Code Academy")
st.markdown("**Final Project – June Data Science Cohort 2025**")

st.markdown(
    """
    ### Welcome!
    This web application was built by **Philip Code Academy** as the **final project** for our **June 2025 Data Science Cohort** 🎓.

    It uses a deep learning model to classify images as either a **Cat** or a **Dog**.
    Upload an image below and let the model do the magic ✨.
    """
)


model_path = "resnet.pth"
# ==============================
# 2️⃣ Load Pre-trained Model
# ==============================
@st.cache_resource
def load_model():
    """Load a pre-trained ResNet18 model fine-tuned for 2 classes (Cat & Dog)."""
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    # Freezing the model before the input size.
    for params in model.parameters():
        params.requires_grad = False
    in_features = model.fc.in_features
    classifier = torch.nn.Sequential()
    classifier.append(torch.nn.Linear(in_features, 256))
    classifier.append(torch.nn.ReLU())
    classifier.append(torch.nn.Dropout())
    classifier.append(torch.nn.Linear(256, 2))
    model.fc = classifier # 2 classes: cat, dog

    # If you have a fine-tuned model saved locally:
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    # model.load_state_dict(torch.load("model/cat_dog_model.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()
class ConvertRGB:
    def __call__(self, img):
        if img.mode != "RGB":
            img.convert("RGB")
        return img
# ==============================
# 3️⃣ Image Preprocessing
# ==============================
transform = transforms.Compose([
    ConvertRGB(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4884, 0.4553, 0.4170], std=[0.2598, 0.2531, 0.2558])
])

# ==============================
# 4️⃣ Image Upload
# ==============================
uploaded_file = st.file_uploader("📤 Upload an image (JPEG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file) # .convert("RGB")
    st.image(image, caption="📸 Uploaded Image", width="content")

    # ==============================
    # 5️⃣ Prediction
    # ==============================
    st.subheader("🔎 Prediction Result")

    with st.spinner("Classifying... ⏳"):
        img_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)

        labels = ["Cat 🐱", "Dog 🐶"]
        predicted_label = labels[predicted.item()]
        confidence_score = confidence.item() * 100

        # ==============================
        # 6️⃣ Display Results
        # ==============================
        st.success(f"✅ Prediction: **{predicted_label}**")
        st.info(f"📊 Confidence: {confidence_score:.2f}%")

        # Optional Explanation
        if confidence_score < 70:
            st.warning("⚠️ The confidence is relatively low. Try uploading a clearer image for better accuracy.")

else:
    st.info("👆 Upload a cat or dog image to start classification.")

# ==============================
# 7️⃣ Footer
# ==============================
st.markdown("---")
st.markdown(
    """
    🔬 **About this project:**  
    Developed by **Philip Code Academy** as part of the **June 2025 Data Science Cohort**.  
    Built with [Streamlit](https://streamlit.io/) and [PyTorch](https://pytorch.org/), leveraging a fine-tuned **ResNet50 CNN** for image classification.
    """
)

