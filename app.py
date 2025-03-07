import streamlit as st
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
from PIL import Image
import os
from class_labels import class_labels  # Import labels from external file

# Define paths
MODEL_PATH = "model.pth"

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)  # Load ResNet18 model
num_classes = len(class_labels)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)  # Modify last layer
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))  # Load trained weights
model.to(device)
model.eval()  # Set model to evaluation mode

# Define transformation for input images
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Resize to match training input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Use same normalization as training
])

# Streamlit UI
st.title("🚦 German Traffic Sign Classification")
st.write("Upload a traffic sign image, and the model will predict its category.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)  # Fix applied

    # Preprocess image
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension

    # Make prediction
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        predicted_class = class_labels[predicted.item()]  # Get class name from dictionary

    # Display prediction
    st.subheader(f"Prediction: **{predicted_class}** ✅")
