#🚦Traffic Sign Classification
This project is a **Traffic Sign Classification system** using PyTorch and Streamlit. The trained model predicts the class of a traffic sign from an uploaded image.

---
## 📌 Features

Preprocessing of traffic sign images using OpenCV and Torch.

Deep learning model training using ResNet18.

Web-based Streamlit UI for user interaction.

Real-time image classification with pre-trained model.

---
## 🛠️ Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Rukki2705/Traffic-Sign-Classification.git
```
### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
### 3️⃣ Run the Application
```bash
streamlit run app.py
```
---


## 📚 How It Works
#### 1️⃣ The preprocessing script (preprocessing.py) loads, resizes, and normalizes images.
#### 2️⃣ The model training script (model.py) trains a ResNet18 model on the dataset.
#### 3️⃣ The trained model (model.pth) is loaded into the Streamlit application (app.py).
#### 4️⃣ The Streamlit UI allows users to upload an image, and the model predicts the traffic sign category.
