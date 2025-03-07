import os
import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
from tqdm import tqdm
from pathlib import Path
import pandas as pd

# Define paths
DATASET_PATH = "./GTSRB"
TRAIN_CSV = os.path.join(DATASET_PATH, "Train.csv")
TEST_CSV = os.path.join(DATASET_PATH, "Test.csv")
PREPROCESSED_DIR = os.path.join(DATASET_PATH, "Preprocessed")

# Create Preprocessed Directory if not exists
Path(PREPROCESSED_DIR).mkdir(parents=True, exist_ok=True)

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((32, 32)),  # Resize images
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
])

def preprocess_images(csv_file, output_file, is_test=False):
    df = pd.read_csv(csv_file)
    images = []
    labels = []

    for index, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {csv_file}"):
        image_path = os.path.join(DATASET_PATH, row["Path"])  # Fix: Use full path from CSV

        if not os.path.exists(image_path):
            print(f"⚠️ Warning: File not found - {image_path}")
            continue

        img = cv2.imread(image_path)
        if img is None:
            print(f"⚠️ Warning: Could not read {image_path}")
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        img_tensor = transform(img)

        images.append(img_tensor)
        if not is_test:
            labels.append(row["ClassId"])  # Assign label only for Train

    if len(images) == 0:
        print(f"❌ Error: No images processed from {csv_file}!")
        return

    # Convert to tensors
    images_tensor = torch.stack(images)

    if not is_test:
        labels_tensor = torch.tensor(labels)
        torch.save((images_tensor, labels_tensor), os.path.join(PREPROCESSED_DIR, output_file))
    else:
        torch.save(images_tensor, os.path.join(PREPROCESSED_DIR, output_file))

    print(f"✅ Preprocessing completed for {output_file}. Saved in {PREPROCESSED_DIR}.")

# Run preprocessing
preprocess_images(TRAIN_CSV, "Train_processed.pt", is_test=False)
preprocess_images(TEST_CSV, "Test_processed.pt", is_test=True)
