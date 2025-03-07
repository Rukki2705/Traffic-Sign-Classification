import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import os

# Define dataset paths
DATASET_PATH = "./GTSRB/Preprocessed"
TRAIN_DATA_PATH = os.path.join(DATASET_PATH, "Train_processed.pt")
TEST_DATA_PATH = os.path.join(DATASET_PATH, "Test_processed.pt")
PLOT_DIR = "./plots"
MODEL_PATH = "model.pth"

# Create a directory for saving plots
os.makedirs(PLOT_DIR, exist_ok=True)

# Load preprocessed training data (images + labels)
train_data, train_labels = torch.load(TRAIN_DATA_PATH)

# Load preprocessed test data (only images, no labels)
test_data = torch.load(TEST_DATA_PATH)

# Hyperparameters
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 10

# Convert training data to Dataset and DataLoader
train_dataset = TensorDataset(train_data, train_labels)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Number of classes
num_classes = len(torch.unique(train_labels))

# Load Pretrained ResNet18
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)  # Modify final layer for traffic sign classes

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define Loss Function & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Lists to store metrics
train_acc_list = []
train_loss_list = []

print("🚀 Training Started...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(train_loader)
    train_accuracy = 100 * correct / total
    train_loss_list.append(avg_loss)
    train_acc_list.append(train_accuracy)

    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")

# Save the trained model
torch.save(model.state_dict(), MODEL_PATH)
print(f"✅ Model training completed and saved as {MODEL_PATH}")

# Plot Training Loss & Accuracy
plt.figure(figsize=(10,4))

# Training Loss Plot
plt.subplot(1,2,1)
plt.plot(range(EPOCHS), train_loss_list, label="Train Loss", color='red')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.legend()
plt.savefig(os.path.join(PLOT_DIR, "training_loss.png"))  # Save plot

# Training Accuracy Plot
plt.subplot(1,2,2)
plt.plot(range(EPOCHS), train_acc_list, label="Train Accuracy", color='blue')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training Accuracy")
plt.legend()
plt.savefig(os.path.join(PLOT_DIR, "train_accuracy.png"))  # Save plot

plt.show()

print(f"✅ Training Loss & Accuracy plots saved in {PLOT_DIR}/")
