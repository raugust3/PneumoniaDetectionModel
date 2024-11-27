import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, roc_auc_score
import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score

# Device configuration (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the directories (set correct paths to your dataset)
train_dir = "train"
val_dir = "val"
test_dir = "test"
model_path = "pneumonia_detection_model.pth"  # Model file path

# Define transformations (we resize images to 224x224, normalize, and augment training data)
transform_train = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Ensure grayscale (1 channel)
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),  # Data augmentation
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Single-channel normalization
])

transform_val_test = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Ensure grayscale
    transforms.Resize((224, 224)),  # Resize validation and test images as well
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Load datasets
train_dataset = datasets.ImageFolder(root=train_dir, transform=transform_train)
val_dataset = datasets.ImageFolder(root=val_dir, transform=transform_val_test)
test_dataset = datasets.ImageFolder(root=test_dir, transform=transform_val_test)

# Data loaders for batching
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"Training Dataset Size: {len(train_dataset)}")
print(f"Validation Dataset Size: {len(val_dataset)}")
print(f"Test Dataset Size: {len(test_dataset)}")



class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2,1,kernel_size=7, stride=1, padding=3)

    def forward(self, x):
       avg_out = torch.mean(x, dim=1, keepdim=True)
       max_out, _ = torch.max(x, dim=1, keepdim=True)
       combined = torch.cat([avg_out, max_out], dim=1)
       attention_map = torch.sigmoid(self.conv(combined))
       return attention_map * x

# Define CNN model for pneumonia detection
class PneumoniaCNN(nn.Module):
    def __init__(self):
        super(PneumoniaCNN, self).__init__()
        # First convolutional layer: in_channels=1 for grayscale images
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)

        # Spatial attention layers
        self.spatial_attention1 = SpatialAttention()
        self.spatial_attention2 = SpatialAttention()
        self.spatial_attention3 = SpatialAttention()

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, 1)  # Binary classification: pneumonia or no pneumonia

        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Dropout layer to prevent overfitting
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.spatial_attention1(x)  # Apply spatial attention

        x = self.pool(F.relu(self.conv2(x)))
        x = self.spatial_attention2(x)
        x = self.pool(F.relu(self.conv3(x)))
        x = self.spatial_attention3(x)

        # Flatten the image for the fully connected layer
        x = x.view(-1, 128 * 28 * 28)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)  # No sigmoid activation here since BCEWithLogitsLoss handles it

        return x

# Instantiate the model
model = PneumoniaCNN().to(device)

# Check if the model exists
if os.path.exists(model_path):
    print("Model exists. Loading the model from disk...")
    model.load_state_dict(torch.load(model_path))
else:
    print("Model does not exist. Training the model...")

    # Define loss function and optimizer
    # Calculate class weights based on dataset size
    num_normal = 2838
    num_pneumonia = 4225
    total_samples = num_normal + num_pneumonia
    weight_for_normal = total_samples / (2 * num_normal)
    weight_for_pneumonia = total_samples / (2 * num_pneumonia)

    weights = torch.tensor([weight_for_normal, weight_for_pneumonia]).to(device)

    # Instantiate the model
    model = PneumoniaCNN().to(device)

    # Define the loss function with pos_weight
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight_for_pneumonia).to(device))

    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop with validation
    num_epochs = 10
    for epoch in range(num_epochs):
        epoch_start_time = time.time()  # Record the start time of the epoch

        model.train()  # Set model to training mode
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device).float().view(-1, 1)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Validation phase
        model.eval()  # Set model to evaluation mode
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for val_images, val_labels in val_loader:
                val_images, val_labels = val_images.to(device), val_labels.to(device).float().view(-1, 1)
                val_outputs = model(val_images)
                val_loss += criterion(val_outputs, val_labels).item()

                predicted = (val_outputs > 0.5).float()
                total += val_labels.size(0)
                correct += (predicted == val_labels).sum().item()

        val_accuracy = 100 * correct / total

        epoch_end_time = time.time()  # Record the end time of the epoch
        epoch_duration = epoch_end_time - epoch_start_time  # Calculate duration

        print(f"Epoch [{epoch + 1}/{num_epochs}], "
              f"Training Loss: {running_loss / len(train_loader):.4f}, "
              f"Validation Loss: {val_loss / len(val_loader):.4f}, "
              f"Validation Accuracy: {val_accuracy:.2f}%, "
              f"Time Taken: {epoch_duration:.2f} seconds")

    # Save the model after training
    print("Training completed. Saving the model to disk...")
    torch.save(model.state_dict(), model_path)

# Test the model on the test set
def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device).float().view(-1, 1)
            outputs = model(images)
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test Accuracy of the model: {100 * correct / total:.2f}%")

# Call the function to test the model
test_model(model, test_loader)

# Evaluate precision, recall, and F1-score
def evaluate_metrics(model, test_loader):
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device).float().view(-1, 1)
            outputs = model(images)
            predicted = (outputs > 0.5).float()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1-score: {f1:.2f}")

# Call the function to evaluate metrics
evaluate_metrics(model, test_loader)

# Plot confusion matrix
def plot_confusion_matrix(model, test_loader):
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device).float().view(-1, 1)
            outputs = model(images)
            predicted = (outputs > 0.5).float()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "Pneumonia"], yticklabels=["Normal", "Pneumonia"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# Call the function to plot the confusion matrix
plot_confusion_matrix(model, test_loader)



def plot_roc_curve(model, test_loader):
    model.eval()
    all_labels = []
    all_probs = []  # Store predicted probabilities

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device).float().view(-1, 1)
            outputs = model(images)
            probabilities = torch.sigmoid(outputs).cpu().numpy()  # Apply sigmoid to get probabilities
            all_probs.extend(probabilities)
            all_labels.extend(labels.cpu().numpy())

    # Convert lists to numpy arrays
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Compute ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    auc = roc_auc_score(all_labels, all_probs)

    # Plot the ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc:.2f})")
    plt.plot([0, 1], [0, 1], 'r--', label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.show()

# Call the function to plot the ROC curve
plot_roc_curve(model, test_loader)



def plot_precision_recall_curve(model, test_loader):
    model.eval()
    all_labels = []
    all_probs = []  # Store predicted probabilities

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device).float().view(-1, 1)
            outputs = model(images)  # Raw logits
            probabilities = torch.sigmoid(outputs).cpu().numpy()  # Convert logits to probabilities
            all_probs.extend(probabilities.flatten())  # Flatten to 1D
            all_labels.extend(labels.cpu().numpy())

    # Convert lists to numpy arrays
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Compute precision-recall curve
    precision, recall, thresholds = precision_recall_curve(all_labels, all_probs)

    # Compute average precision score
    avg_precision = average_precision_score(all_labels, all_probs)

    # Plot Precision-Recall curve
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f"Precision-Recall Curve (AP = {avg_precision:.2f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.show()

# Call the function to plot the Precision-Recall curve
plot_precision_recall_curve(model, test_loader)
