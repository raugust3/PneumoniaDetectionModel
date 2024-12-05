import torch
from PIL import Image
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F

# Define device configuration (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Define Spatial Attention Module
class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3)

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
        # Convolutional layers
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

        # Pooling and dropout layers
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.spatial_attention1(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.spatial_attention2(x)
        x = self.pool(F.relu(self.conv3(x)))
        x = self.spatial_attention3(x)

        # Flatten for fully connected layers
        x = x.view(-1, 128 * 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


# Function to load the model and make a prediction on a single image
def predict_pneumonia(image_path, model_path="pneumonia_detection_model.pth"):
    # Load the model
    model = PneumoniaCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device,weights_only=True))
    model.eval()  # Set to evaluation mode

    # Define the same transformations as used during training
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Load and preprocess the image
    image = Image.open(image_path).convert("L")  # Ensure grayscale
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to device

    # Make the prediction
    with torch.no_grad():
        output = model(image)
        probability = torch.sigmoid(output).item()  # Get probability for pneumonia

    # Interpret the result
    label = "Pneumonia" if probability > 0.5 else "Normal"
    confidence = probability if label == "Pneumonia" else 1 - probability
    return label, confidence


# Example usage
if __name__ == "__main__":
    image_path = "img.png" # Path to the testing image
    model_path = "pneumonia_detection_model.pth"  # Path to the trained model
    # Note: Each time the model is trained, the resulting curves may vary due to 
    # the random initialization and stochastic optimization inherent in the training process. 
    # Additionally, if the main script detects an existing pre-trained model, it may load 
    # this model, resulting in completely different outputs that do not reflect the current 
    # training process. For reference, a supplementary video has been included in the repository 
    # to demonstrate how the metrics were obtained.

    result = predict_pneumonia(image_path, model_path)
    label, confidence = result
    print(f"The model predicts: {label} with confidence {confidence:.2f}")
