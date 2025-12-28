
import sys
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import json

# Define the model architecture matching the trained model
# NOTE: This MUST match the architecture used during training.
# Since I don't have the user's specific class, I'm using a placeholder.
# If this fails, the user needs to paste their model class here.

class DeepFakeCNN(nn.Module):
    def __init__(self):
        super(DeepFakeCNN, self).__init__()
        # Placeholder: standard simple CNN layers
        # The user MUST replace this with their actual model class if different
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 112 * 112, 2) # Assuming 224x224 input

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

def predict(image_path, model_path):
    try:
        # Load the model
        device = torch.device('cpu')
        
        # Try loading entire model first (if saved with torch.save(model))
        try:
            model = torch.load(model_path, map_location=device)
        except:
            # Fallback to state_dict (needs class definition above)
            model = DeepFakeCNN()
            model.load_state_dict(torch.load(model_path, map_location=device))
        
        model.eval()

        # Transform image
        transform = transforms.Compose([
            transforms.Resize((224, 224)), # Adjust size as needed
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image_tensor)
            _, predicted = torch.max(outputs, 1)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
        result = {
            "prediction": "Fake" if predicted.item() == 0 else "Real", # Adjust label mapping
            "confidence": probabilities[0][predicted.item()].item() * 100
        }
        print(json.dumps(result))

    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(json.dumps({"error": "Usage: python predict.py <image_path> <model_path>"}))
        sys.exit(1)
    
    predict(sys.argv[1], sys.argv[2])
