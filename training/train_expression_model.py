import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision import models
from torchvision.models import MobileNet_V2_Weights

# Define transformations for preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load a subset of FER2013 dataset to reduce space usage
dataset_path = "/content/drive/MyDrive/Animotion/data/facial_expressions"
full_dataset = datasets.ImageFolder(root=f"{dataset_path}/train", transform=transform)
train_dataset = torch.utils.data.Subset(full_dataset, range(10000))  # Use only 10K images
val_dataset = datasets.ImageFolder(root=f"{dataset_path}/val", transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

# Load pre-trained MobileNetV2

model = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
model.classifier[1] = nn.Linear(model.last_channel, len(full_dataset.classes))  # Adjust output layer

# Define loss function & optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Save model without unnecessary checkpoint data
torch.save(model.state_dict(), "models/facial_expression_model.pth", _use_new_zipfile_serialization=False)
print("Facial expression model training complete & saved.")

# Evaluate model on validation set
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in val_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    accuracy = correct / total
    print(f"Validation Accuracy: {accuracy:.4f}")
    print("Facial expression model evaluation complete.")

# Save model without unnecessary checkpoint data
torch.save(model.state_dict(), "models/facial_expression_model.pth", _use_new_zipfile_serialization=False)
print("Facial expression model evaluation complete & saved.")
