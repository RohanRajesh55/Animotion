import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

# Load compressed gesture dataset
data = np.load("data/gesture_sequences.npz")
X = data["X"]

# Flatten the dataset for normalization
num_samples, num_landmarks, num_features = X.shape
X_flattened = X.reshape(num_samples, -1)

# Normalize data between 0 and 1
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_flattened)

# Reshape back to original structure
X_processed = X_scaled.reshape(num_samples, num_landmarks, num_features)

# Convert to PyTorch tensor
X_tensor = torch.tensor(X_processed, dtype=torch.float32)

torch.save(X_tensor, "data/gesture_sequences_processed.pt")

print("Gesture dataset preprocessing complete! Saved as PyTorch tensor.")