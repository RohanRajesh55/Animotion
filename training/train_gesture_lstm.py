import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Define LSTM model for gesture prediction
class GestureLSTM(nn.Module):
    def __init__(self, input_size=10, hidden_size=50, output_size=3):
        super(GestureLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn[-1])
        return out

# Load preprocessed gesture dataset
data = torch.load("data/gesture_sequences_processed.pt")
X_train = data  # Processed tensor data

# Generate random labels (for now) with same length as X_train
y_train = torch.randint(0, 3, (X_train.shape[0],))  # 3 classes (Blink, Nod, Neutral)

# Initialize model
model = GestureLSTM(input_size=X_train.shape[2], output_size=len(set(y_train.numpy())))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 20
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Save trained model efficiently
torch.save(model.state_dict(), "models/gesture_lstm.pth", _use_new_zipfile_serialization=False)
print("Gesture LSTM model training complete & saved.")