import torch
import torch.nn as nn
import numpy as np

class LSTMGestureModel(nn.Module):
    """
    LSTM-based model for gesture recognition using sequential hand landmarks.
    """
    def __init__(self, input_size=42, hidden_size=64, num_layers=2, output_size=5):
        """
        Initialize the LSTM model.

        :param input_size: Number of input features (landmark coordinates)
        :param hidden_size: Number of hidden units in LSTM
        :param num_layers: Number of LSTM layers
        :param output_size: Number of gesture classes
        """
        super(LSTMGestureModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)  # Softmax for classification

    def forward(self, x):
        """
        Forward pass through the LSTM network.

        :param x: Input tensor (batch, sequence_length, input_size)
        :return: Gesture class probabilities
        """
        lstm_out, _ = self.lstm(x)  # Get LSTM outputs
        lstm_out = lstm_out[:, -1, :]  # Take only the last output
        output = self.fc(lstm_out)  # Fully connected layer
        return self.softmax(output)  # Return probability distribution


# ===========================
#         HELPERS
# ===========================

def preprocess_gesture_sequence(landmark_sequences):
    """
    Preprocesses a sequence of hand landmarks for LSTM input.

    :param landmark_sequences: List of hand landmark sequences [(x1, y1, x2, y2, ..., x21, y21)]
    :return: Normalized tensor ready for model input
    """
    # Convert to NumPy array and normalize values between -1 and 1
    sequence_array = np.array(landmark_sequences, dtype=np.float32)
    sequence_array = sequence_array / np.max(np.abs(sequence_array), axis=0)  # Normalize

    # Convert to PyTorch tensor (batch, sequence_length, input_size)
    sequence_tensor = torch.tensor(sequence_array).unsqueeze(0)  # Add batch dimension

    return sequence_tensor


def load_lstm_model(model_path="models/lstm_gesture.pth"):
    """
    Loads the trained LSTM model.

    :param model_path: Path to the trained model weights
    :return: Loaded model
    """
    model = LSTMGestureModel()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()  # Set to evaluation mode
    return model
