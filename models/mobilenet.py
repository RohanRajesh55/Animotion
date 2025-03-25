import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
import numpy as np
import cv2

class MobileNetEmotion:
    """
    Loads MobileNetV2 pre-trained on AffectNet for real-time facial emotion recognition.
    """
    def __init__(self, model_path="models/mobilenet_affectnet.pth", device="cpu"):
        self.device = torch.device(device)
        
        # Load pre-trained MobileNetV2
        self.model = models.mobilenet_v2(pretrained=False)
        self.model.classifier[1] = nn.Linear(self.model.last_channel, 8)  # AffectNet has 8 emotion classes
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()  # Set model to evaluation mode
        
        # Define image transformation
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize grayscale images
        ])

        # Emotion labels (AffectNet)
        self.emotion_labels = ["Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Anger", "Contempt"]

    def preprocess_face(self, face_roi):
        """
        Preprocess face image before feeding it to the model.
        
        :param face_roi: Cropped grayscale face (numpy array)
        :return: Torch tensor image
        """
        face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        face_roi = self.transform(face_roi).unsqueeze(0).to(self.device)  # Add batch dimension
        return face_roi

    def predict_emotion(self, face_roi):
        """
        Predicts emotion from the given face ROI.

        :param face_roi: Cropped face image (numpy array)
        :return: (emotion_label, confidence_score)
        """
        face_tensor = self.preprocess_face(face_roi)

        with torch.no_grad():
            output = self.model(face_tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)

        emotion_label = self.emotion_labels[predicted_idx.item()]
        return emotion_label, confidence.item()
