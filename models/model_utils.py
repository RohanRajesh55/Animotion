import torch
import numpy as np
import cv2
import torchvision.transforms as transforms

def load_model(model_class, model_path, device="cpu"):
    """
    Loads a pre-trained model.

    :param model_class: Class of the model to load (e.g., MobileNet, ResEmoNet)
    :param model_path: Path to the model weights file
    :param device: Device to load the model on ("cpu" or "cuda")
    :return: Loaded model instance
    """
    device = torch.device(device)
    model = model_class()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()  # Set to evaluation mode
    return model

def preprocess_face(face_roi, size=(224, 224)):
    """
    Prepares a face image for deep learning models.

    :param face_roi: Cropped face image (numpy array)
    :param size: Target size for the model input
    :return: Preprocessed image tensor
    """
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize grayscale images
    ])
    face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    return transform(face_roi).unsqueeze(0)  # Add batch dimension

def get_top_prediction(output, labels):
    """
    Returns the top predicted class label and confidence score.

    :param output: Model output tensor
    :param labels: List of class labels
    :return: (Predicted label, confidence score)
    """
    probabilities = torch.softmax(output, dim=1)
    confidence, predicted_idx = torch.max(probabilities, 1)
    return labels[predicted_idx.item()], confidence.item()
