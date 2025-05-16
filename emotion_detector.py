import cv2
import torch
from torchvision import transforms
from PIL import Image
from model_loader import load_pretrained_mobilenet_v2_emotion

# Path to the original checkpoint used by the model loader (from d-li14)
CHECKPOINT_PATH = r"C:\AniMotion\models\mobilenetv2-c5e733a8.pth"
# Emotion labels (ordered exactly as in your training for emotion recognition)
EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model = None

# Preprocessing pipeline for 224x224 RGB images using ImageNet normalization.
_preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def _load_model():
    global _model
    if _model is None:
        _model = load_pretrained_mobilenet_v2_emotion(CHECKPOINT_PATH, _device)
    return _model

def predict_emotion(face_roi):
    """
    Predicts the emotion from a face region-of-interest (ROI).
    :param face_roi: A face image in BGR format.
    :return: Predicted emotion label (string).
    """
    if face_roi is None or face_roi.size == 0:
        return "Neutral"
    try:
        model = _load_model()
        # Convert from BGR (OpenCV) to RGB (PIL)
        rgb_image = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        input_tensor = _preprocess(pil_image).unsqueeze(0).to(_device)
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
        pred_idx = int(torch.argmax(probabilities, dim=1).item())
        return EMOTION_LABELS[pred_idx] if 0 <= pred_idx < len(EMOTION_LABELS) else "Unknown"
    except Exception as e:
        print("Error in predict_emotion:", e)
        return "Neutral"