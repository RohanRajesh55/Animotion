import torch
import os
import logging
from models.MobileNetV2_Emotion import MobileNetV2_Emotion

logger = logging.getLogger(__name__)

def load_pretrained_mobilenet_v2_emotion(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    """
    Loads the MobileNetV2_Emotion model with pretrained ImageNet weights (and modified for emotion recognition).
    :param checkpoint_path: Path to the ImageNet pretrained checkpoint from d-li14.
    :param device: Device to load the model on.
    :return: MobileNetV2_Emotion model.
    """
    if not os.path.exists(checkpoint_path):
        logger.error("Checkpoint file not found: %s", checkpoint_path)
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    try:
        # For this model, our MobileNetV2_Emotion class already loads from the checkpoint
        # during initialization if pretrained=True.
        model = MobileNetV2_Emotion(num_classes=7, width_mult=1.0, pretrained=True)
    except Exception as e:
        logger.exception("Error loading checkpoint: %s", e)
        raise RuntimeError("Failed to load model checkpoint.") from e

    model.to(device)
    model.eval()
    return model

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_pretrained_mobilenet_v2_emotion(r"C:\AniMotion\models\mobilenetv2-c5e733a8.pth", device)
    print("Model loaded successfully.")