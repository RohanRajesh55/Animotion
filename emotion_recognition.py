import asyncio
from deepface import DeepFace
import logging

def analyze_emotion_sync(face_roi):
    """
    Synchronously analyze emotion using DeepFace.
    :param face_roi: The region of interest (face) from the video feed.
    :return: The detected emotion.
    """
    try:
        # Perform emotion analysis with enforce_detection set to False
        analysis = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
        emotion = analysis[0]['dominant_emotion']
        # Handle possible error conditions
        if emotion in ["None", "No Emotion", ""]:
            emotion = "Neutral"
        return emotion
    except Exception as e:
        logging.error(f"Error during emotion analysis: {e}")
        return "Neutral"

async def analyze_emotion_async(face_roi) -> str:
    """
    Asynchronously analyze emotion using DeepFace to prevent blocking.
    :param face_roi: The region of interest (face) from the video feed.
    :return: The detected emotion.
    """
    return await asyncio.to_thread(analyze_emotion_sync, face_roi)
