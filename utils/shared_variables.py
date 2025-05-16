from typing import Optional, List, Dict

class SharedVariables:
    """
    A container for shared variables used across the facial tracking and avatar animation pipeline.

    Attributes:
        ear_left (Optional[float]): Left Eye Aspect Ratio (EAR) value.
        ear_right (Optional[float]): Right Eye Aspect Ratio (EAR) value.
        mar (Optional[float]): Mouth Aspect Ratio (MAR) value.
        ebr_left (Optional[float]): Left Eyebrow Raise Ratio (EBR) value.
        ebr_right (Optional[float]): Right Eyebrow Raise Ratio (EBR) value.
        lip_sync_value (Optional[float]): Lip sync metric value.
        yaw (Optional[float]): Head pose yaw angle.
        pitch (Optional[float]): Head pose pitch angle.
        roll (Optional[float]): Head pose roll angle.
        faces (List[Dict]): List of dictionaries holding per-face metrics (for multi-face tracking).
        fps (Optional[float]): Frames per second, used for diagnostics.
        emotion (Optional[str]): Recognized dominant emotion (e.g., "happy", "sad", "neutral").
    """

    def __init__(self) -> None:
        self.ear_left: Optional[float] = None
        self.ear_right: Optional[float] = None
        self.mar: Optional[float] = None
        self.ebr_left: Optional[float] = None
        self.ebr_right: Optional[float] = None
        self.lip_sync_value: Optional[float] = None
        self.yaw: Optional[float] = None
        self.pitch: Optional[float] = None
        self.roll: Optional[float] = None
        self.faces: List[Dict] = []
        self.fps: Optional[float] = None
        self.emotion: Optional[str] = None

    def reset(self) -> None:
        """
        Reset all stored variables to their initial state.
        """
        self.ear_left = None
        self.ear_right = None
        self.mar = None
        self.ebr_left = None
        self.ebr_right = None
        self.lip_sync_value = None
        self.yaw = None
        self.pitch = None
        self.roll = None
        self.faces = []
        self.fps = None
        self.emotion = None

    def __repr__(self) -> str:
        return (
            f"SharedVariables(ear_left={self.ear_left}, ear_right={self.ear_right}, mar={self.mar}, "
            f"ebr_left={self.ebr_left}, ebr_right={self.ebr_right}, lip_sync_value={self.lip_sync_value}, "
            f"yaw={self.yaw}, pitch={self.pitch}, roll={self.roll}, fps={self.fps}, emotion={self.emotion})"
        )