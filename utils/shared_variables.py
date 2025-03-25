import threading

class SharedVariables:
    """Manages shared facial tracking variables for real-time processing."""
    def __init__(self):
        self.lock = threading.Lock()
        self.eye_blink = "No"
        self.mouth_open = "No"
        self.lip_sync = "No"
        self.head_pose = "Neutral"
    
    def update(self, eye_blink=None, mouth_open=None, lip_sync=None, head_pose=None):
        """Safely updates shared variables."""
        with self.lock:
            if eye_blink is not None:
                self.eye_blink = eye_blink
            if mouth_open is not None:
                self.mouth_open = mouth_open
            if lip_sync is not None:
                self.lip_sync = lip_sync
            if head_pose is not None:
                self.head_pose = head_pose
    
    def get_data(self):
        """Retrieves shared variables in a thread-safe manner."""
        with self.lock:
            return {
                "eye_blink": self.eye_blink,
                "mouth_open": self.mouth_open,
                "lip_sync": self.lip_sync,
                "head_pose": self.head_pose
            }