import threading

class SharedVariables:
    """
    A thread-safe class to store shared variables for communication 
    between the main process and the WebSocket client.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.stop_flag = False
        self.data = {
            "eye_blink": "No",
            "mouth_open": "No",
            "lip_sync_active": "No",
            "head_pose": (0, 0, 0),  # Yaw, Pitch, Roll
            "gesture": None
        }

    def update(self, key, value):
        """
        Safely update a shared variable.
        
        :param key: Key of the variable to update
        :param value: New value
        """
        with self.lock:
            self.data[key] = value

    def get(self, key):
        """
        Safely retrieve a shared variable.
        
        :param key: Key of the variable to retrieve
        :return: The stored value
        """
        with self.lock:
            return self.data.get(key, None)

    def stop_websocket(self):
        """
        Sets the stop flag to terminate the WebSocket connection safely.
        """
        with self.lock:
            self.stop_flag = True

    def should_stop(self):
        """
        Checks whether the stop flag is set.
        
        :return: True if the process should stop, False otherwise
        """
        with self.lock:
            return self.stop_flag
