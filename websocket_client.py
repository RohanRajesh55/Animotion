import asyncio
import websockets
import json
import configparser
import logging
import threading
from typing import Any
from utils.shared_variables import SharedVariables

# Load configuration
config = configparser.ConfigParser()
config.read('config.ini')

# Retrieve WebSocket configuration parameters.
VTS_WS_URL: str = config.get('WebSocket', 'VTS_WS_URL', fallback='ws://localhost:8001')
AUTH_TOKEN: str = config.get('WebSocket', 'AUTH_TOKEN', fallback='your_auth_token')

# Set up logging.
logger = logging.getLogger(__name__)

# Function to log connection status more clearly.
def log_connection_status(is_connected: bool):
    if is_connected:
        logger.info("Connected to VTube Studio WebSocket.")
    else:
        logger.warning("Failed to connect to WebSocket. Retrying...")

async def websocket_task(shared_vars: SharedVariables, data_lock: threading.Lock) -> None:
    """
    Asynchronous task to maintain a persistent WebSocket connection and send expression data.

    This function continuously attempts to connect to the WebSocket endpoint. Upon a successful
    connection, it sends an authentication message, then enters a loop where it transmits
    expression data at a controlled rate (approximately 20 messages per second). It implements
    robust error handling with exponential backoff to recover from connection errors.

    Parameters:
        shared_vars (SharedVariables): Shared container holding latest facial metrics.
        data_lock (threading.Lock): A lock ensuring thread-safe access to shared_vars.
    """
    reconnect_delay: int = 3  # Initial reconnect delay in seconds.
    
    while True:
        try:
            async with websockets.connect(VTS_WS_URL) as websocket:
                log_connection_status(True)

                # Construct and send the authentication message.
                auth_message = {
                    "apiName": "VTubeStudioPublicAPI",
                    "apiVersion": "1.0",
                    "requestID": "AuthRequest",
                    "messageType": "AuthenticationRequest",
                    "data": {
                        "pluginName": "FacialTracker",
                        "pluginDeveloper": "YourName",
                        "authenticationToken": AUTH_TOKEN
                    }
                }
                await websocket.send(json.dumps(auth_message))
                response = await websocket.recv()
                logger.info("Authentication Response: %s", response)

                # Reset the reconnect delay on a successful connection.
                reconnect_delay = 3

                # Main loop to send expression data.
                while True:
                    with data_lock:
                        # Only send data if all required shared variables are set.
                        if None not in (
                            shared_vars.ear_left,
                            shared_vars.ear_right,
                            shared_vars.mar,
                            shared_vars.ebr_left,
                            shared_vars.ebr_right,
                            shared_vars.lip_sync_value,
                            shared_vars.yaw,
                            shared_vars.pitch,
                            shared_vars.roll
                        ):
                            expression_data = {
                                "apiName": "VTubeStudioPublicAPI",
                                "apiVersion": "1.0",
                                "requestID": "ExpressionRequest",
                                "messageType": "ExpressionActivationRequest",
                                "data": {
                                    "expressions": [
                                        {"id": "EyeLeftBlink", "value": shared_vars.ear_left},
                                        {"id": "EyeRightBlink", "value": shared_vars.ear_right},
                                        {"id": "MouthOpen", "value": shared_vars.mar},
                                        {"id": "BrowLeftY", "value": shared_vars.ebr_left},
                                        {"id": "BrowRightY", "value": shared_vars.ebr_right},
                                        {"id": "LipSync", "value": shared_vars.lip_sync_value},
                                        {"id": "HeadYaw", "value": shared_vars.yaw},
                                        {"id": "HeadPitch", "value": shared_vars.pitch},
                                        {"id": "HeadRoll", "value": shared_vars.roll}
                                    ]
                                }
                            }
                            await websocket.send(json.dumps(expression_data))
                    # Control the sending rate (approximately 20 messages per second).
                    await asyncio.sleep(0.05)
        except Exception as e:
            log_connection_status(False)
            logger.error("WebSocket connection error: %s", e)
            await asyncio.sleep(reconnect_delay)
            # Exponential backoff: double the delay each time, up to a maximum of 30 seconds.
            reconnect_delay = min(reconnect_delay * 2, 30)

def start_websocket(shared_vars: SharedVariables, data_lock: threading.Lock) -> None:
    """
    Entry point to start the asynchronous WebSocket task.

    This function initiates the event loop to run the websocket_task. It is intended to be
    called from a separate thread so that the main application can continue running independently.

    Parameters:
        shared_vars (SharedVariables): Shared container holding latest facial metrics.
        data_lock (threading.Lock): A lock ensuring thread-safe access to shared_vars.
    """
    asyncio.run(websocket_task(shared_vars, data_lock))
