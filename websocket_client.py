# websocket_client.py

import asyncio
import websockets
import json
import configparser
import logging
from utils.shared_variables import SharedVariables

# Load configuration
config = configparser.ConfigParser()
config.read('config.ini')

# Retrieve WebSocket configuration
VTS_WS_URL = config.get('WebSocket', 'VTS_WS_URL', fallback='ws://localhost:8001')
AUTH_TOKEN = config.get('WebSocket', 'AUTH_TOKEN', fallback='your_auth_token')

# Set up logging
logger = logging.getLogger(__name__)

async def websocket_task(shared_vars, data_lock):
    """
    Asynchronous task to handle WebSocket communication.
    """
    while True:
        try:
            async with websockets.connect(VTS_WS_URL) as websocket:
                logger.info("Connected to VTube Studio WebSocket.")

                # Authenticate
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
                logger.info(f"Authentication Response: {response}")

                while True:
                    with data_lock:
                        if None not in (shared_vars.ear_left, shared_vars.ear_right, shared_vars.mar,
                                        shared_vars.ebr_left, shared_vars.ebr_right, shared_vars.lip_sync_value,
                                        shared_vars.yaw, shared_vars.pitch, shared_vars.roll):
                            # Prepare expression data
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
                            # Send expression data
                            await websocket.send(json.dumps(expression_data))
                    await asyncio.sleep(0.05)
        except Exception as e:
            logger.error(f"WebSocket connection error: {e}. Reconnecting in 3 seconds...")
            await asyncio.sleep(3)

def start_websocket(shared_vars, data_lock):
    """
    Starts the asynchronous websocket task.
    """
    asyncio.run(websocket_task(shared_vars, data_lock))
