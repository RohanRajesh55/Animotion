import asyncio
import json
import logging
import configparser
import websockets
import os
from typing import Any

from utils.shared_variables import SharedVariables
from utils.vtube_mapper import map_metrics_to_vts_params

# Configuration for our plugin.
PLUGIN_NAME: str = "trial"
PLUGIN_DEV: str = "bons"
TOKEN_FILE: str = "vts_token.txt"  # Token cache file

# Setup logging configuration.
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

async def authenticate(websocket: websockets.WebSocketClientProtocol) -> str:
    """
    Retrieve and cache an authentication token from VTube Studio, then use it to authenticate.

    Args:
        websocket: The active WebSocket connection.

    Returns:
        The authentication token as a string.

    Raises:
        Exception: If authentication fails.
    """
    token: str = ""

    # Attempt to load token from file.
    if os.path.exists(TOKEN_FILE):
        try:
            with open(TOKEN_FILE, "r") as f:
                token = f.read().strip()
            if not token:
                logger.warning("Token file is empty; a new token will be requested.")
        except Exception as e:
            logger.error(f"Error reading token file: {e}")

    # Request a new token if unavailable.
    if not token:
        request_payload = {
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": "1.0",
            "requestID": "getToken",
            "messageType": "AuthenticationTokenRequest",
            "data": {
                "pluginName": PLUGIN_NAME,
                "pluginDeveloper": PLUGIN_DEV
            }
        }
        logger.info("Requesting new authentication token from VTube Studio...")
        await websocket.send(json.dumps(request_payload))
        response_str = await websocket.recv()
        try:
            response = json.loads(response_str)
            token = response["data"]["authenticationToken"]
            requires_verification = response["data"].get("requiresVerification", False)
        except Exception as e:
            raise Exception(f"Failed to parse token response: {e}") from e

        if requires_verification:
            logger.info("Token requested; waiting for user approval in VTube Studio...")
            await asyncio.sleep(5)

        try:
            with open(TOKEN_FILE, "w") as f:
                f.write(token)
            logger.info("New authentication token saved.")
        except Exception as e:
            logger.error(f"Error writing token file: {e}")

    # Authenticate using the token.
    auth_request = {
        "apiName": "VTubeStudioPublicAPI",
        "apiVersion": "1.0",
        "requestID": "authWithToken",
        "messageType": "AuthenticationRequest",
        "data": {
            "pluginName": PLUGIN_NAME,
            "pluginDeveloper": PLUGIN_DEV,
            "authenticationToken": token
        }
    }
    logger.info("Sending authentication request with token...")
    await websocket.send(json.dumps(auth_request))
    auth_response_str = await websocket.recv()
    try:
        auth_response = json.loads(auth_response_str)
    except Exception as e:
        raise Exception(f"Failed to parse authentication response: {e}") from e

    if auth_response.get("messageType") != "AuthenticationResponse" or not auth_response["data"].get("authenticated", False):
        raise Exception("Authentication failed. Delete the token file and retry.")
    
    logger.info("Successfully authenticated with VTube Studio API.")
    return token

async def websocket_handler(shared_vars: SharedVariables, uri: str) -> None:
    """
    Establish and maintain a WebSocket connection to VTube Studio, sending updated facial metrics
    as VTube Studio parameters at approximately 30 FPS.

    Args:
        shared_vars: Shared variables object containing computed facial metrics.
        uri: The VTube Studio WebSocket server URI.
    """
    while True:
        try:
            logger.info(f"Attempting connection to WebSocket server at {uri}...")
            async with websockets.connect(uri) as websocket:
                logger.info("Connected to VTube Studio WebSocket server.")
                await authenticate(websocket)
                
                # Main loop: retrieve the latest metrics, map them, and send the payload.
                while True:
                    params = map_metrics_to_vts_params(shared_vars)
                    payload = {
                        "apiName": "VTubeStudioPublicAPI",
                        "apiVersion": "1.0",
                        "requestID": "setParam_001",
                        "messageType": "SetParameterValuesRequest",
                        "data": {
                            "parameterValues": [
                                {"id": key, "value": value} for key, value in params.items()
                            ]
                        }
                    }
                    
                    await websocket.send(json.dumps(payload))
                    logger.info(f"Sent parameter update: {payload}")
                    await asyncio.sleep(1 / 30)  # Maintain ~30 FPS update rate.
                    
        except asyncio.CancelledError:
            logger.info("WebSocket handler task cancelled. Exiting...")
            break
        except Exception as e:
            logger.error(f"WebSocket connection error: {e}")
            await asyncio.sleep(5)  # Wait before attempting to reconnect.

def start_websocket_client(shared_vars: SharedVariables) -> None:
    """
    Reads configuration for the WebSocket URI and starts the WebSocket handler.

    Args:
        shared_vars: Shared variables to be updated with facial metrics.
    """
    config = configparser.ConfigParser()
    config.read("config.ini")
    uri = config.get("WebSocket", "VTS_WS_URL", fallback="ws://localhost:8001")
    asyncio.run(websocket_handler(shared_vars, uri))

if __name__ == '__main__':
    # Instantiate the shared variables container (populated by other modules in production).
    shared_vars = SharedVariables()
    start_websocket_client(shared_vars)