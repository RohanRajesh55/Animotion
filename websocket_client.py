import asyncio
import websockets
import logging
import configparser
import json

PLUGIN_NAME = "trial"
PLUGIN_DEV = "bons"
TOKEN_FILE = "vts_token.txt"  # Save token for reuse

async def authenticate(websocket):
    # Load or request new token
    token = None
    try:
        with open(TOKEN_FILE, "r") as f:
            token = f.read().strip()
    except FileNotFoundError:
        # Request a new token
        request = {
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": "1.0",
            "requestID": "getToken",
            "messageType": "AuthenticationTokenRequest",
            "data": {
                "pluginName": PLUGIN_NAME,
                "pluginDeveloper": PLUGIN_DEV
            }
        }
        await websocket.send(json.dumps(request))
        response = json.loads(await websocket.recv())
        token = response["data"]["authenticationToken"]
        requires_verification = response["data"]["requiresVerification"]

        if requires_verification:
            print("Please approve the plugin in VTube Studio.")
            await asyncio.sleep(5)  # Give user time to approve

        with open(TOKEN_FILE, "w") as f:
            f.write(token)

    # Authenticate using the token
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

    await websocket.send(json.dumps(auth_request))
    response = json.loads(await websocket.recv())
    if response.get("messageType") != "AuthenticationResponse" or not response["data"]["authenticated"]:
        raise Exception("Authentication failed. Try deleting vts_token.txt and retrying.")

    logging.info("Authenticated with VTube Studio API.")

async def websocket_handler(shared_vars, data_lock, uri):
    while True:
        try:
            async with websockets.connect(uri) as websocket:
                logging.info(f"Connected to WebSocket server at {uri}")

                # Authenticate
                await authenticate(websocket)

                # Main loop: send parameter values
                while True:
                    with data_lock:
                        mouth_open = shared_vars.lip_sync_value  # Replace with any tracked expression

                    # Clamp to [0,1] for VTube Studio
                    mouth_open = max(0.0, min(1.0, float(mouth_open)))

                    # Send parameter update
                    payload = {
                        "apiName": "VTubeStudioPublicAPI",
                        "apiVersion": "1.0",
                        "requestID": "setParam_001",
                        "messageType": "SetParameterValuesRequest",
                        "data": {
                            "parameterValues": [
                                {
                                    "id": "ParamMouthOpenY",
                                    "value": mouth_open
                                }
                            ]
                        }
                    }

                    await websocket.send(json.dumps(payload))
                    await asyncio.sleep(1/30)  # 30 FPS update rate

        except Exception as e:
            logging.error(f"WebSocket connection error: {e}")
            await asyncio.sleep(5)  # Retry after delay

def start_websocket(shared_vars, data_lock):
    config = configparser.ConfigParser()
    config.read('config.ini')
    uri = config.get('WebSocket', 'VTS_WS_URL', fallback="ws://localhost:8001")
    asyncio.run(websocket_handler(shared_vars, data_lock, uri))
