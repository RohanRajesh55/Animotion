import asyncio
import logging
import websockets
import json
import os
import uuid

TOKEN_FILE = "vts_token.txt"
PLUGIN_NAME = "testPlugin"
PLUGIN_DEV = "you"
URI = "ws://localhost:8001"

async def get_token(ws):
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
    await ws.send(json.dumps(request))
    try:
        raw_response = await ws.recv()
        logging.info(f"Raw token response: {raw_response}")
        response = json.loads(raw_response)
        logging.info(f"Parsed token response: {response}")
        token = response["data"].get("authenticationToken")
        if not token:
            logging.error("No authentication token in response")
            raise Exception("No authentication token provided")
        if response["data"].get("requiresVerification", False):
            logging.info("Please approve the plugin in VTube Studio.")
        with open(TOKEN_FILE, "w") as f:
            f.write(token)
        return response
    except Exception as e:
        logging.error(f"Error in get_token: {e}")
        raise

async def authenticate(ws, token):
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
    await ws.send(json.dumps(auth_request))
    try:
        raw_response = await ws.recv()
        logging.info(f"Raw auth response: {raw_response}")
        response = json.loads(raw_response)
        logging.info(f"Parsed auth response: {response}")
        if response.get("messageType") != "AuthenticationResponse" or not response["data"].get("authenticated"):
            error_msg = response.get("data", {}).get("message", "Unknown error")
            logging.error(f"Authentication failed: {error_msg}")
            raise Exception(f"Authentication failed: {error_msg}")
        logging.info("Authenticated successfully!")
        return response
    except Exception as e:
        logging.error(f"Error in authenticate: {e}")
        raise

async def get_current_model(ws):
    request = {
        "apiName": "VTubeStudioPublicAPI",
        "apiVersion": "1.0",
        "requestID": str(uuid.uuid4()),
        "messageType": "CurrentModelRequest",
        "data": {}
    }
    await ws.send(json.dumps(request))
    response = json.loads(await ws.recv())
    logging.info(f"Current model response: {response}")
    if response.get("messageType") == "CurrentModelResponse":
        model_id = response["data"].get("modelID")
        if not model_id:
            logging.warning("No model is currently loaded in VTube Studio.")
        return model_id
    else:
        logging.error(f"Failed to get current model: {response.get('data', {}).get('message', 'Unknown error')}")
        raise Exception("Failed to get current model")

async def get_parameter_list(ws, model_id):
    request = {
        "apiName": "VTubeStudioPublicAPI",
        "apiVersion": "1.0",
        "requestID": str(uuid.uuid4()),
        "messageType": "InputParameterListRequest",
        "data": {
            "modelID": model_id
        }
    }
    await ws.send(json.dumps(request))
    response = json.loads(await ws.recv())
    logging.info(f"Parameter list response: {json.dumps(response, indent=2)}")
    if response.get("messageType") == "InputParameterListResponse":
        parameters = response["data"].get("parameters", [])
        logging.info(f"Found {len(parameters)} parameters: {[p.get('name', 'Unnamed') for p in parameters]}")
        return parameters
    else:
        logging.error(f"Failed to get parameter list: {response.get('data', {}).get('message', 'Unknown error')}")
        return []

async def set_mouth_open(ws, value):
    message = {
        "apiName": "VTubeStudioPublicAPI",
        "apiVersion": "1.0",
        "requestID": str(uuid.uuid4()),
        "messageType": "SetParameterValueRequest",
        "data": {
            "parameterValues": [
                {
                    "id": "MouthOpen",
                    "value": value
                }
            ]
        }
    }
    await ws.send(json.dumps(message))

async def send_parameter_updates(ws, param_dict):
    message = {
        "apiName": "VTubeStudioPublicAPI",
        "apiVersion": "1.0",
        "requestID": str(uuid.uuid4()),
        "messageType": "SetParameterValueRequest",
        "data": {
            "parameterValues": [
                {"id": key, "value": float(val)}
                for key, val in param_dict.items()
            ]
        }
    }
    await ws.send(json.dumps(message))

async def main():
    async with websockets.connect(URI) as ws:
        logging.info("Connected!")
        token = None
        if os.path.exists(TOKEN_FILE):
            with open(TOKEN_FILE, "r") as f:
                token = f.read().strip()
            logging.info(f"Loaded token from file: {token}")
        else:
            token_response = await get_token(ws)
            token = token_response["data"].get("authenticationToken")

        auth_response = await authenticate(ws, token)
        model_id = await get_current_model(ws)
        if model_id:
            await get_parameter_list(ws, model_id)
            logging.info("Sending mouth open values... Press Ctrl+C to stop.")
            try:
                while True:
                    simulated_mouth_value = 0.6
                    await set_mouth_open(ws, simulated_mouth_value)
                    await asyncio.sleep(1 / 30)
            except KeyboardInterrupt:
                logging.info("Stopped by user.")
        else:
            logging.error("No model loaded. Please load a model in VTube Studio and try again.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    try:
        asyncio.run(main())
    except Exception as e:
        logging.error(f"An error occurred: {e}")