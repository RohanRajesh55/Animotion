import asyncio
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
    response = json.loads(await ws.recv())
    print("Token response:", response)

    token = response["data"]["authenticationToken"]
    if response["data"].get("requiresVerification", False):
        print("Please approve the plugin in VTube Studio.")
        await asyncio.sleep(10)
    with open(TOKEN_FILE, "w") as f:
        f.write(token)
    return token

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
    response = json.loads(await ws.recv())
    print("Authentication response:", response)

    if response.get("messageType") != "AuthenticationResponse" or not response["data"]["authenticated"]:
        raise Exception("Authentication failed! Try deleting vts_token.txt and retrying.")
    print("Authenticated successfully!")

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
    print("Current model response:", response)
    if response.get("messageType") == "CurrentModelResponse":
        model_id = response["data"].get("modelID")
        if not model_id:
            print("⚠️ No model is currently loaded in VTube Studio.")
        return model_id
    else:
        raise Exception("Failed to get current model")

async def get_parameter_list(ws, model_id):
    request = {
        "apiName": "VTubeStudioPublicAPI",
        "apiVersion": "1.0",
        "requestID": str(uuid.uuid4()),
        "messageType": "InputParameterListRequest",  # ← Updated message type
        "data": {
            "modelID": model_id
        }
    }
    await ws.send(json.dumps(request))
    response = json.loads(await ws.recv())
    print("Parameter list response:", json.dumps(response, indent=2))  # Pretty-print

    if response.get("messageType") == "InputParameterListResponse":
        parameters = response["data"].get("parameters", [])
        print(f"✅ Found {len(parameters)} parameters:")
        for param in parameters:
            print(f" - {param.get('name', 'Unnamed')} (ID: {param['id']})")
        return parameters
    else:
        print(f"❌ Failed: {response.get('data', {}).get('message', 'Unknown error')}")
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
                    "id": "MouthOpen",  # or check from get_parameter_list
                    "value": value
                }
            ]
        }
    }
    await ws.send(json.dumps(message))


async def main():
    async with websockets.connect(URI) as ws:
        print("Connected!")
        token = None
        if os.path.exists(TOKEN_FILE):
            with open(TOKEN_FILE, "r") as f:
                token = f.read().strip()
            print(f"Loaded token from file: {token}")
        else:
            token = await get_token(ws)

        await authenticate(ws, token)
        model_id = "27f71b1596cd47db84292b6085dc3f91"

        if model_id:
            await get_parameter_list(ws,model_id)
        else:
            print("❌ No model loaded. Please load a model in VTube Studio and try again.")
        if model_id:
            await get_parameter_list(ws, model_id)
            print("📤 Sending mouth open values... Press Ctrl+C to stop.")
            try:
                while True:
                    # TODO: Replace with your actual tracking value (0.0 to 1.0)
                    simulated_mouth_value = 0.6

                    await set_mouth_open(ws, simulated_mouth_value)
                    await asyncio.sleep(1 / 30)  # 30 FPS
            except KeyboardInterrupt:
                print("Stopped by user.")

        else:
            print("❌ No model loaded. Please load a model in VTube Studio and try again.")



if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print("An error occurred:", e)
