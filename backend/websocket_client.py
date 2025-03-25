import asyncio
import websockets
import json
import logging

def create_payload(shared_vars):
    """Creates optimized payload with only necessary tracking data."""
    return json.dumps({
        "eye_blink": shared_vars.eye_blink,
        "mouth_open": shared_vars.mouth_open,
        "lip_sync": shared_vars.lip_sync,
        "head_pose": shared_vars.head_pose
    })

async def send_tracking_data(uri, shared_vars, data_lock):
    """Handles WebSocket communication with optimized data transfer."""
    while True:
        try:
            async with websockets.connect(uri, ping_interval=5) as websocket:
                while True:
                    with data_lock:
                        payload = create_payload(shared_vars)
                    await websocket.send(payload)
                    await asyncio.sleep(0.02)  # Reduced delay for better real-time performance
        except websockets.exceptions.ConnectionClosedError:
            logging.warning("WebSocket connection lost. Reconnecting...")
            await asyncio.sleep(1)  # Attempt reconnect after delay
        except Exception as e:
            logging.error(f"Unexpected WebSocket error: {e}")
            await asyncio.sleep(1)

def start_websocket(shared_vars, data_lock):
    """Starts the WebSocket communication asynchronously."""
    uri = "ws://localhost:8001"
    asyncio.run(send_tracking_data(uri, shared_vars, data_lock))
    logging.info("WebSocket connection closed.")
    