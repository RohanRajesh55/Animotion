import asyncio
import websockets

async def echo(websocket, path):
    async for message in websocket:
        print(f"Received: {message}")  # Print received data to verify it's being sent
        await websocket.send("Acknowledged")  # Send a response to confirm reception

start_server = websockets.serve(echo, "localhost", 8001)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
