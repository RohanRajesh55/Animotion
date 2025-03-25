import asyncio
import websockets

async def echo(websocket, path):
    async for message in websocket:
        print(f"Received: {message}")
        await websocket.send(f"Echo: {message}")

start_server = websockets.serve(echo, "localhost", 8001)

print("WebSocket Server started on ws://localhost:8001")
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
