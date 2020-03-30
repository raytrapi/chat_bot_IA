import asyncio
import websockets
import bot_ia

async def echo(websocket, path):
    async for message in websocket:
        print(message)
        mensaje=bot_ia.responder(message)
        print("EL mensaje es\n",mensaje)
        await websocket.send(mensaje)

asyncio.get_event_loop().run_until_complete(
    websockets.serve(echo, 'localhost', 8100))
asyncio.get_event_loop().run_forever()
