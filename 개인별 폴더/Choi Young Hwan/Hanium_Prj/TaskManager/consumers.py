import base64
from channels.generic.websocket import AsyncWebsocketConsumer


class ChatConsumer(AsyncWebsocketConsumer):
    # connect to Websocket
    async def connect(self):
        await self.accept()

    async def receive(self, text_data):
        data = text_data
        print(type(data))
        await self.send("echo : " + data)

        data = data[22:]
        temp = base64.urlsafe_b64decode(data)
        print(type(temp))
