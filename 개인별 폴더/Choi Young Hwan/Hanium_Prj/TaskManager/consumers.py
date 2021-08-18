import base64
from channels.generic.websocket import AsyncWebsocketConsumer


class ChatConsumer(AsyncWebsocketConsumer):
    # connect to Websocket
    async def connect(self):
        await self.accept()

    async def receive(self, text_data):
        self.data = ''
        self.data = text_data
        print(type(self.data))
        await self.send("echo :", self.data)

        # data slicing(base64 문자열 앞에 붙는 필요 없는 문자열제거)
        self.data = self.data[22:]
        # base64 문자열 디코딩(str -> bytes)
        self.img = b''
        self.img = base64.urlsafe_b64decode(self.data)
        print(type(self.img))

    async def send_frame(self):
        return self.img