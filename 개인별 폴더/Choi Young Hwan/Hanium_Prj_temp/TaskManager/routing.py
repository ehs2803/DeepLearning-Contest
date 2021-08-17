from django.urls import re_path

from TaskManager import consumers

websocket_urlpatterns = [
    re_path(r'ws/test/(?P<room_name>\w+)/$', consumers.ChatConsumer.as_asgi()),
]