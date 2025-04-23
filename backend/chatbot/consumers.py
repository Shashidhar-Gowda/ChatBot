# import json
# from channels.generic.websocket import AsyncWebsocketConsumer
# from channels.db import database_sync_to_async
# from django.contrib.auth.models import AnonymousUser
# from rest_framework_simplejwt.tokens import AccessToken
# from django.core.exceptions import ValidationError

# class ChatConsumer(AsyncWebsocketConsumer):
#     async def connect(self):
#         self.user = await self.get_user()
#         if isinstance(self.user, AnonymousUser):
#             await self.close()
#         else:
#             await self.accept()

#     @database_sync_to_async
#     def get_user(self):
#         try:
#             token = self.scope['query_string'].decode().split('=')[1]
#             access_token = AccessToken(token)
#             return access_token.payload.get('user_id')
#         except (IndexError, KeyError, ValidationError):
#             return AnonymousUser()

#     async def disconnect(self, close_code):
#         pass

#     async def receive(self, text_data):
#         try:
#             data = json.loads(text_data)
#             if data.get('type') == 'ping':
#                 await self.send(text_data=json.dumps({'type': 'pong'}))
#             else:
#                 await self.handle_message(data)
#         except json.JSONDecodeError:
#             await self.send(text_data=json.dumps({
#                 'error': 'Invalid JSON format'
#             }))

#     async def handle_message(self, data):
#         await self.send(text_data=json.dumps({
#             'type': 'message',
#             'content': f"Received: {data.get('message')}",
#             'user': self.user.id
#         }))
