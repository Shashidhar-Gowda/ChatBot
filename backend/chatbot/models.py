from django.contrib.auth.models import User
from django.db import models
from django.utils import timezone
import uuid

class ChatSession(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    created_at = models.DateTimeField(default=timezone.now)
    name = models.CharField(max_length=255, default="New Chat")

class ChatHistory(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    prompt = models.TextField()
    response = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)
    session = models.ForeignKey(ChatSession, on_delete=models.CASCADE, null=True, blank=True)

    def __str__(self):
        return f"{self.user.email} - {self.timestamp} - {self.session.name if self.session else 'No session'}"

class UploadedFile(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    file = models.FileField(upload_to='user_uploads/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    filename = models.CharField(max_length=255, blank=True, null=True)

    def __str__(self):
        return self.file.name

    class Meta:
        db_table = 'chatbot_uploadedfile'
