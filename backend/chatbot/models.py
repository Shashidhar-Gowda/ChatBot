from django.db import models

# Create your models here.
from django.contrib.auth.models import User
from django.db import models

from django.utils import timezone
import uuid

class ChatSession(models.Model):  # or a MongoDB schema
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    created_at = models.DateTimeField(default=timezone.now)
    name = models.CharField(max_length=255, default="New Chat")

class ChatHistory(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    prompt = models.TextField()
    response = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.email} - {self.timestamp}"


# File upload model
class UserFile(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    file = models.FileField(upload_to="user_files/")
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.email} - {self.file.name}"
