from django.db import models

# Create your models here.
from django.contrib.auth.models import User
from django.db import models

class ChatHistory(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    prompt = models.TextField()
    response = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.email} - {self.timestamp}"
