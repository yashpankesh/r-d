from django.db import models
from django.contrib.auth.models import User

class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    phone_number = models.CharField(max_length=15, blank=True, null=True)

    def __str__(self):
        return self.user.username

class ObjectAlert(models.Model):
    object_name = models.CharField(max_length=100)
    confidence_score = models.FloatField()
    timestamp = models.DateTimeField(auto_now_add=True)
    user_email = models.EmailField()

    def __str__(self):
        return f"Alert: {self.object_name} detected with confidence {self.confidence_score}"

class ContactMessage(models.Model):
    name = models.CharField(max_length=100)
    email = models.EmailField()
    message = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.name} - {self.email}"