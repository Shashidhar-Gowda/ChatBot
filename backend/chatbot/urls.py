from django.urls import path
from .views import LoginView, SignupView, save_chat_history, get_chats
from rest_framework_simplejwt.views import (
    TokenObtainPairView,
    TokenRefreshView,
)

urlpatterns = [
    path("api/login/", LoginView.as_view(), name="login"),
    path("api/signup/", SignupView.as_view(), name="signup"),
    path("api/save_chat/", save_chat_history),
    path('api/get_chats/', get_chats, name='get_chats'),
    path('api/token/', TokenObtainPairView.as_view(), name='token_obtain_pair'),
    path('api/token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
]
