from django.urls import path
from . import views
from .views import (
    LoginView, 
    SignupView, 
    save_chat_history, 
    get_chats, 
    get_ai_response_view,
    verify_token,
    FileUploadView,
    get_grouped_chat_history
)
from rest_framework_simplejwt.views import (
    TokenObtainPairView,
    TokenRefreshView,
)

urlpatterns = [
    path("api/login/", LoginView.as_view(), name="login"),
    path('upload-file/', FileUploadView.as_view(), name='upload-file'),
    path("api/signup/", SignupView.as_view(), name="signup"),
    path('api/save_chat_history/', views.save_chat_history, name='save_chat_history'),
    path('api/get_chats/', get_chats, name='get_chats'),
    path('api/token/', TokenObtainPairView.as_view(), name='token_obtain_pair'),
    path('api/token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
    path('api/get_ai_response/', get_ai_response_view, name='get_ai_response'),
    path('api/verify_token/', verify_token, name='verify_token'),
    path('api/get_grouped_chat_history/', views.get_grouped_chat_history, name='get_grouped_chat_history'),

    # path('api/get_chat_messages/<str:chat_id>/', views.get_chat_messages, name='get_chat_messages'),
    # path('api/save_chat/', views.save_user_chat, name='save_user_chat'),
    # path('api/list_chats/', views.list_user_chats, name='list_user_chats'),
    # path('api/get_chat/<str:session_id>/', views.get_chat_messages, name='get_chat_messages'),
    # path('api/upload_and_process/', views.upload_and_process_file, name='upload_and_process_file'),
]