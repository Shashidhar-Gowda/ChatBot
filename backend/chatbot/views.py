from django.shortcuts import render

# Create your views here.
from rest_framework_simplejwt.tokens import RefreshToken
from django.contrib.auth import authenticate
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.contrib.auth.models import User
from rest_framework.permissions import IsAuthenticated
from rest_framework.decorators import api_view, permission_classes
from .models import ChatHistory
from .serializers import ChatHistorySerializer

class LoginView(APIView):
    def post(self, request):
        email = request.data.get("username")  # ✅ Check for "username"
        password = request.data.get("password")

        print("Login attempt:", email)

        user = authenticate(username=email, password=password)
        print("Authenticated user:", user)

        if user is not None:
            refresh = RefreshToken.for_user(user)
            return Response({
                "message": "Login success",
                "access": str(refresh.access_token),
                "refresh": str(refresh),
            }, status=status.HTTP_200_OK)
        
        return Response({"error": "Invalid credentials"}, status=status.HTTP_401_UNAUTHORIZED)

class SignupView(APIView):
    def post(self, request):
        email = request.data.get("email")
        password = request.data.get("password")

        print("Signup data:", email, password)

        if not email or not password:
            return Response({"error": "Email and password are required"}, status=status.HTTP_400_BAD_REQUEST)

        if User.objects.filter(username=email).exists():
            return Response({"error": "User already exists"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            user = User.objects.create_user(username=email, email=email, password=password)
            user.save()
            return Response({"message": "User created successfully"}, status=status.HTTP_201_CREATED)
        except Exception as e:
            print("Signup error:", e)
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
from rest_framework.permissions import IsAuthenticated
from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from .models import ChatHistory

@api_view(["POST"])
@permission_classes([IsAuthenticated])
def save_chat_history(request):
    user = request.user
    prompt = request.data.get("prompt")
    response = request.data.get("response")

    if prompt and response:
        ChatHistory.objects.create(user=user, prompt=prompt, response=response)
        return Response({"message": "Chat saved successfully"})
    return Response({"error": "Missing data"}, status=400)

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_chats(request):
    user = request.user
    chats = ChatHistory.objects.filter(user=user).order_by('-timestamp')  # newest first
    serializer = ChatHistorySerializer(chats, many=True)
    return Response(serializer.data)