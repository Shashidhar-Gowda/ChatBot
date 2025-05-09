from django.shortcuts import render
import json
import os
from uuid import uuid4
from django.http import JsonResponse, HttpResponseBadRequest
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from django.conf import settings
from rest_framework_simplejwt.tokens import RefreshToken
from django.contrib.auth import authenticate
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.contrib.auth.models import User
from rest_framework.permissions import IsAuthenticated
from rest_framework.decorators import api_view, permission_classes
from .models import ChatHistory
from .serializers import UploadedFileSerializer
from .llm.main import get_bot_response 
from .mongo_chat_history import save_chat_history as mongo_save_chat_history, get_chat_history_by_session, get_grouped_chat_history
from django.contrib.auth import get_user_model
from django.shortcuts import render, redirect
from .models import UploadedFile
from django.contrib.auth.decorators import login_required
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import json
from bson import ObjectId
import traceback


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def get_ai_response_view(request):
    try:
        user_id = request.user.username  # Use user ID from JWT

        if request.content_type == 'application/json':
            try:
                request_data = json.loads(request.body)
            except json.JSONDecodeError:
                return Response({'error': 'Invalid JSON'}, status=400)
        else:
            request_data = request.data

        prompt = request_data.get('prompt')
        file_id = request_data.get('file_id')  # Get file_id from request
        if not prompt:
            return Response({'error': 'Prompt is required'}, status=400)

        # Use the new get_bot_response from main.py, passing file_id
        ai_response = get_bot_response(prompt)

        # Save chat history in MongoDB
        mongo_save_chat_history(user_id, prompt, ai_response, session_id=None)

        return Response({
            'response': ai_response,
            'status': 'success'
        })
    except Exception as e:
        print(f"Error: {str(e)}\n{traceback.format_exc()}")
        return Response({'error': 'Internal server error'}, status=500)
    
class LoginView(APIView):
    def post(self, request):
        print("RAW request body:", request.body)
        print("Parsed request data:", request.data)

        email = request.data.get("email")  # Changed to email
        password = request.data.get("password")

        print("Login attempt:", email)

        # Query the user by email
        User = get_user_model()  # Use custom user model if applicable
        try:
            user = User.objects.get(email=email)  # Look up by email
        except User.DoesNotExist:
            user = None

        # Check the password
        if user and user.check_password(password):  # Check if the password is correct
            # If password matches, generate JWT token
            refresh = RefreshToken.for_user(user)
            return Response({
                "message": "Login success",
                "access": str(refresh.access_token),
                "refresh": str(refresh),
            }, status=status.HTTP_200_OK)

        print("Authenticated user:", user)

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

import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

@csrf_exempt
def save_chat_history(request):
    if request.method == 'POST':
        try:
            # Extract 'prompt' and 'response' from request body
            data = json.loads(request.body)
            prompt = data.get('prompt')
            response = data.get('response')

            if not prompt or not response:
                return JsonResponse({'error': 'Prompt and response are required'}, status=400)

            # Save chat history logic here (e.g., saving to database or MongoDB)
            # Your logic to save chat history should go here
            return JsonResponse({'status': 'success'}, status=200)

        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON data'}, status=400)
    else:
        return JsonResponse({'error': 'Invalid HTTP method'}, status=405)

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_chats(request):
    user = request.user
    session_id = request.query_params.get('session_id')
    if not session_id:
        return Response({"error": "session_id query parameter is required"}, status=400)
    # Fetch chat history from MongoDB by session
    mongo_chats = get_chat_history_by_session(user.username, session_id)
    print(f"MongoDB chats raw data for user {user.username} session {session_id}: {mongo_chats}")
    # Format MongoDB chat history for frontend
    formatted_chats = []
    for chat in mongo_chats:
        formatted_chats.append({
            "prompt": chat.get("prompt", ""),
            "response": chat.get("response", ""),
            "timestamp": chat.get("timestamp").isoformat() if chat.get("timestamp") else None
        })
    return Response(formatted_chats)

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def verify_token(request):
    return Response({"status": "success", "message": "Token is valid."})

from django.shortcuts import render
import json
from django.http import JsonResponse, HttpResponseBadRequest, HttpResponseServerError
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework import status
from django.db import connection
import pandas as pd
import io
import traceback
from .llm.main import get_bot_response
from sqlalchemy import text


import os
import pandas as pd
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
from django.conf import settings
import json
from .models import UploadedFile # Assuming you have a model for uploaded data
from .llm.main import get_bot_response  # We'll define this utility function

@csrf_exempt
def get_ai_response(request):
    if request.method == "POST":
        data = json.loads(request.body)
        prompt = data.get('prompt')

        # Call your LLM with only prompt
        response = get_bot_response(prompt)
        print("Prompt:", prompt)
        print("Response:", response)

        bot_reply = response["messages"][-1].content if response.get("messages") else "No response."

        return JsonResponse({"response": bot_reply}, status=200)
    else:
        return JsonResponse({"error": "Only POST method allowed"}, status=405)


@csrf_exempt
def upload_and_process(request):
    if request.method == "POST":
        prompt = request.POST.get('prompt')
        file = request.FILES.get('file')

        if not file:
            return JsonResponse({"error": "No file uploaded"}, status=400)

        # Save file locally temporarily
        file_path = os.path.join(settings.MEDIA_ROOT, file.name)
        with open(file_path, 'wb+') as destination:
            for chunk in file.chunks():
                destination.write(chunk)

        # Process file (example: if CSV)
        if file.name.endswith('.csv'):
            df = pd.read_csv(file_path)
            # Save data into PostgreSQL
            for _, row in df.iterrows():
                UploadedFile.objects.create(**row.to_dict())

        # Now call your LLM and tell it: "use uploaded data"
        # (You can pass prompt + indicate to LLM that DB has fresh data)

        response = get_bot_response(prompt, use_db=True)

        return JsonResponse({"response": response}, status=200)
    else:
        return JsonResponse({"error": "Only POST method allowed"}, status=405)

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from .models import UploadedFile
import os
from django.conf import settings
from django.utils.text import get_valid_filename
from django.utils import timezone

class FileUploadView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request, *args, **kwargs):
        uploaded_file = request.FILES.get('file')
        if not uploaded_file:
            return Response({'error': 'No file uploaded.'}, status=400)

        filename = get_valid_filename(uploaded_file.name)
        file_path = os.path.join(settings.MEDIA_ROOT, 'user_uploads', filename)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Save file to disk
        with open(file_path, 'wb+') as destination:
            for chunk in uploaded_file.chunks():
                destination.write(chunk)

        # Save file to database
        uploaded_instance = UploadedFile.objects.create(
            user=request.user,
            file='user_uploads/' + filename,
            filename=uploaded_file.name,
            uploaded_at=timezone.now()
        )

        # 🛠 Save uploaded file metadata in session
        uploaded_files = request.session.get('uploaded_files', {})
        uploaded_files[uploaded_instance.filename] = uploaded_instance.id
        request.session['uploaded_files'] = uploaded_files
        request.session.modified = True

        print(request.session.get('uploaded_files'))

        return Response({
            'message': 'File uploaded successfully!',
            'file_id': uploaded_instance.id
        }, status=201)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_grouped_chat_history(request):
    try:
        # Extract user_id correctly from the authenticated user
        user_id = request.user.username # Use 'id' if it's the primary key for the user
        print(f"Fetching chat history for user_id: {user_id}")
        grouped = get_grouped_chat_history(user_id)
        print("Grouped chat:", grouped)
        return Response(grouped)
    except Exception as e:
        print("Chat history error:", str(e))
        return Response({"error": "Unable to fetch chat history"}, status=500)
