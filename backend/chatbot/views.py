from django.shortcuts import render
import json

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
from endpoints.llm_chain import get_bot_response  # Import the AI response function directly

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def get_ai_response_view(request):
    try:
        # Get user ID from JWT token
        user_id = request.user.username  # Using username as user ID
        
        # Handle different content types
        if request.content_type == 'application/json':
            try:
                request_data = json.loads(request.body)
            except json.JSONDecodeError:
                return Response({'error': 'Invalid JSON'}, status=400)
        else:
            request_data = request.data
            
        # Get prompt from request data
        prompt = request_data.get('prompt')
        if not prompt:
            return Response({'error': 'Prompt is required'}, status=400)
            
        # Get AI response with user context
        ai_response = get_bot_response(user_id, prompt)
        
        # Save to chat history
        ChatHistory.objects.create(
            user=request.user,
            prompt=prompt,
            response=ai_response
        )
        
        return Response({
            'response': ai_response,
            'status': 'success'
        })
    except Exception as e:
        import traceback
        print(f"Error in get_ai_response_view: {str(e)}\n{traceback.format_exc()}")
        error_details = str(e)
        if isinstance(e, json.JSONDecodeError):
            error_details = "Invalid JSON data"
        return Response(
            {
                'error': 'Internal server error', 
                'details': error_details,
                'content_type': request.content_type
            },
            status=500
        )

class LoginView(APIView):
    def post(self, request):
        email = request.data.get("email")  # Changed from username to email
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

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def upload_file(request):
    try:
        if 'file' not in request.FILES:
            return Response({'error': 'No file provided'}, status=400)
            
        uploaded_file = request.FILES['file']
        prompt = request.POST.get('prompt', '')
        
        # Validate prompt exists with file upload
        if not prompt.strip():
            return Response(
                {'error': 'Analysis prompt is required with file upload'}, 
                status=400
            )
        
        # Validate file type
        if not uploaded_file.name.endswith(('.csv', '.xlsx', '.xls', '.json', '.docx')):
            return Response({'error': 'Only CSV, Excel, JSON, and Word files are allowed'}, status=400)
            
        # Validate file size (10MB max)
        if uploaded_file.size > 10 * 1024 * 1024:
            return Response({'error': 'File size exceeds 10MB limit'}, status=400)
            
        # Save file to uploads directory
        import os
        from django.conf import settings
        
        upload_dir = os.path.join(settings.BASE_DIR, 'endpoints', 'uploaded_files')
        os.makedirs(upload_dir, exist_ok=True)
        
        file_path = os.path.join(upload_dir, uploaded_file.name)
        with open(file_path, 'wb+') as destination:
            for chunk in uploaded_file.chunks():
                destination.write(chunk)
                
        # Process the file using the enhanced analyzer
        try:
            from endpoints.llm_chain import analyze_file_data
            analysis_result = analyze_file_data(file_path, prompt)
            
            if analysis_result['status'] == 'error':
                response = analysis_result['fallback']
            else:
                response = analysis_result['result']
                
            # Log successful analysis
            ChatHistory.objects.create(
                user=request.user,
                prompt=prompt,
                response=response,
                file_used=uploaded_file.name
            )
            return Response({
                'status': 'success',
                'message': 'File uploaded successfully',
                'response': response
            })
        except Exception as e:
            return Response({'error': f'Analysis failed: {str(e)}'}, status=500)
            
    except Exception as e:
        return Response({'error': str(e)}, status=500)
