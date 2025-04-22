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
from endpoints.mongo_chat_history import save_chat_history as mongo_save_chat_history, get_chat_history_by_session

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
        
        # Save to chat history in Django ORM
        ChatHistory.objects.create(
            user=request.user,
            prompt=prompt,
            response=ai_response
        )
        
        # Save to chat history in MongoDB
        mongo_save_chat_history(user_id, prompt, ai_response)
        
        print("AI Response Structure:", {
            'raw_response': ai_response,
            'type': type(ai_response),
            'is_dict': isinstance(ai_response, dict)
        })
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

@api_view(["POST"])
@permission_classes([IsAuthenticated])
def save_chat_history(request):
    user = request.user
    prompt = request.data.get("prompt")
    response = request.data.get("response")
    session_id = request.data.get("session_id")

    if prompt and response and session_id:
        ChatHistory.objects.create(user=user, prompt=prompt, response=response)
        # Also save in MongoDB with session_id
        mongo_save_chat_history(user.username, prompt, response, session_id)
        return Response({"message": "Chat saved successfully"})
    return Response({"error": "Missing data or session_id"}, status=400)

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

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def upload_file(request):
    try:
        if 'file' not in request.FILES:
            return Response({'error': 'No file provided'}, status=400)
        
        uploaded_file = request.FILES['file']
        prompt = request.POST.get('prompt', '')
        print(f"Received upload with prompt: {prompt}")  # Log prompt for debugging
        
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
        
        # Process file in memory without saving to disk
        file_content = uploaded_file.read()
        
        # Process the file content using the enhanced analyzer
        try:
            from endpoints.llm_chain import analyze_file_data_in_memory
            analysis_result = analyze_file_data_in_memory(file_content, uploaded_file.name, prompt)
            
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
                'message': 'File uploaded and processed successfully',
                'response': response
            })
        except Exception as e:
            print(f"Analysis failed: {str(e)}")  # Log error for debugging
            return Response({'error': f'Analysis failed: {str(e)}'}, status=500)
    except Exception as e:
        print(f"Upload file error: {str(e)}")  # Log error for debugging
        return Response({'error': str(e)}, status=500)
