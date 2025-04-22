from fastapi import FastAPI, HTTPException, Depends, APIRouter, UploadFile, Form, Request, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer
import httpx
from pydantic import BaseModel
import requests
import shutil
from tempfile import NamedTemporaryFile
from datetime import datetime, timedelta
from langchain.document_loaders import CSVLoader, PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from django.contrib.auth import get_user_model
try:
    from endpoints.auth import verify_token
except ImportError:
    from .auth import verify_token
try:
    from endpoints.llm_chain import get_bot_response, reset_memory, llm
except ImportError:
    from .llm_chain import get_bot_response, llm
import os
from uuid import uuid4
from typing import Optional
from endpoints.llm_chain import agent 

app = FastAPI()
router = APIRouter()
security = HTTPBearer()

origins = [
    "http://localhost:5174",
    "http://127.0.0.1:5174",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Include the router to register its routes with the app
app.include_router(router, prefix="/api")

class ChatRequest(BaseModel):
    prompt: str

class PromptRequest(BaseModel):
    prompt: str

class IntentResponse(BaseModel):
    handled: bool
    reply: Optional[str] = None

@app.post("/api/login")
async def login(request: Request):
    body = await request.json()
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://backend:8000/api/login/",
                json={"username": body["email"], "password": body["password"]}
            )

        if response.status_code == 200:
            return {"message": "Login successful", **response.json()}
        else:
            raise HTTPException(status_code=401, detail="Invalid credentials")

    except httpx.RequestError as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

@app.post("/api/signup")
async def signup(request: Request):
    body = await request.json()
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://backend:8000/api/signup/",
                json={"email": body["email"], "password": body["password"]}
            )

        if response.status_code == 201:
            return {"message": "Signup successful"}
        else:
            raise HTTPException(status_code=response.status_code, detail=response.json())

    except httpx.RequestError as e:
        # Log the error details
        print(f"HTTPX request error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Request failed: {str(e)}")
    except Exception as e:
        # Log the full error trace for better debugging
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Signup failed: {str(e)}")


@router.post("/api/chat")
async def chat_with_llm(
    request: Request, 
    chat_request: ChatRequest,
    token: str = Depends(verify_token)
):
    try:
        # Get user ID from verified token
        user_id = token.get("sub")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token")
            
        response = get_bot_response(str(user_id), chat_request.prompt)
        return {"reply": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {str(e)}")

from endpoints.tools import AnalysisTools
import pandas as pd
from io import StringIO, BytesIO

@router.post("/api/ask-file")
async def ask_file(
    prompt: str = Form(...),
    file: UploadFile = File(...),
    token: str = Depends(verify_token)
):
    try:
        # Validate file and save it (as in your original code)
        MAX_SIZE = 50 * 1024 * 1024
        file.file.seek(0, 2)
        file_size = file.file.tell()
        file.file.seek(0)
        valid_extensions = ['.csv', '.json', '.xlsx', '.xls']
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_size > MAX_SIZE or file_ext not in valid_extensions:
            raise HTTPException(status_code=400, detail="Invalid file")

        file_id = str(uuid4())
        file_path = os.path.join(UPLOAD_DIR, f"{file_id}_{file.filename}")
        with open(file_path, "wb") as buffer:
            while chunk := await file.read(1024 * 1024):
                buffer.write(chunk)

        # Construct the input for the LangChain agent, including the file path
        tool_input = f"file_path={file_path}; {prompt}"  # Adjust format as needed

        # Invoke the LangChain agent with the constructed input
        response = await agent.ainvoke({"input": tool_input})
        answer = response.get("output", "No answer found.")

        return {"answer": answer, "file_path": file_path}

    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse({"error": f"File processing failed: {str(e)}"}, status_code=500)

@app.post("/api/reset_chat")
async def reset_chat(token: str = Depends(verify_token)):
    reset_memory()
    return {"message": "Chat memory reset"}

@app.post("/api/upload_dataset")
async def upload_dataset(file: UploadFile = File(...)):
    try:
        # Validate file size (max 50MB)
        MAX_SIZE = 50 * 1024 * 1024  # 50MB
        file.file.seek(0, 2)  # Seek to end
        file_size = file.file.tell()
        file.file.seek(0)  # Reset pointer
        
        if file_size > MAX_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Max size is {MAX_SIZE/1024/1024}MB"
            )

        # Validate file extension
        valid_extensions = ['.csv', '.json', '.xlsx', '.xls']
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in valid_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Supported types: {', '.join(valid_extensions)}"
            )

        # Create unique filename
        file_id = str(uuid4())
        file_path = os.path.join(UPLOAD_DIR, f"{file_id}_{file.filename}")

        # Save file in chunks to handle large files
        with open(file_path, "wb") as buffer:
            while True:
                chunk = file.file.read(1024 * 1024)  # 1MB chunks
                if not chunk:
                    break
                buffer.write(chunk)

        return JSONResponse(content={
            "message": "File uploaded successfully",
            "path": file_path,
            "size": file_size,
            "extension": file_ext
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Upload failed: {str(e)}"
        )
