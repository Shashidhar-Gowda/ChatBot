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
    from .llm_chain import get_bot_response, reset_memory, llm
import os
from uuid import uuid4
from typing import Optional

app = FastAPI()
router = APIRouter()
security = HTTPBearer()

origins = [
    "http://localhost:5174",
    "http://127.0.0.1:5174",
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
                "http://127.0.0.1:8000/api/login/",
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
                "http://127.0.0.1:8000/api/signup/",
                json={"email": body["email"], "password": body["password"]}
            )

        if response.status_code == 201:
            return {"message": "Signup successful"}
        else:
            raise HTTPException(status_code=response.status_code, detail=response.json())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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

@router.post("/api/ask-file")
async def ask_file(prompt: str = Form(...), file: UploadFile = Form(...), token=Depends(security)):
    with NamedTemporaryFile(delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    if file.filename.endswith(".csv"):
        loader = CSVLoader(file_path=tmp_path)
    elif file.filename.endswith(".pdf"):
        loader = PyPDFLoader(file_path=tmp_path)
    else:
        return JSONResponse({"error": "Unsupported file format"}, status_code=400)

    docs = loader.load()
    embeddings = HuggingFaceEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    docs_similar = vectorstore.similarity_search(prompt)
    chain = load_qa_chain(llm, chain_type="stuff")
    response = chain.run(input_documents=docs_similar, question=prompt)

    return {"answer": response}

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

