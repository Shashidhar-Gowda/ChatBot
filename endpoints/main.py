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
from langchain_community.chat_models import ChatOpenAI
from django.contrib.auth import get_user_model
from auth import verify_token
from llm_chain import get_bot_response, reset_memory, llm
import os
from uuid import uuid4
from typing import Optional


import getpass
if not os.environ.get("GROQ_API_KEY"):
    os.environ["GROQ_API_KEY"] = getpass.getpass("Enter API key for Groq: ")




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

import getpass
if not os.environ.get("GROQ_API_KEY"):
    os.environ["GROQ_API_KEY"] = getpass.getpass("Enter API key for Groq: ")

GROQ_API_KEY = os.environ["GROQ_API_KEY"]

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

class ChatRequest(BaseModel):
    prompt: str

@router.post("/api/chat")
async def chat_with_groq(request: ChatRequest):
    try:
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }

        data = {
            "model": "deepseek-r1-distill-llama-70b",
            "messages": [
                {"role": "user", "content": request.prompt}
            ]
        }

        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=data
        )

        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response.text)

        reply = response.json()["choices"][0]["message"]["content"]
        return {"reply": reply}

    except Exception as e:
        print(f"Groq error: {str(e)}")
        raise HTTPException(status_code=500, detail="LLM request failed")

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

@router.post("/api/reset_chat")
async def reset_chat(token: str = Depends(verify_token)):
    reset_memory()
    return {"message": "Chat memory reset"}


UPLOAD_DIR = "uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/api/upload_dataset")
async def upload_dataset(file: UploadFile = File(...)):
    try:
        file_id = str(uuid4())
        file_path = os.path.join(UPLOAD_DIR, f"{file_id}_{file.filename}")
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Triggering agent pipeline here
        result = run_rule_based_agent(input_type="csv", input_value=file_path)

        return JSONResponse(content={"message": "File uploaded", "path": file_path})
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

class PromptRequest(BaseModel):
    prompt: str

class IntentResponse(BaseModel):
    handled: bool
    reply: Optional[str] = None

@router.post("/api/predict_intent", response_model=IntentResponse)
async def predict_intent(data: PromptRequest):
    try:
        # 🔍 Try to get a tool-based or intent-based reply first
        response = get_intent_response(data.prompt)

        if response:
            return {"handled": True, "reply": response}
        else:
            return {"handled": False}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

app.include_router(router)

