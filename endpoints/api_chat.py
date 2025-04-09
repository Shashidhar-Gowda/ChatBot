from fastapi import APIRouter, Request, HTTPException, Depends
from pydantic import BaseModel
from endpoints.chatbot import predict_intent
from .router import route_to_tool
from fastapi.responses import JSONResponse

router = APIRouter()

class ChatRequest(BaseModel):
    prompt: str

@router.post("/chat")
async def chat_endpoint(request: Request, chat_request: ChatRequest):
    try:
        prompt = chat_request.prompt

        # Step 1: Predict intent
        intent = predict_intent(prompt)

        # Step 2: Route to appropriate tool
        response = await route_to_tool(intent, prompt)

        return JSONResponse(content={"reply": response})

    except Exception as e:
        print("Error in /chat:", e)
        raise HTTPException(status_code=500, detail="Internal server error")