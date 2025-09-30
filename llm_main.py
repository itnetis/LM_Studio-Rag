import os
import uuid
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# =========================
# Config
# =========================
LMSTUDIO_API_URL = os.getenv("LMSTUDIO_API_URL", "http://127.0.0.1:1234/v1/chat/completions")
MODEL_NAME = os.getenv("MODEL_NAME", "humanizerai")

# =========================
# Pydantic Models
# =========================
class ChatRequest(BaseModel):
    prompt: str

class ChatResponse(BaseModel):
    id: str
    prompt: str
    response: str

# =========================
# FastAPI App
# =========================
app = FastAPI(title="Basic LM Studio API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Endpoints
# =========================
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(chat: ChatRequest):
    try:
        r = requests.post(
            LMSTUDIO_API_URL,
            json={
                "model": MODEL_NAME,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": chat.prompt}
                ],
                "temperature": 0.7
            },
            timeout=30
        )
        r.raise_for_status()
        data = r.json()
        reply = data["choices"][0]["message"]["content"]

        return ChatResponse(
            id=str(uuid.uuid4()),
            prompt=chat.prompt,
            response=reply
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LM Studio call failed: {e}")

@app.get("/health")
async def health_check():
    return {"status": "online"}

# Run with: uvicorn main:app --host 0.0.0.0 --port 9000
