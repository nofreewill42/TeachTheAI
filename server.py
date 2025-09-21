# Minimal OpenAI-compatible server for OpenWebUI
# Run: uvicorn server:app --host 127.0.0.1 --port 8000

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional, Literal
from uuid import uuid4
import time

app = FastAPI(title="Garage Echo - OpenWebUI")

from serve_model import generate

def generate_reply(messages, temperature=None, max_tokens=None) -> str:
    # Convert Pydantic models to plain dicts and pass full convo
    convo = [{"role": m.role, "content": m.content} for m in messages if m.role in ("user", "assistant")]
    return generate(convo, temperature=temperature, max_tokens=max_tokens)

MODEL_ID = "garage-echo-1"

class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: List[Message]
    stream: Optional[bool] = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None

@app.get("/")
def root():
    return {"ok": True, "model": MODEL_ID}

@app.get("/v1/models")
def list_models():
    now = int(time.time())
    return {
        "object": "list",
        "data": [{"id": MODEL_ID, "object": "model", "created": now, "owned_by": "you"}],
    }

@app.post("/v1/chat/completions")
def chat(req: ChatRequest):
    reply = generate_reply(req.messages, temperature=req.temperature, max_tokens=req.max_tokens)

    now = int(time.time())
    choice = {
        "index": 0,
        "message": {"role": "assistant", "content": reply},
        "finish_reason": "stop",
    }
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    return {
        "id": f"chatcmpl-{uuid4().hex}",
        "object": "chat.completion",
        "created": now,
        "model": req.model,
        "choices": [choice],
        "usage": usage,
    }
