# Minimal OpenAI-compatible echo server for OpenWebUI
# Run: uvicorn server:app --host 127.0.0.1 --port 8000

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional, Literal
from uuid import uuid4
import time

app = FastAPI(title="Garage Echo - OpenWebUI")

# ---- 1) Replace this function with your own model call ----------------------
# def generate_reply(user_text: str) -> str:
#     """
#     Plug your model here.
#     Example:
#         return my_model.generate(user_text)
#     For now we echo back the user's last message.
#     """
#     return user_text
from serve_model import generate
def generate_reply(user_text: str) -> str:
    return generate(user_text)
# -----------------------------------------------------------------------------

MODEL_ID = "garage-echo-1"

# Data models that match OpenAI's Chat Completions API (only what we need)
class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: List[Message]
    stream: Optional[bool] = False  # OpenWebUI works fine with stream=False
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
    # Pick the last user message - this is the usual prompt for your model
    user_text = next((m.content for m in reversed(req.messages) if m.role == "user"), "")
    reply = generate_reply(user_text)

    now = int(time.time())
    choice = {
        "index": 0,
        "message": {"role": "assistant", "content": reply},
        "finish_reason": "stop",
    }
    # usage is optional - keep simple
    usage = {"prompt_tokens": 0, "completion_tokens": len(reply.split()), "total_tokens": len(reply.split())}

    return {
        "id": f"chatcmpl-{uuid4().hex}",
        "object": "chat.completion",
        "created": now,
        "model": req.model,
        "choices": [choice],
        "usage": usage,
    }
