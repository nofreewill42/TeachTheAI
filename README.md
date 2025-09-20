# Garage Echo — OpenWebUI starter

Tiny OpenAI-compatible FastAPI server that plugs into OpenWebUI. It echoes the user's last message. Swap one function to call your own model.

## Files

**server.py**
```python
# Minimal OpenAI-compatible FastAPI server for OpenWebUI.
# Run: uvicorn server:app --host 127.0.0.1 --port 8000

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Literal, Optional
from uuid import uuid4
import time

app = FastAPI()
MODEL_ID = "garage-echo-1"

# ---- CHANGE THIS to call your own model -------------------------------------
def generate_reply(user_text: str) -> str:
    # Example: return my_model.generate(user_text)
    return user_text
# -----------------------------------------------------------------------------

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
def health():
    return {"ok": True, "model": MODEL_ID}

@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{
            "id": MODEL_ID,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "you"
        }]
    }

@app.post("/v1/chat/completions")
def chat(req: ChatRequest):
    user_text = next((m.content for m in reversed(req.messages) if m.role == "user"), "")
    reply = generate_reply(user_text)
    return {
        "id": f"chatcmpl-{uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": reply},
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": len(reply.split()),
            "total_tokens": len(reply.split())
        },
    }
```

**requirements.txt**
```txt
fastapi
uvicorn
pydantic
```

## TL;DR

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn server:app --host 127.0.0.1 --port 8000
```

Point OpenWebUI to `http://127.0.0.1:8000/v1` with **Autorización: Ninguno**, select `garage-echo-1`, and chat.

## Why this works

OpenWebUI understands the OpenAI Chat Completions format. Expose:

- `GET /v1/models`
- `POST /v1/chat/completions`

…and return the expected JSON.

## Integrate your model

Edit **only** this in `server.py`:

```python
def generate_reply(user_text: str) -> str:
    # return my_model.generate(user_text)
    return user_text
```

(Optional) load weights once:

```python
@app.on_event("startup")
def load_model():
    global my_model
    my_model = load_your_weights()
```

## Quick tests

Health:

```bash
curl http://127.0.0.1:8000/
# -> {"ok":true,"model":"garage-echo-1"}
```

List models:

```bash
curl http://127.0.0.1:8000/v1/models
```

Echo a chat:

```bash
curl -H "Content-Type: application/json" \
  -d '{"model":"garage-echo-1","messages":[{"role":"user","content":"hi"}]}' \
  http://127.0.0.1:8000/v1/chat/completions
```

## Wire it to OpenWebUI (Spanish UI labels)

1. **Ajustes de Admin → Conexiones → Añadir conexión**  
2. **Tipo de Conexión**: Externo  
3. **Tipo de Proveedor**: OpenAI  
4. **URL Base API**: `http://127.0.0.1:8000/v1`  
5. **Autorización**: **Ninguno**  
6. (Opcional) **IDs Modelo**: `garage-echo-1`  
7. **Guardar**, luego selecciona `garage-echo-1` en el chat

## Notes

- Streaming is **not** required; this server returns a single non-streamed completion.
- OpenWebUI reads `choices[0].message.content`.
- You can add token usage if you want; it's optional here.
