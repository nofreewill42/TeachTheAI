# TeachTheAI - OpenWebUI Starter

Minimal OpenAI-compatible FastAPI server that plugs into OpenWebUI. There is exactly **one function** you change to use your own local model; everything else is glue so the UI can talk to your code.

## Why OpenWebUI

**Pros**
- Clean chat UI (history, retry, copy) - no UI coding needed
- Uses OpenAI Chat Completions format (works with many tools)  
- Swap models easily - edit one function, keep the same UI
- Fully offline if your model runs locally

**Cons**
- No built-in token-level sampling (e.g., choose top-k per token)
- OpenAI JSON shape is verbose (but you don't touch it here)

## What this repo does

This repo runs a tiny HTTP server (`server.py`) that implements just enough of the OpenAI API for OpenWebUI to call your code.

- `server.py` exposes two endpoints OpenWebUI needs:
  - `GET /v1/models` - tells UI your model IDs (e.g., "garage-echo-1")
  - `POST /v1/chat/completions` - UI sends chat messages; you return a reply

- You add your model logic in **one function**: `generate_reply(user_text: str)`

- Everything else in `server.py` is plumbing: parse request → call your function → wrap reply in OpenAI JSON (OpenWebUI reads `choices[0].message.content`)

**Flow:**
```
OpenWebUI → GET /v1/models → server.py (returns "garage-echo-1")
OpenWebUI → POST /v1/chat/completions → server.py (extracts user text)  
server.py → calls generate_reply(text) → your code returns string
server.py → returns JSON → OpenWebUI shows it in chat
```

## The one thing you change

Open `server.py` and edit this function:

```python
def generate_reply(user_text: str) -> str:
    # Replace this line with your model call, e.g.:
    # return my_model.generate(user_text)
    return user_text
```

Optional: load weights once at startup (in `server.py`):

```python
@app.on_event("startup")
def load_model():
    # global my_model
    # my_model = load_your_weights()
    pass
```

That's it. Don't touch request parsing or JSON response unless you know why.

## Prerequisites

- Python 3.10+
- OpenWebUI installed (any env where `open-webui` runs)

## Quick start

Run two processes (API on port 8000, UI on port 3000).

**Process A - API** (inside this repo):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn server:app --host 127.0.0.1 --port 8000
```

**Process B - OpenWebUI** (another terminal):

```bash
# activate env where you installed open-webui, e.g.:
source ~/venvs/openwebui/bin/activate
open-webui serve --host 127.0.0.1 --port 3000
```

Open browser at: **http://127.0.0.1:3000**

## Connect OpenWebUI to API

Use these exact Spanish UI labels:

1. **Ajustes de Admin** → **Conexiones** → **Añadir conexión**
2. **Tipo de Conexión**: Externo  
3. **Tipo de Proveedor**: OpenAI
4. **URL Base API**: `http://127.0.0.1:8000/v1`
5. **Autorización**: **Ninguno**
6. (Opcional) **IDs Modelo**: `garage-echo-1`
7. **Guardar**, then in chat select `garage-echo-1` and send a message

## Sanity checks

Health:
```bash
curl http://127.0.0.1:8000/
# -> {"ok":true,"model":"garage-echo-1"}
```

Models:
```bash
curl http://127.0.0.1:8000/v1/models
```

Chat echo:
```bash
curl -H "Content-Type: application/json" \
-d '{"model":"garage-echo-1","messages":[{"role":"user","content":"hi"}]}' \
http://127.0.0.1:8000/v1/chat/completions
```

## Requirements

Exact pins (match your tested versions):

```txt
fastapi==0.116.2
uvicorn==0.36.0  
pydantic==2.11.9
```

Or loose pins:

```txt
fastapi>=0.116.2,<1
uvicorn>=0.36.0,<1
pydantic>=2.11.9,<3
```

## Troubleshooting

- **Port in use (8000/3000)**: change `--port` or stop other process
- **OpenWebUI can't reach API**: use URL Base API `http://127.0.0.1:8000/v1` and **Autorización: Ninguno**
- **Empty reply**: include at least one `{"role":"user","content":"..."}` message  
- **Wrong model ID**: keep "garage-echo-1" or change both code and UI config consistently

## FAQ

**Why GET/POST routes?**  
OpenWebUI uses the OpenAI API. It finds models via `GET /v1/models` and sends chats to `POST /v1/chat/completions`. This repo does just that minimum.

**Add streaming later?**  
Yes - this is non-streamed for simplicity. Extend if needed.

## License

MIT recommended. Add LICENSE if accepting contributions.