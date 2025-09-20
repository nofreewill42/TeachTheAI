# TeachTheAI - OpenWebUI Starter

Minimal OpenAI-compatible FastAPI server that plugs into OpenWebUI. There is exactly **one function** you change to use your own local model; everything else is glue so the UI can talk to your code.

## Why OpenWebUI

**Pros**
- Clean chat UI (history, retry, copy) - no UI coding needed
- Uses OpenAI Chat Completions format (works with many tools)  
- Swap models easily - edit one function, keep the same UI
- Fully offline if your model runs locally

**Cons**
- No built-in token-level sampling (e.g., choose top-k per token) - I mean I don't think and I even doubt it exists; but/although I haven't checked
- The OpenAI JSON format for API communication is a bit complex, but you don't need to understand it since server.py handles it automatically

## What this repo does

This repo runs a tiny HTTP server (`server.py`) that implements just enough of the OpenAI API for OpenWebUI to call your code.

- `server.py` exposes two endpoints OpenWebUI needs:
  - `GET /v1/models` - tells UI your model IDs (e.g., "garage-echo-1")
  - `POST /v1/chat/completions` - UI sends chat messages; you return a reply

- You add your model logic by modifying **one function** defined in server.py: `generate_reply(user_text: str)`

- Everything else in `server.py` is plumbing: parse request → call your function → wrap reply in OpenAI JSON (OpenWebUI reads `choices[0].message.content` - referred to in the cons)

**Flow:**
```
OpenWebUI → GET /v1/models → server.py (returns "garage-echo-1")
OpenWebUI → POST /v1/chat/completions → server.py (extracts user text)  
server.py → calls generate_reply(text) → your code returns string
server.py → returns JSON → OpenWebUI shows it in chat
```

## Prerequisites

**For the API server (this repo):**
- Python 3.10+

**For the UI:**
- OpenWebUI installed in any Python environment

**Install OpenWebUI:**
```bash
python -m venv ~/venvs/openwebui
source ~/venvs/openwebui/bin/activate
pip install open-webui
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

## Quick start

Run two processes (API on port 8000, UI on port 3000).

**Process A - API server** (inside this repo - create venv and install reqs):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn server:app --host 127.0.0.1 --port 8000
```

**Process B - OpenWebUI** (another terminal):

```bash
source ~/venvs/openwebui/bin/activate
open-webui serve --host 127.0.0.1 --port 3000
```

Open browser at: **http://127.0.0.1:3000**

## Connect OpenWebUI to your API

Refer to these Spanish UI labels:

1. **Ajustes de Admin** at the bottom left → **Conexiones** → **Añadir conexión** with a + sign on the right
2. **Tipo de Conexión**: Externo  
3. **Tipo de Proveedor**: OpenAI
4. **URL Base API**: `http://127.0.0.1:8000/v1`
5. **Autorización**: **Ninguno**
6. (Opcional) **IDs Modelo**: `garage-echo-1`
7. **Guardar**, then in chat select `garage-echo-1` and send a message

## Sanity checks

Health check:
```bash
curl http://127.0.0.1:8000/
# -> {"ok":true,"model":"garage-echo-1"}
```

Models returned by the GET /v1/models:
```bash
curl http://127.0.0.1:8000/v1/models
```

Chat echo test:
```bash
curl -H "Content-Type: application/json" \
-d '{"model":"garage-echo-1","messages":[{"role":"user","content":"hi"}]}' \
http://127.0.0.1:8000/v1/chat/completions
```

## Troubleshooting

- **Port in use (8000/3000)**: change `--port` or find the conflicting process with lsof -i :8000 and stop it manually
- **OpenWebUI can't reach API**: verify URL Base API is `http://127.0.0.1:8000/v1` and **Autorización: Ninguno**
- **Empty reply**: ensure request includes at least one `{"role":"user","content":"..."}` message  
- **Wrong model ID**: keep "garage-echo-1" or change both the code and OpenWebUI model ID consistently
- **OpenWebUI won't start**: make sure you activated the correct environment where you installed `open-webui`

## FAQ

**Why GET/POST routes?**  
OpenWebUI speaks the OpenAI API format. It discovers models via `GET /v1/models` and sends chats to `POST /v1/chat/completions`. This repo implements just that minimal subset so that you can enjoy the pros of using your custom models through OpenWebUI.

**Can I add streaming later?**  
Yes - this starter uses non-streamed responses for simplicity. You can extend it to stream tokens as they're generated.

**Do I need to understand the OpenAI API?**  
No. The `server.py` handles all the API complexity. You just implement `generate_reply(text) -> text`.

## License

MIT recommended. Add LICENSE if accepting contributions.
