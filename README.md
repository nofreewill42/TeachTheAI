# TeachTheAI - OpenWebUI starter

Minimal OpenAI-compatible FastAPI server that plugs into OpenWebUI. It echoes the last user message so you can swap in your own local model later.

## What you get
- A tiny HTTP API exposing:
  - `GET /v1/models`
  - `POST /v1/chat/completions`
- Works with OpenWebUI out of the box
- One function to replace with your own model call

## Prerequisites
- Python 3.10+ installed
- OpenWebUI installed in any env where `open-webui` runs

## Quick start

You will run two processes:
- Process A: API on port 8000
- Process B: OpenWebUI on port 3000

```bash
# Process A - run the API inside this repo
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn server:app --host 127.0.0.1 --port 8000
```

```bash
# Process B - start OpenWebUI in another terminal
# activate the environment where you installed open-webui
# example:
source ~/venvs/openwebui/bin/activate
open-webui serve --host 127.0.0.1 --port 3000
```

Open your browser at `http://127.0.0.1:3000`.

## Connect OpenWebUI to the API

Use Spanish labels exactly as shown for clarity:

1. **Ajustes de Admin → Conexiones → Añadir conexión**  
2. **Tipo de Conexión**: Externo  
3. **Tipo de Proveedor**: OpenAI  
4. **URL Base API**: `http://127.0.0.1:8000/v1`  
5. **Autorización**: **Ninguno**  
6. (Opcional) **IDs Modelo**: `garage-echo-1`  
7. **Guardar**, then select `garage-echo-1` in the chat view and send a message

## Sanity checks

```bash
# Health
curl http://127.0.0.1:8000/
# -> {"ok":true,"model":"garage-echo-1"}

# Models
curl http://127.0.0.1:8000/v1/models

# Chat echo
curl -H "Content-Type: application/json" \
  -d '{"model":"garage-echo-1","messages":[{"role":"user","content":"hi"}]}' \
  http://127.0.0.1:8000/v1/chat/completions
```

## Swap in your own model

Edit `server.py` and change only this function:

```python
def generate_reply(user_text: str) -> str:
    # return my_model.generate(user_text)
    return user_text
```

Optional: load weights once at startup in `server.py`:

```python
@app.on_event("startup")
def load_model():
    global my_model
    my_model = load_your_weights()
```

## Troubleshooting

- Cannot open the UI: ensure OpenWebUI is running on `http://127.0.0.1:3000`.
- OpenWebUI cannot reach the API: in the connection use **URL Base API** `http://127.0.0.1:8000/v1` and **Autorización: Ninguno**.
- Port in use: pick another `--port` or stop the conflicting process.
- Empty reply: your request must include at least one `{"role":"user","content":"..."}` message.

## License

MIT recommended. Add a `LICENSE` file if you plan to accept contributions.
