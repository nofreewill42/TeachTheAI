# TeachTheAI - prototype
Minimal OpenAI-compatible FastAPI server that plugs into OpenWebUI, plus a branch with a lightweight local character LM prototype.

## Quick start
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn server:app --host 127.0.0.1 --port 8000
```

**OpenWebUI:**
- Ajustes de Admin → Conexiones → Añadir conexión
- Tipo de Conexión: Externo
- Tipo de Proveedor: OpenAI
- URL Base API: http://127.0.0.1:8000/v1
- Autorización: Ninguno
- Selecciona garage-echo-1 en el chat

(works with the non-streamed /v1/chat/completions response)

## Branches
- `main` - echo server scaffold for OpenWebUI. Endpoints: `/v1/models` and `/v1/chat/completions`. Good for wiring and smoke tests.
- `model/my-first-model` - adds a simple local model surface, training scripts, and a serve entrypoint that calls `model.generate(...)`.

## Data format
Training conversations live at `data/chats.json` as a list of chats. Each chat is a list of turns - train means there is intention to train the model on that turn's completion:

```json
[
  [
    {"role":"user","message":"hi","train":false},
    {"role":"ai","message":"hi","train":false},
    {"role":"user","message":"hi","train":true},
    {"role":"ai","message":"hi","train":true},
  ],
  [
    {"role":"user","message":"a","train":false},
    {"role":"ai","message":"a","train":true}
  ]
]
```

## Minimal dataset
`dataset.py` exposes one conversation per index:

```python
from dataset import DS
ds = DS("data/chats.json")
conv = ds[0] # list[dict], untouched
print(len(conv), conv) # you can later build masks from the 'train' flags
```

## Minimal tokenizer
`tokenizer.py` is a fixed ASCII char tokenizer with BOS/EOS. Quick demo:

```bash
python tokenizer.py
# prints vocab size, a round-trip encode/decode, and UNK handling
```

## Local model branch
Use the prototype branch for model work:

```bash
git checkout model/my-first-model
# Train a tiny baseline
python 02_train_model.py --data data/chats.json --max_steps 500
# Serve the model
uvicorn 03_serve_model:app --host 127.0.0.1 --port 8000
```

Edit only `model.py` to change generation. `03_serve_model.py` calls your model and exposes the same OpenAI endpoints.

## Roadmap
- Keep one chat as one training sample - build masks from train per turn.
- Start with char-level LM for simplicity - swap later if it shows promise.
