# TeachTheAI - model/my-first-model Branch

This branch builds on **main** by adding a minimal local model implementation. You now have a dedicated entrypoint (`serve_model.py`) that wires your custom model into the OpenAI-compatible API for OpenWebUI.

## What's New vs Main

**Added files:**
- `model.py` - your model class with `generate(text) -> str` method
- `serve_model.py` - FastAPI app that calls your model (instead of the generic echo)

**Unchanged:** `server.py` (kept as reference), `requirements.txt`, `.gitignore`

## How Your Model Works

### Edit ONE place: `model.py`
```python
class MyFirstModel:
    def __init__(self):
        # Load weights/resources here once
        pass

    def generate(self, text: str) -> str:
        # Your inference logic here
        # Input: user text from OpenWebUI chat
        # Output: string to show in UI
        return f"[MY MODEL]: {text}"  # Replace with real generation
```

### `serve_model.py` calls it automatically
```python
# Inside serve_model.py (don't edit unless you know why)
def generate_reply(user_text: str) -> str:
    return model.generate(user_text)  # Calls YOUR model - that is created from inside serve_model.py
```

The rest wraps your string into OpenAI JSON that OpenWebUI expects.

## Sanity Checks

**Health:**
```bash
curl http://127.0.0.1:8000/
# -> {"ok":true,"model":"my-model-1"}
```

**Models:**
```bash
curl http://127.0.0.1:8000/v1/models
```

**Test your model:**
```bash
curl -H "Content-Type: application/json" \
  -d '{"model":"my-model-1","messages":[{"role":"user","content":"hello"}]}' \
  http://127.0.0.1:8000/v1/chat/completions
```

## Compared to Main Branch

| Main (`server.py`) | This Branch (`serve_model.py`) |
|---|---|
| Echoes user input | Calls your `model.generate()` |
| Generic template | Model-specific wiring |
| `MODEL_ID = "garage-echo-1"` | `MODEL_ID = "my-model-1"` (change as needed) |
| Good for testing API | Good for actual model dev |

Both use the same OpenWebUI connection steps.

## Next Steps

1. Test with simple inputs in OpenWebUI - tested, it works alright
2. Export chats → prepare liked branches (when ready)
3. Implement real LM in `model.py.generate()` - working at the character level
4. Test LM with randomly filled embeddings as a sanity check.
5. Build out the training pipeline.
6. Try a model with some training, see if there is a significant difference or not.
6. Online training? Is it possible to train the model one step when I hit like for a message?

## License

Recommended MIT (add `LICENSE` file if accepting contributions)