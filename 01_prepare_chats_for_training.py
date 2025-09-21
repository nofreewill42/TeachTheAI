
import json
import sys
from typing import Dict, List, Any, Iterable

DEFAULT_PATH = "./chats/chat-export-1758424771315.json"
DEFAULT_OUTPUT_PATH = "./data/chats.json"


def load_export(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else [data]


def get_messages(chat: dict) -> Dict[str, dict]:
    return ((chat.get("history") or {}).get("messages")) or {}


def normalize_messages(messages: Dict[str, dict]) -> None:
    for m in messages.values():
        m.setdefault("childrenIds", [])
        m.setdefault("parentId", None)
        m.setdefault("content", "")
        m.setdefault("role", "user")


def is_liked_assistant(msg: dict) -> bool:
    ann = msg.get("annotation")
    return (
        msg.get("role") == "assistant"
        and isinstance(ann, dict)
        and ann.get("rating") == 1
    )


def children_ids(messages: Dict[str, dict], mid: str) -> List[str]:
    return [c for c in (messages.get(mid, {}).get("childrenIds") or []) if c in messages]


def has_liked_descendant(messages: Dict[str, dict], start_id: str) -> bool:
    """Return True if ANY descendant (excluding start) is a liked assistant."""
    stack = children_ids(messages, start_id)[:]
    while stack:
        nid = stack.pop()
        m = messages[nid]
        if is_liked_assistant(m):
            return True
        stack.extend(children_ids(messages, nid))
    return False


def path_to_root(messages: Dict[str, dict], mid: str) -> List[str]:
    """Return [root ... mid] by following parentId."""
    path = [mid]
    cur = mid
    seen = set([mid])
    while True:
        pid = messages[cur].get("parentId")
        if not pid or pid not in messages or pid in seen:
            break
        path.append(pid)
        seen.add(pid)
        cur = pid
    path.reverse()
    return path


def role_out(role: str) -> str:
    return "ai" if role == "assistant" else "user"


def chain_from_path(messages: Dict[str, dict], ids: Iterable[str]) -> List[dict]:
    chain = []
    for mid in ids:
        m = messages[mid]
        chain.append({
            "role": role_out(m.get("role", "user")),
            "message": m.get("content", ""),
            "train": is_liked_assistant(m),
        })
    return chain


def extract_branches_last_liked(conversations: List[dict]) -> List[List[dict]]:
    all_branches: List[List[dict]] = []

    for conv in conversations:
        chat = (conv or {}).get("chat") or {}
        messages = get_messages(chat)
        if not messages:
            continue

        normalize_messages(messages)

        # Collect liked assistant messages first, then iterate by timestamp order.
        liked_ids = [
            mid for mid, m in messages.items() if is_liked_assistant(m)
        ]
        liked_ids.sort(key=lambda mid: (messages[mid].get("timestamp", 0), mid))

        for lid in liked_ids:
            # Skip if there exists ANY liked descendant under this liked node.
            if has_liked_descendant(messages, lid):
                continue

            # Last liked in its branch → build path from root to this liked node.
            path_ids = path_to_root(messages, lid)
            branch = chain_from_path(messages, path_ids)
            all_branches.append(branch)

    return all_branches


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATH
    conversations = load_export(path)
    result = extract_branches_last_liked(conversations)
    from pathlib import Path
    data_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUTPUT_PATH
    data_path = Path(data_path)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    with open(data_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
