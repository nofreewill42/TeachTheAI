#!/usr/bin/env python3
# scripts/extract_liked_branches.py
#
# Usage:
#   python scripts/extract_liked_branches.py ./chats/chat-export-1758401790554.json > ./chats/branches.json
#   # If you omit the arg, it defaults to ./chats/chat-export-1758401790554.json

import json
import sys
from pathlib import Path

DEFAULT_PATH = "./chats/chat-export-1758401790554.json"
DEFAULT_OUTPUT_PATH = "./data/chats.json"

def load_data(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def liked(msg: dict) -> bool:
    # Only accept the exact export format:
    ann = msg.get("annotation")
    return bool(isinstance(ann, dict) and ann.get("rating") == 1)

def children_sorted(messages: dict, mid: str):
    ids = messages.get(mid, {}).get("childrenIds") or []
    # Stable order: by timestamp then id
    return sorted(
        (i for i in ids if i in messages),
        key=lambda k: (messages[k].get("timestamp", 0), k),
    )

def find_roots(messages: dict):
    all_ids = set(messages.keys())
    roots = []
    for mid, m in messages.items():
        pid = m.get("parentId")
        if not pid or pid not in all_ids:
            roots.append(mid)
    # Prefer user roots, then by timestamp
    roots.sort(key=lambda mid: (messages[mid].get("role") != "user", messages[mid].get("timestamp", 0)))
    return roots

def role_out(role: str) -> str:
    return "ai" if role == "assistant" else "user"

def enumerate_branches(messages: dict):
    """Yield each root→leaf path as a list of message ids."""
    for root in find_roots(messages):
        stack = [(root, [root])]
        while stack:
            node, path = stack.pop()
            kids = children_sorted(messages, node)
            if not kids:
                yield path
            else:
                # push in reverse to keep natural order when popping
                for k in reversed(kids):
                    stack.append((k, path + [k]))

def path_to_chain(messages: dict, path_ids: list):
    chain = []
    for mid in path_ids:
        m = messages[mid]
        chain.append({
            "role": role_out(m.get("role", "user")),
            "message": m.get("content", ""),
            "train": liked(m),
        })
    return chain

def extract_branches(conversations: list):
    all_branches = []
    for conv in conversations:
        chat = (conv or {}).get("chat") or {}
        messages = ((chat.get("history") or {}).get("messages")) or {}
        if not isinstance(messages, dict) or not messages:
            continue
        # Normalize missing fields lightly
        for m in messages.values():
            m.setdefault("childrenIds", [])
            m.setdefault("parentId", None)
            m.setdefault("role", "user")
            m.setdefault("content", "")
        for path_ids in enumerate_branches(messages):
            all_branches.append(path_to_chain(messages, path_ids))
    return all_branches

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATH
    chats = load_data(path)
    conversations = chats if isinstance(chats, list) else [chats]
    result = extract_branches(conversations)
    data_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUTPUT_PATH
    data_path = Path(data_path)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    with open(data_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()


#!/usr/bin/env python3
# scripts/branches_last_liked.py
#
# Usage:
#   python scripts/branches_last_liked.py ./chats/chat-export-1758403710257.json > ./chats/branches_last_liked.json
#   # If path is omitted, defaults to ./chats/chat-export-1758403710257.json
#
# What it does:
# - Parses the OpenWebUI export format you shared.
# - Collects ALL liked assistant messages first (annotation.rating == 1).
# - For each liked message, if it has ANY liked descendant, it is skipped.
# - For liked messages that are the LAST liked in their branch, outputs the
#   conversation path from root -> that liked message.
# - Each output conversation is a list of {role, message, train} dicts,
#   where train==True only for liked assistant messages.

import json
import sys
from typing import Dict, List, Any, Iterable

DEFAULT_PATH = "./chats/chat-export-1758403710257.json"


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
