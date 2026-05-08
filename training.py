"""Persistent user training notes — feed corrections back into future generations."""

from __future__ import annotations

import json
import os
from uuid import uuid4

TRAINING_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_notes.json")

ISSUE_TYPES = [
    "角色行為矛盾",
    "名字／稱謂錯誤",
    "地點邏輯錯誤",
    "重複套路／橋段",
    "劇情不合理",
    "文字風格問題",
    "其他",
]


def load_notes() -> list[dict]:
    try:
        with open(TRAINING_FILE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def save_notes(notes: list[dict]) -> None:
    with open(TRAINING_FILE, "w", encoding="utf-8") as f:
        json.dump(notes, f, ensure_ascii=False, indent=2)


def add_note(excerpt: str, issue: str, issue_type: str) -> dict:
    notes = load_notes()
    note = {
        "id": str(uuid4()),
        "excerpt": excerpt.strip(),
        "issue": issue.strip(),
        "type": issue_type,
    }
    notes.append(note)
    save_notes(notes)
    return note


def delete_note(note_id: str) -> None:
    save_notes([n for n in load_notes() if n.get("id") != note_id])


def notes_to_prompt_block(notes: list[dict]) -> str:
    if not notes:
        return ""
    lines = [
        "【用戶訓練指令 — 最高優先級，必須百分之百遵守，不得重犯以下問題】",
    ]
    for n in notes:
        excerpt_part = f"，問題段落：「{n['excerpt']}」" if n.get("excerpt") else ""
        lines.append(f"- [{n['type']}] {n['issue']}{excerpt_part}")
    return "\n".join(lines)
