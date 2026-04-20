#lägga till senare: export a session; filter by trip; validate schema


from __future__ import annotations

import json
from pathlib import Path


class HistoryManager:
    """Backend for loading saved history sessions from disk."""

    def __init__(self) -> None:
        self.sessions_dir = Path("data/history/sessions")
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def list_sessions(self) -> list[dict]:
        """Read all session JSON files and return valid session dicts."""
        sessions: list[dict] = []

        for file_path in sorted(self.sessions_dir.glob("*.json")):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                if isinstance(data, dict):
                    sessions.append(data)

            except Exception:
                # Skip broken files for now.
                # Later add logging if needed.
                continue

        return sessions