
from __future__ import annotations

import json
from pathlib import Path


class HistoryManager:
    """
    Lavnivå-komponent for lesing av historikkdata fra disk.

    Klassen laster alle lagrede økter fra JSON-filer og returnerer
    dem som strukturerte Python-objekter. Den inneholder ingen
    aggregeringslogikk.
    """

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