#skriver session historikk till json fil

import json
from pathlib import Path


class SessionManager:
    def __init__(self, sessions_dir: str = "data/history/sessions") -> None:
        self.sessions_dir = Path(sessions_dir)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def save_session(self, session_data: dict) -> Path:
        session_id = session_data["session_id"]
        file_path = self.sessions_dir / f"{session_id}.json"

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(session_data, f, indent=2, ensure_ascii=False)

        return file_path