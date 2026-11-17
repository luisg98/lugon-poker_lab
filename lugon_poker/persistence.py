"""SQLite persistence for observed hand histories and player profiles."""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3
from typing import Any, Iterable

from .history import HandHistory
from .profiles import PlayerProfile, build_profiles


def parse_histories_payload(payload: Any) -> list[HandHistory]:
    """Parse a JSON payload containing one hand, a list of hands, or {'hands': [...]}."""
    if isinstance(payload, dict):
        if "hands" in payload:
            raw_list = payload["hands"]
            if not isinstance(raw_list, list) or not raw_list:
                raise ValueError("'hands' field must be a non-empty list of hand history objects")
            return [HandHistory.from_dict(item) for item in raw_list]
        return [HandHistory.from_dict(payload)]
    if isinstance(payload, list):
        if not payload:
            raise ValueError("hand history list cannot be empty")
        return [HandHistory.from_dict(item) for item in payload]
    raise ValueError("hand history payload must be a JSON object or array")


def load_history_file(path: Path) -> list[HandHistory]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return parse_histories_payload(raw)


class ProfileStore:
    """Local SQLite store for deduplicated hand histories and player profiles."""

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._init_schema()

    def _init_schema(self) -> None:
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS hands (
                    hand_id TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS hand_players (
                    hand_id TEXT NOT NULL,
                    nickname TEXT NOT NULL,
                    position TEXT NOT NULL,
                    PRIMARY KEY (hand_id, nickname),
                    FOREIGN KEY (hand_id) REFERENCES hands(hand_id) ON DELETE CASCADE
                )
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_hand_players_nickname ON hand_players(nickname)"
            )

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "ProfileStore":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def import_hands(self, histories: Iterable[HandHistory]) -> dict[str, int]:
        imported = 0
        skipped = 0
        with self._conn:
            for history in histories:
                payload = json.dumps(history.to_dict(), ensure_ascii=False)
                cursor = self._conn.execute(
                    "INSERT OR IGNORE INTO hands (hand_id, payload_json) VALUES (?, ?)",
                    (history.hand_id, payload),
                )
                if cursor.rowcount == 1:
                    imported += 1
                    for seat in history.players:
                        self._conn.execute(
                            "INSERT INTO hand_players (hand_id, nickname, position) VALUES (?, ?, ?)",
                            (history.hand_id, seat.nickname, seat.position),
                        )
                else:
                    skipped += 1
        total = self.hand_count()
        return {"imported": imported, "skipped_duplicates": skipped, "total_stored": total}

    def hand_count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM hands").fetchone()
        return int(row[0]) if row else 0

    def list_nicknames(self) -> list[str]:
        rows = self._conn.execute(
            "SELECT DISTINCT nickname FROM hand_players ORDER BY nickname ASC"
        ).fetchall()
        return [str(row[0]) for row in rows]

    def load_histories(self, nickname: str | None = None) -> list[HandHistory]:
        if nickname is None:
            rows = self._conn.execute(
                "SELECT payload_json FROM hands ORDER BY rowid ASC"
            ).fetchall()
        else:
            rows = self._conn.execute(
                """
                SELECT h.payload_json
                FROM hands h
                JOIN hand_players hp ON hp.hand_id = h.hand_id
                WHERE hp.nickname = ?
                ORDER BY h.rowid ASC
                """,
                (nickname,),
            ).fetchall()
        return [HandHistory.from_dict(json.loads(row[0])) for row in rows]

    def get_profile(self, nickname: str) -> PlayerProfile | None:
        histories = self.load_histories(nickname=nickname)
        if not histories:
            return None
        profiles = build_profiles(histories)
        return profiles.get(nickname)

    def all_profiles(self) -> dict[str, PlayerProfile]:
        histories = self.load_histories()
        return build_profiles(histories)
