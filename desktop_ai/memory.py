"""Persistent memory storage and retrieval for assistant turns."""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import re
import threading

from desktop_ai.config import MemoryConfig
from desktop_ai.types import MemoryRecord

_TOKEN_PATTERN = re.compile(r"[a-z0-9]{2,}")
_STOP_WORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "for",
        "from",
        "how",
        "i",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "what",
        "when",
        "where",
        "who",
        "why",
        "with",
        "you",
    }
)


def _truncate_text(value: str, *, max_chars: int) -> str:
    """Collapse whitespace and truncate text to a bounded character budget."""
    collapsed: str = " ".join(str(value).split()).strip()
    if len(collapsed) <= max_chars:
        return collapsed
    if max_chars <= 3:
        return collapsed[:max_chars]
    return collapsed[: max_chars - 3].rstrip() + "..."


def _tokenize(value: str) -> set[str]:
    """Tokenize input text into lowercase terms with light stop-word filtering."""
    tokens: set[str] = set(_TOKEN_PATTERN.findall(value.lower()))
    return {token for token in tokens if token not in _STOP_WORDS}


def _parse_datetime(value: str) -> datetime:
    """Parse ISO datetime text while preserving UTC defaults."""
    try:
        parsed: datetime = datetime.fromisoformat(value)
    except ValueError:
        return datetime.now(timezone.utc)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


@dataclass(slots=True)
class SQLiteMemoryStore:
    """SQLite-backed memory store with relevance-based recall."""

    config: MemoryConfig
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger(__name__))
    _connection: sqlite3.Connection = field(init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _fts_enabled: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        """Open SQLite connection and initialize schema."""
        db_path: Path = self.config.database_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(
            db_path,
            check_same_thread=False,
            timeout=8.0,
        )
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA journal_mode=WAL;")
        self._connection.execute("PRAGMA synchronous=NORMAL;")
        self._initialize_schema()

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        with self._lock:
            self._connection.close()

    def remember(
        self,
        *,
        created_at: datetime,
        user_note: str | None,
        assistant_reply: str,
        context: Mapping[str, str],
        action_summary: str,
    ) -> None:
        """Persist one assistant turn to durable memory."""
        note_text: str = _truncate_text(user_note or "", max_chars=600)
        reply_text: str = _truncate_text(assistant_reply or "", max_chars=1400)
        action_text: str = _truncate_text(action_summary or "", max_chars=1000)
        context_text: str = self._render_context_summary(context)
        searchable_text: str = " ".join(
            value for value in (note_text, reply_text, action_text, context_text) if value
        )
        if not searchable_text:
            searchable_text = "(empty memory record)"

        created_iso: str = created_at.astimezone(timezone.utc).isoformat()
        with self._lock:
            cursor = self._connection.execute(
                """
                INSERT INTO memory_events (
                    created_at,
                    user_note,
                    assistant_reply,
                    action_summary,
                    context_summary,
                    searchable_text
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    created_iso,
                    note_text,
                    reply_text,
                    action_text,
                    context_text,
                    searchable_text,
                ),
            )
            row_id: int = int(cursor.lastrowid)
            if self._fts_enabled:
                self._connection.execute(
                    "INSERT INTO memory_events_fts(rowid, searchable_text) VALUES (?, ?)",
                    (row_id, searchable_text),
                )
            self._prune_if_needed_locked()
            self._connection.commit()

    def recall(self, *, query: str, limit: int) -> tuple[MemoryRecord, ...]:
        """Recall relevant memories using FTS when available, else lexical fallback."""
        bounded_limit: int = max(0, min(50, int(limit)))
        if bounded_limit <= 0:
            return ()

        query_text: str = query.strip()
        with self._lock:
            if self._fts_enabled and query_text:
                fts_query: str = self._build_fts_query(query_text)
                if fts_query:
                    rows = self._connection.execute(
                        """
                        SELECT
                            events.id,
                            events.created_at,
                            events.user_note,
                            events.assistant_reply,
                            events.action_summary,
                            events.context_summary,
                            bm25(memory_events_fts) AS rank
                        FROM memory_events_fts
                        JOIN memory_events AS events
                            ON events.id = memory_events_fts.rowid
                        WHERE memory_events_fts MATCH ?
                        ORDER BY rank ASC, events.id DESC
                        LIMIT ?
                        """,
                        (fts_query, bounded_limit),
                    ).fetchall()
                    if rows:
                        return tuple(
                            self._row_to_record(row, score=self._rank_to_score(row["rank"]))
                            for row in rows
                        )

            candidate_rows = self._connection.execute(
                """
                SELECT
                    id,
                    created_at,
                    user_note,
                    assistant_reply,
                    action_summary,
                    context_summary,
                    searchable_text
                FROM memory_events
                ORDER BY id DESC
                LIMIT ?
                """,
                (self.config.search_lookback,),
            ).fetchall()

        return self._fallback_recall(
            rows=tuple(candidate_rows),
            query=query_text,
            limit=bounded_limit,
        )

    def _initialize_schema(self) -> None:
        """Create required tables and FTS index when available."""
        with self._lock:
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    user_note TEXT NOT NULL,
                    assistant_reply TEXT NOT NULL,
                    action_summary TEXT NOT NULL,
                    context_summary TEXT NOT NULL,
                    searchable_text TEXT NOT NULL
                )
                """
            )
            self._connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_memory_events_created_at ON memory_events(created_at)"
            )
            try:
                self._connection.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS memory_events_fts
                    USING fts5(searchable_text)
                    """
                )
                self._fts_enabled = True
            except sqlite3.OperationalError as error:
                self._fts_enabled = False
                self.logger.warning(
                    "SQLite FTS5 unavailable for memory recall; using lexical fallback. Detail: %s",
                    error,
                )
            self._connection.commit()

    def _build_fts_query(self, query: str) -> str:
        """Build a safe FTS query from filtered token terms."""
        tokens: list[str] = sorted(_tokenize(query))
        if not tokens:
            return ""
        terms: list[str] = [f"{token}*" for token in tokens[:12]]
        return " OR ".join(terms)

    def _render_context_summary(self, context: Mapping[str, str]) -> str:
        """Render context map into a bounded summary string."""
        if not context:
            return ""
        items: list[str] = [
            f"{key}={_truncate_text(value, max_chars=160)}"
            for key, value in sorted(context.items())
        ]
        return _truncate_text("; ".join(items), max_chars=self.config.context_chars)

    def _prune_if_needed_locked(self) -> None:
        """Delete oldest records if the configured retention limit is exceeded."""
        current_count: int = int(
            self._connection.execute("SELECT COUNT(*) FROM memory_events").fetchone()[0]
        )
        overflow: int = current_count - self.config.max_entries
        if overflow <= 0:
            return

        rows = self._connection.execute(
            "SELECT id FROM memory_events ORDER BY id ASC LIMIT ?",
            (overflow,),
        ).fetchall()
        stale_ids: list[int] = [int(row["id"]) for row in rows]
        if not stale_ids:
            return

        placeholders: str = ",".join("?" for _ in stale_ids)
        if self._fts_enabled:
            self._connection.execute(
                f"DELETE FROM memory_events_fts WHERE rowid IN ({placeholders})",
                stale_ids,
            )
        self._connection.execute(
            f"DELETE FROM memory_events WHERE id IN ({placeholders})",
            stale_ids,
        )

    def _fallback_recall(
        self,
        *,
        rows: tuple[sqlite3.Row, ...],
        query: str,
        limit: int,
    ) -> tuple[MemoryRecord, ...]:
        """Rank memories in Python when FTS results are unavailable."""
        if not rows:
            return ()

        if not query:
            selected_rows = rows[:limit]
            return tuple(
                self._row_to_record(row, score=max(0.05, 1.0 - (index * 0.08)))
                for index, row in enumerate(selected_rows)
            )

        query_tokens: set[str] = _tokenize(query)
        if not query_tokens:
            return ()

        scored_rows: list[tuple[float, sqlite3.Row]] = []
        total_rows: int = len(rows)
        for index, row in enumerate(rows):
            searchable_text: str = str(row["searchable_text"])
            row_tokens: set[str] = _tokenize(searchable_text)
            overlap: int = len(query_tokens.intersection(row_tokens))
            if overlap == 0:
                continue
            lexical_score: float = overlap / float(len(query_tokens))
            recency_weight: float = 1.0 - min(1.0, index / max(1, total_rows - 1))
            combined_score: float = lexical_score + (0.15 * recency_weight)
            scored_rows.append((combined_score, row))

        if not scored_rows:
            return ()

        scored_rows.sort(key=lambda item: item[0], reverse=True)
        return tuple(
            self._row_to_record(row, score=score)
            for score, row in scored_rows[:limit]
        )

    def _rank_to_score(self, rank: float | None) -> float:
        """Convert FTS rank to a positive similarity-like score."""
        if rank is None:
            return 0.0
        normalized_rank: float = abs(float(rank))
        return 1.0 / (1.0 + normalized_rank)

    def _row_to_record(self, row: sqlite3.Row, *, score: float) -> MemoryRecord:
        """Convert SQLite row data into MemoryRecord domain object."""
        return MemoryRecord(
            id=int(row["id"]),
            created_at=_parse_datetime(str(row["created_at"])),
            user_note=str(row["user_note"]),
            assistant_reply=str(row["assistant_reply"]),
            action_summary=str(row["action_summary"]),
            context_summary=str(row["context_summary"]),
            score=max(0.0, float(score)),
        )
