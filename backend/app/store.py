from __future__ import annotations

import sqlite3
from pathlib import Path
from threading import Lock
from typing import Optional, Protocol

from app.models import ReportSummary, ResearchReport


class ReportStore(Protocol):
    def save(self, report: ResearchReport) -> None: ...

    def get(self, report_id: str) -> Optional[ResearchReport]: ...

    def list_reports(self, limit: int = 20) -> list[ReportSummary]: ...

    def clear(self) -> None: ...


class InMemoryReportStore:
    def __init__(self) -> None:
        self._reports: dict[str, ResearchReport] = {}
        self._lock = Lock()

    def save(self, report: ResearchReport) -> None:
        with self._lock:
            self._reports[report.id] = report

    def get(self, report_id: str) -> Optional[ResearchReport]:
        with self._lock:
            return self._reports.get(report_id)

    def list_reports(self, limit: int = 20) -> list[ReportSummary]:
        with self._lock:
            reports = sorted(self._reports.values(), key=lambda item: item.created_at, reverse=True)
        return [
            ReportSummary(
                id=report.id,
                question=report.question,
                created_at=report.created_at,
                paper_count=len(report.papers),
                warning_count=len(report.warnings),
            )
            for report in reports[:limit]
        ]

    def clear(self) -> None:
        with self._lock:
            self._reports.clear()


class SQLiteReportStore:
    def __init__(self, db_path: str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS reports (
                    id TEXT PRIMARY KEY,
                    payload TEXT NOT NULL
                )
                """
            )
            connection.commit()

    def save(self, report: ResearchReport) -> None:
        payload = report.model_dump_json()
        with self._lock:
            with self._connect() as connection:
                connection.execute(
                    "INSERT OR REPLACE INTO reports (id, payload) VALUES (?, ?)",
                    (report.id, payload),
                )
                connection.commit()

    def get(self, report_id: str) -> Optional[ResearchReport]:
        with self._lock:
            with self._connect() as connection:
                row = connection.execute(
                    "SELECT payload FROM reports WHERE id = ?",
                    (report_id,),
                ).fetchone()
        if not row:
            return None
        return ResearchReport.model_validate_json(row[0])

    def list_reports(self, limit: int = 20) -> list[ReportSummary]:
        with self._lock:
            with self._connect() as connection:
                rows = connection.execute("SELECT payload FROM reports").fetchall()
        reports = [ResearchReport.model_validate_json(row[0]) for row in rows]
        reports.sort(key=lambda item: item.created_at, reverse=True)
        return [
            ReportSummary(
                id=report.id,
                question=report.question,
                created_at=report.created_at,
                paper_count=len(report.papers),
                warning_count=len(report.warnings),
            )
            for report in reports[:limit]
        ]

    def clear(self) -> None:
        with self._lock:
            with self._connect() as connection:
                connection.execute("DELETE FROM reports")
                connection.commit()
