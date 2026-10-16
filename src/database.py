"""
Mis-Info Verify - Database Persistence Module.

Manages local SQLite persistence for claims, evidence records, and verification audits.
Uses parameterized queries, transaction safety, and graceful exception handling.
"""

import sqlite3
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

from config import config
from src import VerificationResult, EvidenceItem

logger = logging.getLogger("mindmesh.database")


class DatabaseManager:
    """Manages SQLite database initialization, schema migration, and verification persistence."""

    def __init__(self, db_path: Optional[str] = None) -> None:
        self.db_path = Path(db_path or config.database_path)
        # Ensure parent directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    def get_connection(self) -> sqlite3.Connection:
        """Create and return a database connection with dictionary row formatting."""
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self) -> None:
        """Initialize database schema tables if they do not already exist and run safe migrations."""
        logger.info("Initializing database tables at: %s", self.db_path)
        create_tables_sql = """
        CREATE TABLE IF NOT EXISTS claims (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            raw_input TEXT NOT NULL,
            cleaned_claim TEXT NOT NULL,
            source_url TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS verifications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            claim_id INTEGER NOT NULL,
            status TEXT NOT NULL,
            confidence_score REAL NOT NULL,
            uncertainty_score REAL NOT NULL,
            explanation TEXT NOT NULL,
            summary TEXT,
            limitations TEXT,
            model_used TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (claim_id) REFERENCES claims (id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS evidence (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            verification_id INTEGER NOT NULL,
            evidence_id TEXT,
            title TEXT,
            source_name TEXT NOT NULL,
            url TEXT NOT NULL,
            snippet TEXT NOT NULL,
            reliability_score REAL DEFAULT 0.5,
            published_date TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (verification_id) REFERENCES verifications (id) ON DELETE CASCADE
        );
        """
        try:
            with self.get_connection() as conn:
                conn.executescript(create_tables_sql)

                # Safe non-destructive column migrations for existing databases
                self._migrate_columns_if_missing(conn)

            logger.info("Database schema initialized and verified successfully.")
        except Exception as e:
            logger.error("Failed to initialize database schema: %s", e)
            raise

    def _migrate_columns_if_missing(self, conn: sqlite3.Connection) -> None:
        """Add optional Phase 2 columns if inspecting an older database version."""
        cursor = conn.cursor()

        # Check verifications table columns
        cursor.execute("PRAGMA table_info(verifications)")
        existing_cols = {row["name"] for row in cursor.fetchall()}
        for col, col_type in [("summary", "TEXT"), ("limitations", "TEXT"), ("model_used", "TEXT")]:
            if col not in existing_cols:
                cursor.execute(f"ALTER TABLE verifications ADD COLUMN {col} {col_type}")

        # Check evidence table columns
        cursor.execute("PRAGMA table_info(evidence)")
        ev_cols = {row["name"] for row in cursor.fetchall()}
        for col, col_type in [("evidence_id", "TEXT"), ("title", "TEXT"), ("image_url", "TEXT")]:
            if col not in ev_cols:
                cursor.execute(f"ALTER TABLE evidence ADD COLUMN {col} {col_type}")
        conn.commit()

    def save_claim(self, raw_input: str, cleaned_claim: str, source_url: Optional[str] = None) -> int:
        """Insert a newly extracted claim into the claims table."""
        sql = "INSERT INTO claims (raw_input, cleaned_claim, source_url) VALUES (?, ?, ?)"
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, (raw_input, cleaned_claim, source_url))
            claim_id = cursor.lastrowid
            conn.commit()
            logger.debug("Saved claim with ID: %d", claim_id)
            return claim_id

    def save_verification(self, claim_id: int, result: VerificationResult) -> int:
        """Persist a completed VerificationResult linked to a claim_id with full evidence relations.

        Uses transaction block to guarantee integrity across verifications and evidence tables.
        """
        ver_sql = """
        INSERT INTO verifications (
            claim_id, status, confidence_score, uncertainty_score,
            explanation, summary, limitations, model_used
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        ev_sql = """
        INSERT INTO evidence (
            verification_id, evidence_id, title, source_name,
            url, snippet, reliability_score, published_date, image_url
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                ver_sql,
                (
                    claim_id,
                    result.status.value,
                    result.confidence_score,
                    result.uncertainty_score,
                    result.explanation,
                    result.summary or "",
                    result.limitations or "",
                    result.model_used or "Unknown",
                ),
            )
            verification_id = cursor.lastrowid

            for item in result.evidence_items:
                cursor.execute(
                    ev_sql,
                    (
                        verification_id,
                        getattr(item, "id", ""),
                        getattr(item, "title", item.source_name),
                        item.source_name,
                        item.url,
                        item.snippet,
                        item.reliability_score,
                        item.published_date or "",
                        getattr(item, "image_url", None) or "",
                    ),
                )
            conn.commit()
            logger.info("Saved verification audit with ID: %d (linked %d evidence items)", verification_id, len(result.evidence_items))
            return verification_id

    def save_full_pipeline_audit(self, result: VerificationResult) -> int:
        """Convenience method to save both claim and verification in a single call."""
        claim_id = self.save_claim(
            raw_input=result.claim.raw_input,
            cleaned_claim=result.claim.cleaned_claim,
            source_url=result.claim.source_url,
        )
        return self.save_verification(claim_id, result)

    def list_recent_claims(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieve recent claims from database."""
        sql = "SELECT id, raw_input, cleaned_claim, source_url, created_at FROM claims ORDER BY id DESC LIMIT ?"
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, (limit,))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def list_recent_verifications(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieve recent verifications with associated claim text."""
        sql = """
        SELECT
            v.id AS verification_id,
            v.status,
            v.confidence_score,
            v.uncertainty_score,
            v.summary,
            v.model_used,
            v.created_at,
            c.cleaned_claim,
            c.raw_input
        FROM verifications v
        JOIN claims c ON v.claim_id = c.id
        ORDER BY v.id DESC
        LIMIT ?
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, (limit,))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
