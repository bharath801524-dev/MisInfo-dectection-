"""
Mis-Info Verify - Temporal Tracker Module.

Tracks how verification assessments, evidence availability, and narrative framing
evolve over time for breaking or developing news stories.
"""

import logging
from datetime import datetime, timezone
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from src import VerificationStatus, VerificationResult

logger = logging.getLogger("mindmesh.temporal_tracker")


class TemporalSnapshot(BaseModel):
    """Represents the verification state of a claim at a specific timestamp."""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    status: VerificationStatus
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    uncertainty_score: float = Field(..., ge=0.0, le=1.0)
    source_count: int = Field(default=0, ge=0)
    trigger_note: str = Field(default="Initial assessment")


class TemporalHistory(BaseModel):
    """Collection of verification snapshots for a single claim over time."""
    claim_text: str
    snapshots: List[TemporalSnapshot] = Field(default_factory=list)


class TemporalTracker:
    """Manages and analyzes chronological changes in claim credibility."""

    def __init__(self) -> None:
        # In-memory store for tracking claims across time (synced to database in Phase 2)
        self._history: Dict[str, TemporalHistory] = {}
        logger.debug("TemporalTracker initialized.")

    def record_assessment(
        self,
        result: VerificationResult,
        trigger_note: str = "Standard check",
    ) -> TemporalSnapshot:
        """Record an assessment snapshot in the claim's temporal timeline.

        Args:
            result: VerificationResult to log.
            trigger_note: Brief note explaining why this snapshot was recorded.

        Returns:
            The newly created TemporalSnapshot.
        """
        claim_key = result.claim.cleaned_claim.strip().lower()
        if claim_key not in self._history:
            self._history[claim_key] = TemporalHistory(claim_text=result.claim.cleaned_claim)

        snapshot = TemporalSnapshot(
            status=result.status,
            confidence_score=result.confidence_score,
            uncertainty_score=result.uncertainty_score,
            source_count=len(result.evidence_items),
            trigger_note=trigger_note,
        )

        self._history[claim_key].snapshots.append(snapshot)
        logger.info(
            "Logged snapshot for claim '%s': %s (total snapshots: %d)",
            claim_key[:40],
            snapshot.status.value,
            len(self._history[claim_key].snapshots),
        )
        return snapshot

    def get_timeline(self, claim_text: str) -> List[TemporalSnapshot]:
        """Retrieve chronological assessment history for a specific claim."""
        claim_key = claim_text.strip().lower()
        history = self._history.get(claim_key)
        return history.snapshots if history else []

    def detect_status_shift(self, claim_text: str) -> Optional[Dict[str, str]]:
        """Detect whether a claim's verdict has flipped or shifted significantly."""
        timeline = self.get_timeline(claim_text)
        if len(timeline) < 2:
            return None

        previous = timeline[-2]
        current = timeline[-1]

        if previous.status != current.status:
            return {
                "previous_status": previous.status.value,
                "current_status": current.status.value,
                "confidence_change": f"{current.confidence_score - previous.confidence_score:+.2f}",
                "reason": current.trigger_note,
            }
        return None
