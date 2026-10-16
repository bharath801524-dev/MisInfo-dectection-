"""
Unit tests for Mis-Info Verify Temporal Tracker module.
"""

from src import VerificationStatus, ExtractedClaim, VerificationResult
from src.temporal_tracker import TemporalTracker


def test_record_snapshot_and_get_timeline() -> None:
    tracker = TemporalTracker()
    claim = ExtractedClaim(
        raw_input="Developing story regarding local transit update.",
        cleaned_claim="Developing story regarding local transit update.",
    )
    result = VerificationResult(
        claim=claim,
        status=VerificationStatus.NEEDS_VERIFICATION,
        confidence_score=0.4,
        uncertainty_score=0.7,
        explanation="Initial developing story.",
        evidence_items=[],
        source_urls=[],
    )

    snapshot = tracker.record_assessment(result, trigger_note="Initial post")
    assert snapshot.status == VerificationStatus.NEEDS_VERIFICATION
    assert snapshot.confidence_score == 0.4

    timeline = tracker.get_timeline(claim.cleaned_claim)
    assert len(timeline) == 1
    assert timeline[0].status == VerificationStatus.NEEDS_VERIFICATION


def test_detect_status_shift() -> None:
    tracker = TemporalTracker()
    claim = ExtractedClaim(
        raw_input="Power grid failure reported in city center.",
        cleaned_claim="Power grid failure reported in city center.",
    )

    # Initial assessment: NEEDS VERIFICATION
    r1 = VerificationResult(
        claim=claim,
        status=VerificationStatus.NEEDS_VERIFICATION,
        confidence_score=0.3,
        uncertainty_score=0.8,
        explanation="Developing report.",
        evidence_items=[],
        source_urls=[],
    )
    tracker.record_assessment(r1, trigger_note="Breaking claim")

    # Second assessment: LIKELY RELIABLE after utility statement
    r2 = VerificationResult(
        claim=claim,
        status=VerificationStatus.LIKELY_RELIABLE,
        confidence_score=0.85,
        uncertainty_score=0.15,
        explanation="Utility company confirmed outage.",
        evidence_items=[],
        source_urls=[],
    )
    tracker.record_assessment(r2, trigger_note="Official utility confirmation")

    shift = tracker.detect_status_shift(claim.cleaned_claim)
    assert shift is not None
    assert shift["previous_status"] == VerificationStatus.NEEDS_VERIFICATION.value
    assert shift["current_status"] == VerificationStatus.LIKELY_RELIABLE.value
    assert shift["reason"] == "Official utility confirmation"


def test_no_shift_when_status_unchanged() -> None:
    tracker = TemporalTracker()
    claim = ExtractedClaim(
        raw_input="Steady weather conditions expected.",
        cleaned_claim="Steady weather conditions expected.",
    )
    r1 = VerificationResult(
        claim=claim,
        status=VerificationStatus.LIKELY_RELIABLE,
        confidence_score=0.8,
        uncertainty_score=0.2,
        explanation="Forecast confirmed.",
    )
    tracker.record_assessment(r1)

    r2 = VerificationResult(
        claim=claim,
        status=VerificationStatus.LIKELY_RELIABLE,
        confidence_score=0.85,
        uncertainty_score=0.15,
        explanation="Second weather agency concurs.",
    )
    tracker.record_assessment(r2)

    shift = tracker.detect_status_shift(claim.cleaned_claim)
    assert shift is None
