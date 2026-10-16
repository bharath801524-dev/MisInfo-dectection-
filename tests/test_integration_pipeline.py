"""
Integration Pipeline Tests for Mis-Info Verify.

Exercises the full internal sequence across all modules:
input -> claim extraction -> evidence collection -> processing -> verification -> persistence -> report generation.
Mocks external network boundaries for Tavily and Gemini.
Includes an optional live-API smoke test disabled by default.
"""

import os
import json
import pytest
from typing import Optional
from unittest.mock import MagicMock, patch

from config import config
from src import VerificationStatus, ExtractedClaim, EvidenceItem
from src.claim_extractor import ClaimExtractor
from src.evidence_collector import EvidenceCollector
from src.evidence_processor import EvidenceProcessor
from src.verification_engine import VerificationEngine
from src.database import DatabaseManager
from src.report_generator import ReportGenerator


@pytest.fixture
def temp_db(tmp_path) -> DatabaseManager:
    """Provide a fresh isolated SQLite database for each test run."""
    db_file = tmp_path / "test_mindmesh.db"
    return DatabaseManager(db_path=str(db_file))


@pytest.fixture
def temp_outputs(tmp_path) -> ReportGenerator:
    """Provide an isolated report generator output directory."""
    out_dir = tmp_path / "test_outputs"
    return ReportGenerator(output_dir=str(out_dir))


def test_end_to_end_supporting_evidence_pipeline(temp_db: DatabaseManager, temp_outputs: ReportGenerator) -> None:
    """Test full pipeline when evidence corroborates the claim -> LIKELY RELIABLE."""
    raw_input = "James Webb Telescope identifies water vapor on habitable zone exoplanet."

    # 1. Claim Extraction
    extractor = ClaimExtractor()
    extracted = extractor.extract(raw_input)
    assert extracted.cleaned_claim != ""
    assert "James Webb" in extracted.cleaned_claim

    # 2. Mock Evidence Collection (Tavily)
    mock_tavily = MagicMock()
    mock_tavily.search.return_value = {
        "results": [
            {
                "title": "NASA JWST Observations Confirm Water Vapor",
                "url": "https://reuters.com/science/jwst-water-vapor",
                "content": "Astrophysicists confirmed verified water vapor absorption in the exoplanet atmosphere.",
                "score": 0.95,
                "published_date": "2024-05-10",
            },
            {
                "title": "AP News: Exoplanet Atmosphere Study",
                "url": "https://apnews.com/article/jwst-atmosphere-study",
                "content": "Independent astronomers verified the spectral transmission data accurately.",
                "score": 0.91,
                "published_date": "2024-05-11",
            },
        ]
    }

    collector = EvidenceCollector(tavily_api_key="tvly-test-key")
    with patch("tavily.TavilyClient", return_value=mock_tavily):
        collected = collector.collect_evidence(extracted)

    assert len(collected) == 2

    # 3. Evidence Processing
    processor = EvidenceProcessor()
    processed = processor.process_evidence(collected)
    assert len(processed.unique_items) == 2
    assert processed.independent_sources_count == 2

    # 4. Mock Gemini Verification Engine
    mock_gemini = MagicMock()
    mock_response = MagicMock()
    mock_response.text = json.dumps({
        "status": "LIKELY RELIABLE",
        "summary": "Corroborated by space agency press releases and independent analysis.",
        "reasoning": "Evidence ev_1 and ev_2 confirm the presence of atmospheric water vapor.",
        "supporting_evidence_ids": ["ev_1", "ev_2"],
        "contradicting_evidence_ids": [],
        "mixed_or_contextual_evidence_ids": [],
        "limitations": "Direct surface habitability remains unconfirmed.",
        "confidence": "High",
        "investigated_claim": extracted.cleaned_claim,
    })
    mock_gemini.models.generate_content.return_value = mock_response

    engine = VerificationEngine(processor=processor, gemini_client=mock_gemini)
    result = engine.assess_claim(extracted, processed.unique_items)

    assert result.status == VerificationStatus.LIKELY_RELIABLE
    assert result.confidence_score >= 0.8
    assert len(result.supporting_evidence_ids) == 2

    # 5. Database Persistence
    verif_id = temp_db.save_full_pipeline_audit(result)
    assert verif_id > 0
    recent = temp_db.list_recent_verifications(limit=1)
    assert len(recent) == 1
    assert recent[0]["status"] == VerificationStatus.LIKELY_RELIABLE.value

    # 6. Report Export
    report_file = temp_outputs.export_report(result, filename_stem="test_pipeline_run")
    assert report_file.exists()
    content = report_file.read_text(encoding="utf-8")
    assert "LIKELY RELIABLE" in content
    assert "https://reuters.com/science/jwst-water-vapor" in content


def test_end_to_end_contradicted_evidence_pipeline(temp_db: DatabaseManager, temp_outputs: ReportGenerator) -> None:
    """Test full pipeline when evidence debunks the claim -> POTENTIAL FAKE / MISINFORMATION."""
    raw_input = "Breaking: Drinking distilled kerosene completely cures lung infections in 24 hours."

    # 1. Extraction
    extractor = ClaimExtractor()
    extracted = extractor.extract(raw_input)

    # 2. Mock Tavily search returning debunking evidence
    mock_tavily = MagicMock()
    mock_tavily.search.return_value = {
        "results": [
            {
                "title": "Medical Fact Check: Kerosene Ingestion Hoax",
                "url": "https://snopes.com/fact-check/kerosene-cure-hoax",
                "content": "Rating: FALSE. Ingesting kerosene is extremely toxic and dangerous; claims of cure are a deadly hoax.",
                "score": 0.96,
                "published_date": "2024-01-20",
            },
            {
                "title": "WHO Warning on Toxic Household Substances",
                "url": "https://who.int/news/toxic-remedies-debunk",
                "content": "Health authorities debunk viral social media posts falsely promoting fuel consumption.",
                "score": 0.94,
                "published_date": "2024-01-22",
            },
        ]
    }

    collector = EvidenceCollector(tavily_api_key="tvly-test-key")
    with patch("tavily.TavilyClient", return_value=mock_tavily):
        collected = collector.collect_evidence(extracted)

    # 3. Processing
    processor = EvidenceProcessor()
    processed = processor.process_evidence(collected)

    # 4. Mock Gemini Verification Engine
    mock_gemini = MagicMock()
    mock_response = MagicMock()
    mock_response.text = json.dumps({
        "status": "POTENTIAL FAKE / MISINFORMATION",
        "summary": "Directly debunked by global health organizations and fact-checkers.",
        "reasoning": "Evidence ev_1 and ev_2 identify this as a hazardous medical falsehood.",
        "supporting_evidence_ids": [],
        "contradicting_evidence_ids": ["ev_1", "ev_2"],
        "mixed_or_contextual_evidence_ids": [],
        "limitations": "None. The scientific consensus is unanimous.",
        "confidence": "High",
        "investigated_claim": extracted.cleaned_claim,
    })
    mock_gemini.models.generate_content.return_value = mock_response

    engine = VerificationEngine(processor=processor, gemini_client=mock_gemini)
    result = engine.assess_claim(extracted, processed.unique_items)

    assert result.status == VerificationStatus.POTENTIAL_FAKE
    assert len(result.contradicting_evidence_ids) == 2

    # 5. Persistence & Export
    temp_db.save_full_pipeline_audit(result)
    report_file = temp_outputs.export_report(result)
    assert report_file.exists()
    assert "POTENTIAL FAKE / MISINFORMATION" in report_file.read_text(encoding="utf-8")


def test_end_to_end_insufficient_evidence_pipeline() -> None:
    """When Tavily returns 0 hits, pipeline MUST return NEEDS_VERIFICATION with high uncertainty."""
    raw_input = "Unknown obscure claim with zero internet footprint."

    extractor = ClaimExtractor()
    extracted = extractor.extract(raw_input)

    mock_tavily = MagicMock()
    mock_tavily.search.return_value = {"results": []}

    collector = EvidenceCollector(tavily_api_key="tvly-test-key")
    with patch("tavily.TavilyClient", return_value=mock_tavily):
        collected = collector.collect_evidence(extracted)

    assert len(collected) == 0

    engine = VerificationEngine()
    result = engine.assess_claim(extracted, collected)

    assert result.status == VerificationStatus.NEEDS_VERIFICATION
    assert result.uncertainty_score >= 0.8
    assert "No corroborating or refuting external evidence was found" in result.explanation


def test_end_to_end_search_failure_pipeline() -> None:
    """When search encounters a network failure, pipeline handles gracefully without crashing."""
    raw_input = "Viral claim during network outage."

    extractor = ClaimExtractor()
    extracted = extractor.extract(raw_input)

    collector = EvidenceCollector(tavily_api_key="tvly-test-key", max_retries=0)
    with patch("tavily.TavilyClient.search", side_effect=Exception("API gateway timeout 504")):
        res = collector.collect_with_status(extracted)

    assert res.status == "ERROR"
    assert len(res.items) == 0

    engine = VerificationEngine()
    result = engine.assess_claim(extracted, res.items)
    assert result.status == VerificationStatus.NEEDS_VERIFICATION


def test_end_to_end_ai_analysis_failure_pipeline() -> None:
    """When Gemini returns unparseable or error response, deterministic fallback takes over."""
    raw_input = "NASA confirms new exoplanet atmosphere."
    extractor = ClaimExtractor()
    extracted = extractor.extract(raw_input)

    evidence = [
        EvidenceItem(
            id="ev_1",
            source_name="reuters.com",
            url="https://reuters.com/news/1",
            snippet="NASA confirmed and verified new atmosphere observation.",
        )
    ]

    mock_gemini = MagicMock()
    mock_gemini.models.generate_content.side_effect = Exception("Model quota exceeded 429")

    engine = VerificationEngine(gemini_client=mock_gemini)
    result = engine.assess_claim(extracted, evidence)

    # Must fall back gracefully to heuristic keyword evaluation
    assert result.status == VerificationStatus.LIKELY_RELIABLE
    assert "Fallback" in result.limitations or "Baseline" in result.model_used


def test_end_to_end_citation_integrity_filtering() -> None:
    """Server-side citation filter must strip non-existent IDs hallucinated by LLM."""
    raw_input = "Space agency rover reaches destination."
    extracted = ClaimExtractor().extract(raw_input)

    evidence = [
        EvidenceItem(
            id="ev_1",
            source_name="bbc.com",
            url="https://bbc.com/news/space-1",
            snippet="Rover has arrived safely.",
        )
    ]

    mock_gemini = MagicMock()
    mock_response = MagicMock()
    mock_response.text = json.dumps({
        "status": "LIKELY RELIABLE",
        "summary": "Confirmed arrival.",
        "reasoning": "Source ev_1 and non-existent ev_invented_999 confirm this.",
        "supporting_evidence_ids": ["ev_1", "ev_invented_999"],
        "contradicting_evidence_ids": [],
        "mixed_or_contextual_evidence_ids": [],
        "limitations": "Standard caveats.",
        "confidence": "Medium",
        "investigated_claim": extracted.cleaned_claim,
    })
    mock_gemini.models.generate_content.return_value = mock_response

    engine = VerificationEngine(gemini_client=mock_gemini)
    result = engine.assess_claim(extracted, evidence)

    assert "ev_invented_999" not in result.supporting_evidence_ids
    assert result.supporting_evidence_ids == ["ev_1"]
    assert "Citation Integrity Audit" in result.limitations


def _is_valid_live_key(key: Optional[str]) -> bool:
    """Check that an API key is present, non-empty, and not a template placeholder."""
    if not key or not key.strip():
        return False
    stripped = key.strip()
    return not stripped.startswith("your_") and not stripped.startswith("<")


def _is_live_api_ready() -> bool:
    """Determine if live API testing is enabled with genuine non-placeholder keys."""
    if not os.getenv("RUN_LIVE_API_TESTS"):
        return False
    tavily_key = os.getenv("TAVILY_API_KEY") or config.tavily_api_key
    gemini_key = os.getenv("GEMINI_API_KEY") or config.gemini_api_key
    return bool(_is_valid_live_key(tavily_key) and _is_valid_live_key(gemini_key))


def test_live_smoke_test_skip_condition_rejects_placeholders() -> None:
    """Verify that placeholder credentials never enable live smoke tests."""
    assert _is_valid_live_key(None) is False
    assert _is_valid_live_key("") is False
    assert _is_valid_live_key("   ") is False
    assert _is_valid_live_key("your_tavily_api_key") is False
    assert _is_valid_live_key("your_gemini_api_key") is False
    assert _is_valid_live_key("tvly-real-credential-sample") is True


@pytest.mark.skipif(
    not _is_live_api_ready(),
    reason="Live API tests require RUN_LIVE_API_TESTS=1 and real, non-placeholder TAVILY_API_KEY and GEMINI_API_KEY.",
)
def test_manual_live_api_smoke_test() -> None:
    """Manual live smoke test querying real Tavily and Gemini endpoints (disabled by default)."""
    raw_input = "James Webb Space Telescope detects water vapor on exoplanet."
    extracted = ClaimExtractor().extract(raw_input)

    tavily_key = os.getenv("TAVILY_API_KEY") or config.tavily_api_key
    gemini_key = os.getenv("GEMINI_API_KEY") or config.gemini_api_key

    collector = EvidenceCollector(tavily_api_key=tavily_key)
    collected = collector.collect_evidence(extracted, max_results=3)

    assert len(collected) > 0, "Live Tavily query returned 0 results."

    processor = EvidenceProcessor()
    processed = processor.process_evidence(collected)

    engine = VerificationEngine(processor=processor, gemini_api_key=gemini_key)
    result = engine.assess_claim(extracted, processed.unique_items)

    assert result.status in [
        VerificationStatus.LIKELY_RELIABLE,
        VerificationStatus.NEEDS_VERIFICATION,
        VerificationStatus.POTENTIAL_FAKE,
    ]
    assert len(result.source_urls) > 0
    assert result.summary is not None

