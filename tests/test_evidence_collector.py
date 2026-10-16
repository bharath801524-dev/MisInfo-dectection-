"""
Unit tests for Mis-Info Verify Evidence Collector module.
Tests Tavily SDK integration, retry loops, error handling, and offline behavior.
"""

from unittest.mock import MagicMock, patch
import pytest

from src import ExtractedClaim
from src.evidence_collector import EvidenceCollector, CollectionResult


def test_missing_tavily_key_returns_safe_status() -> None:
    """When Tavily key is missing or empty, collector must return NO_API_KEY and never fabricate hits."""
    collector = EvidenceCollector(tavily_api_key=None)
    claim = ExtractedClaim(
        raw_input="Sample breaking news",
        cleaned_claim="Sample breaking news",
    )

    result = collector.collect_with_status(claim)
    assert result.status == "NO_API_KEY"
    assert len(result.items) == 0
    assert "TAVILY_API_KEY is not set" in result.error_message


def test_placeholder_tavily_key_treated_as_missing() -> None:
    collector = EvidenceCollector(tavily_api_key="your_tavily_api_key")
    claim = ExtractedClaim(
        raw_input="Sample news",
        cleaned_claim="Sample news",
    )

    result = collector.collect_with_status(claim)
    assert result.status == "NO_API_KEY"
    assert len(result.items) == 0


def test_has_valid_key_matrix() -> None:
    """Test key validation logic across missing, blank, placeholder, and valid-looking keys."""
    # Missing / None
    assert EvidenceCollector(tavily_api_key=None).has_valid_key() is False
    # Empty string
    assert EvidenceCollector(tavily_api_key="").has_valid_key() is False
    # Whitespace only
    assert EvidenceCollector(tavily_api_key="   ").has_valid_key() is False
    # Placeholder variations
    assert EvidenceCollector(tavily_api_key="your_tavily_api_key").has_valid_key() is False
    assert EvidenceCollector(tavily_api_key="your_key_here").has_valid_key() is False
    assert EvidenceCollector(tavily_api_key="your_api_key").has_valid_key() is False
    # Valid-looking keys (must not make any network calls during validation)
    assert EvidenceCollector(tavily_api_key="tvly-mock-valid-key-12345").has_valid_key() is True
    assert EvidenceCollector(tavily_api_key="tvly-test-sample-secret").has_valid_key() is True



def test_mocked_successful_tavily_response() -> None:
    """Verify that valid Tavily API responses normalize into structured EvidenceItems."""
    mock_tavily_client = MagicMock()
    mock_tavily_client.search.return_value = {
        "query": '"mars rover discovers water"',
        "results": [
            {
                "title": "NASA Rover Finds Liquid Water Evidence",
                "url": "https://reuters.com/science/mars-water-2024",
                "content": "Scientists confirm definitive traces of water molecules in Mars crater.",
                "score": 0.94,
                "published_date": "2024-04-15",
            },
            {
                "title": "AP: Mars Discovery Verified",
                "url": "https://apnews.com/article/mars-rover-analysis",
                "content": "Independent astronomers verified spectra data from the Mars rover.",
                "score": 0.88,
                "published_date": "2024-04-16",
            },
        ],
    }

    collector = EvidenceCollector(tavily_api_key="tvly-mock-test-key-12345")
    claim = ExtractedClaim(
        raw_input="Mars rover discovers water in crater",
        cleaned_claim="Mars rover discovers water in crater",
    )

    with patch("tavily.TavilyClient", return_value=mock_tavily_client):
        result = collector.collect_with_status(claim, max_results=5)

    assert result.status == "SUCCESS"
    assert len(result.items) == 2

    first = result.items[0]
    assert first.id == "ev_1"
    assert first.title == "NASA Rover Finds Liquid Water Evidence"
    assert first.source_name == "reuters.com"
    assert first.url == "https://reuters.com/science/mars-water-2024"
    assert "Scientists confirm definitive traces" in first.snippet
    assert first.published_date == "2024-04-15"
    assert first.relevance_score == 0.94


def test_tavily_deduplicates_duplicate_urls() -> None:
    """If multiple queries return the same canonical URL, collector must deduplicate."""
    mock_tavily_client = MagicMock()
    # Both calls return the exact same URL
    mock_tavily_client.search.return_value = {
        "results": [
            {
                "title": "Article A",
                "url": "https://example.com/story-1",
                "content": "Excerpt content",
                "score": 0.9,
            },
            {
                "title": "Article A Duplicate",
                "url": "https://example.com/story-1",
                "content": "Excerpt content duplicate",
                "score": 0.85,
            },
        ]
    }

    collector = EvidenceCollector(tavily_api_key="tvly-mock-key")
    claim = ExtractedClaim(raw_input="Example story", cleaned_claim="Example story")

    with patch("tavily.TavilyClient", return_value=mock_tavily_client):
        result = collector.collect_with_status(claim, max_results=5)

    assert len(result.items) == 1
    assert result.items[0].url == "https://example.com/story-1"


def test_tavily_bounded_retry_and_error_handling() -> None:
    """Verify that network errors trigger bounded retries and report safe error status."""
    mock_tavily_client = MagicMock()
    mock_tavily_client.search.side_effect = Exception("Connection reset by peer")

    collector = EvidenceCollector(tavily_api_key="tvly-mock-key", max_retries=1)
    claim = ExtractedClaim(raw_input="Failing query", cleaned_claim="Failing query")

    with patch("tavily.TavilyClient", return_value=mock_tavily_client):
        result = collector.collect_with_status(claim)

    assert result.status == "ERROR"
    assert len(result.items) == 0
    assert "Connection reset by peer" in result.error_message
    # max_retries=1 means initial try + 1 retry = 2 calls per query
    assert mock_tavily_client.search.call_count >= 2


def test_tavily_empty_results_handling() -> None:
    """When Tavily returns 0 search results, collector handles it cleanly."""
    mock_tavily_client = MagicMock()
    mock_tavily_client.search.return_value = {"results": []}

    collector = EvidenceCollector(tavily_api_key="tvly-mock-key")
    claim = ExtractedClaim(raw_input="Obscure nonexistent event", cleaned_claim="Obscure nonexistent event")

    with patch("tavily.TavilyClient", return_value=mock_tavily_client):
        result = collector.collect_with_status(claim)

    assert len(result.items) == 0
    assert result.total_found == 0
