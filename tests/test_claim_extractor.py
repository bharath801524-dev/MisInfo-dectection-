"""
Unit tests for Mis-Info Verify Claim Extractor module.
"""

import pytest
from src.claim_extractor import ClaimExtractor, ClaimExtractorError


def test_extract_standard_claim() -> None:
    extractor = ClaimExtractor()
    raw_text = "RT @breaking: Scientists at NASA discover signs of ancient water on Mars!"
    claim = extractor.extract(raw_text)

    assert claim.raw_input == raw_text.strip()
    assert "Scientists at NASA discover signs" in claim.cleaned_claim
    assert not claim.cleaned_claim.lower().startswith("rt @breaking:")
    assert "NASA" in claim.detected_entities or "Mars" in claim.detected_entities
    assert claim.source_url is None


def test_extract_url_input() -> None:
    extractor = ClaimExtractor()
    url = "https://www.reuters.com/world/science-breakthrough"
    claim = extractor.extract(url)

    assert claim.source_url == url
    assert extractor.is_url(url) is True
    assert "reuters.com" in claim.cleaned_claim
    assert "www.reuters.com" in claim.detected_entities


def test_extract_empty_input_raises_error() -> None:
    extractor = ClaimExtractor()

    with pytest.raises(ClaimExtractorError):
        extractor.extract("")

    with pytest.raises(ClaimExtractorError):
        extractor.extract("   \n\t  ")


def test_clean_text_spacing() -> None:
    extractor = ClaimExtractor()
    messy_text = "Headline   with    excessive \t spaces  and \n newlines. "
    cleaned = extractor.clean_text(messy_text)
    assert cleaned == "Headline with excessive spaces and newlines."


def test_url_safety_blocks_localhost_and_private_ips() -> None:
    extractor = ClaimExtractor()

    # Disallow localhost
    is_safe, reason = extractor.validate_url_safety("http://localhost:8080/admin")
    assert is_safe is False
    assert "restricted" in reason.lower()

    # Disallow loopback 127.0.0.1
    is_safe, reason = extractor.validate_url_safety("http://127.0.0.1:8000/secret")
    assert is_safe is False

    # Disallow unsupported schemes like ftp
    is_safe, reason = extractor.validate_url_safety("ftp://files.example.com/data")
    assert is_safe is False
    assert "unsupported url scheme" in reason.lower()


def test_extract_dates_and_locations() -> None:
    extractor = ClaimExtractor()
    text = "In Paris yesterday, leaders met in 2024 to discuss the summit in Tokyo."
    dates = extractor.extract_dates(text)
    locations = extractor.extract_locations(text)

    assert "2024" in dates or "Yesterday" in dates
    assert "Paris" in locations or "Tokyo" in locations


def test_segment_claims_multiple_propositions() -> None:
    extractor = ClaimExtractor()
    text = "The lunar mission has launched successfully today. Ground control established direct communication. Solar arrays have deployed."
    primary, secondaries = extractor.segment_claims(text)

    assert "lunar mission has launched" in primary.lower()
    assert len(secondaries) >= 1

