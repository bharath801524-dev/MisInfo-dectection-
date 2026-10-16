"""
Unit tests for Mis-Info Verify Evidence Processor module.
"""

from src import EvidenceItem
from src.evidence_processor import EvidenceProcessor


def test_extract_domain() -> None:
    processor = EvidenceProcessor()
    assert processor.extract_domain("https://www.reuters.com/article/123") == "reuters.com"
    assert processor.extract_domain("http://bbc.co.uk/news") == "bbc.co.uk"
    assert processor.extract_domain("invalid-url") == ""


def test_estimate_source_credibility() -> None:
    processor = EvidenceProcessor()
    reuters_score = processor.estimate_source_credibility("https://www.reuters.com/news")
    unknown_score = processor.estimate_source_credibility("https://unverified-blog.xyz/post")

    assert reuters_score >= 0.9
    assert unknown_score == 0.5


def test_deduplicate_evidence() -> None:
    processor = EvidenceProcessor()
    items = [
        EvidenceItem(
            source_name="Reuters",
            url="https://reuters.com/news/1",
            snippet="Report A",
            reliability_score=0.9,
        ),
        EvidenceItem(
            source_name="Reuters Duplicate",
            url="https://reuters.com/news/1",
            snippet="Report A Duplicate",
            reliability_score=0.9,
        ),
        EvidenceItem(
            source_name="AP News",
            url="https://apnews.com/news/2",
            snippet="Report B",
            reliability_score=0.95,
        ),
    ]

    deduped = processor.deduplicate(items)
    assert len(deduped) == 2
    urls = [i.url for i in deduped]
    assert "https://reuters.com/news/1" in urls
    assert "https://apnews.com/news/2" in urls


def test_process_empty_evidence() -> None:
    processor = EvidenceProcessor()
    processed = processor.process_evidence([])

    assert len(processed.unique_items) == 0
    assert len(processed.unique_domains) == 0
    assert processed.domain_diversity_score == 0.0
    assert processed.average_source_reliability == 0.0


def test_domain_diversity_calculation() -> None:
    processor = EvidenceProcessor()
    items = [
        EvidenceItem(source_name="A", url="https://a.com/1", snippet="s1"),
        EvidenceItem(source_name="B", url="https://b.com/2", snippet="s2"),
        EvidenceItem(source_name="C", url="https://c.com/3", snippet="s3"),
    ]
    processed = processor.process_evidence(items)
    assert processed.domain_diversity_score == 3 / 5.0


def test_near_duplicate_snippet_deduplication() -> None:
    processor = EvidenceProcessor()
    # Different URLs but identical syndicated content
    items = [
        EvidenceItem(
            source_name="Affiliate 1",
            url="https://affiliate1.com/news",
            snippet="Scientists at CERN announce groundbreaking measurement of Higgs boson decay rates today in Geneva.",
        ),
        EvidenceItem(
            source_name="Affiliate 2",
            url="https://affiliate2.com/news",
            snippet="Scientists at CERN announce groundbreaking measurement of Higgs boson decay rates today in Geneva.",
        ),
    ]
    deduped = processor.deduplicate(items)
    assert len(deduped) == 1


def test_prepare_bundle_for_gemini() -> None:
    from src import ExtractedClaim
    processor = EvidenceProcessor()
    claim = ExtractedClaim(raw_input="Sample claim", cleaned_claim="Sample claim")
    items = [
        EvidenceItem(
            id="ev_1",
            source_name="reuters.com",
            title="Reuters Breaking News",
            url="https://reuters.com/news/1",
            snippet="Relevant excerpt for verification.",
            published_date="2024-05-01",
        )
    ]
    processed = processor.process_evidence(items)
    bundle = processor.prepare_bundle_for_gemini(claim, processed)

    assert len(bundle) == 1
    assert bundle[0]["id"] == "ev_1"
    assert bundle[0]["source"] == "reuters.com"
    assert bundle[0]["published_date"] == "2024-05-01"

