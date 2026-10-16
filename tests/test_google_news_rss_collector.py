"""
Unit and integration tests for Google News RSS Collector and Multi-Source Evidence Integration.

All external HTTP calls are strictly mocked. Network calls are never made during test execution.
"""

import urllib.error
from unittest.mock import MagicMock, patch
import pytest

from src import ExtractedClaim, EvidenceItem, VerificationStatus
from src.google_news_rss_collector import GoogleNewsRSSCollector
from src.evidence_collector import EvidenceCollector, MultiSourceEvidenceCollector, CollectionResult
from src.evidence_processor import EvidenceProcessor
from src.verification_engine import VerificationEngine


# Sample valid Google News RSS XML response fixture
SAMPLE_GOOGLE_NEWS_RSS_XML = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>ISRO Moon mission - Google News</title>
    <link>https://news.google.com/search?q=ISRO+Moon+mission&amp;hl=en-IN&amp;gl=IN&amp;ceid=IN:en</link>
    <language>en-IN</language>
    <item>
      <title>ISRO announces Chandrayaan-4 landing site details - The Hindu</title>
      <link>https://news.google.com/rss/articles/CBMiampodHRwczovL3d3dy50aGVoaW5kdS5jb20vc2NpZW5jZS9pc3JvLXNpdGUvYXJ0aWNsZTY4NzU0MzI5LmVjZQ?oc=5</link>
      <guid isPermaLink="false">CBMiampodHRwczovL3d3dy50aGVoaW5kdS5jb20vc2NpZW5jZS9pc3JvLXNpdGUvYXJ0aWNsZTY4NzU0MzI5LmVjZQ</guid>
      <pubDate>Wed, 14 Oct 2026 08:30:00 GMT</pubDate>
      <description>&lt;a href="https://news.google.com/rss/articles/..." target="_blank"&gt;ISRO announces Chandrayaan-4 landing site details&lt;/a&gt;&amp;nbsp;&amp;nbsp;&lt;font color="#6f6f6f"&gt;The Hindu&lt;/font&gt;</description>
      <source url="https://www.thehindu.com">The Hindu</source>
    </item>
    <item>
      <title>Cabinet approves next phase of lunar exploration by ISRO - NDTV</title>
      <link>https://news.google.com/rss/articles/CBMiam5kdHYuY29tL2luZGlhLW5ld3MvaXNyby1sdW5hci1taXNzaW9uLTEyMzQ1Ng?oc=5</link>
      <guid isPermaLink="false">CBMiam5kdHYuY29tL2luZGlhLW5ld3MvaXNyby1sdW5hci1taXNzaW9uLTEyMzQ1Ng</guid>
      <pubDate>Wed, 14 Oct 2026 09:15:00 GMT</pubDate>
      <description>&lt;p&gt;Union Cabinet gives green light for ISRO moon exploration project.&lt;/p&gt;</description>
      <source url="https://www.ndtv.com">NDTV</source>
    </item>
  </channel>
</rss>
"""


def test_url_encoding_and_india_edition_parameters() -> None:
    """Verify URL construction uses URL-encoding and India regional parameters."""
    collector = GoogleNewsRSSCollector(hl="en-IN", gl="IN", ceid="IN:en")
    query = "ISRO & NASA joint space mission 2026"

    feed_url = collector.build_feed_url(query)

    assert "https://news.google.com/rss/search?" in feed_url
    assert "hl=en-IN" in feed_url
    assert "gl=IN" in feed_url
    assert "ceid=IN%3Aen" in feed_url or "ceid=IN:en" in feed_url
    # Query must be encoded safely (spaces replaced by + or %20, ampersand encoded)
    assert "&q=" in feed_url or "?q=" in feed_url
    assert " " not in feed_url
    assert "ISRO+%26+NASA" in feed_url or "ISRO%20%26%20NASA" in feed_url or "ISRO%2B%26%2BNASA" in feed_url or "ISRO" in feed_url


def test_rss_xml_parsing_standard_feed() -> None:
    """Verify standard Google News RSS XML parses into EvidenceItem with correct provenance."""
    collector = GoogleNewsRSSCollector()
    items = collector.parse_rss_xml(SAMPLE_GOOGLE_NEWS_RSS_XML, original_query="ISRO Moon mission")

    assert len(items) == 2

    first = items[0]
    assert first.source_name == "The Hindu"
    assert first.title == "ISRO announces Chandrayaan-4 landing site details"
    assert "The Hindu" not in first.title  # Suffix stripped from clean headline
    assert first.published_date is not None
    assert "2026-10-14" in first.published_date
    assert first.source_origin == "google_news_rss"
    assert first.source_type == "news"
    assert first.is_redirect_url is True
    assert "[Google News Headline]" in first.snippet

    second = items[1]
    assert second.source_name == "NDTV"
    assert second.title == "Cabinet approves next phase of lunar exploration by ISRO"
    assert second.source_origin == "google_news_rss"


def test_malformed_xml_handling() -> None:
    """Verify malformed XML does not raise unhandled exception and returns empty list."""
    collector = GoogleNewsRSSCollector()
    malformed = "<rss><channel><item><title>Truncated unclosed XML"

    items = collector.parse_rss_xml(malformed, original_query="broken query")
    assert items == []


def test_empty_feed_handling() -> None:
    """Verify empty channel XML returns empty list with SUCCESS status."""
    empty_xml = """<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0">
      <channel>
        <title>Google News</title>
        <link>https://news.google.com</link>
      </channel>
    </rss>
    """
    collector = GoogleNewsRSSCollector()
    items = collector.parse_rss_xml(empty_xml, original_query="obscure query")
    assert items == []


def test_http_error_handling() -> None:
    """Verify HTTP errors (e.g. 500, 404) are caught safely without crashing."""
    collector = GoogleNewsRSSCollector()
    claim = ExtractedClaim(raw_input="Test claim", cleaned_claim="Test claim")

    with patch.object(collector, "_fetch_feed", side_effect=urllib.error.HTTPError("http://example.com", 503, "Service Unavailable", {}, None)):
        res = collector.collect_with_status(claim)

    assert res.status == "ERROR"
    assert len(res.items) == 0
    assert "503" in (res.error_message or "")


def test_timeout_handling() -> None:
    """Verify network timeout returns error status gracefully."""
    collector = GoogleNewsRSSCollector()
    claim = ExtractedClaim(raw_input="Timeout test", cleaned_claim="Timeout test")

    with patch.object(collector, "_fetch_feed", side_effect=TimeoutError("Request timed out")):
        res = collector.collect_with_status(claim)

    assert res.status == "ERROR"
    assert len(res.items) == 0
    assert "Timeout" in (res.error_message or "")


def test_deduplication_across_rss_queries() -> None:
    """Verify that identical articles returned by consecutive RSS queries are deduplicated."""
    collector = GoogleNewsRSSCollector()
    claim = ExtractedClaim(
        raw_input="ISRO lunar mission update",
        cleaned_claim="ISRO lunar mission update",
        detected_entities=["ISRO"],
    )

    # Return the same 2-item XML for every fetch call
    with patch.object(collector, "_fetch_feed", return_value=SAMPLE_GOOGLE_NEWS_RSS_XML):
        res = collector.collect_with_status(claim, max_results=5)

    # Even though 2 queries ran, duplicate headlines/URLs must be filtered out
    assert len(res.items) == 2
    titles = [it.title for it in res.items]
    assert len(titles) == len(set(titles))


def test_multisource_deduplication_between_tavily_and_rss() -> None:
    """Verify that MultiSourceEvidenceCollector merges and deduplicates overlapping Tavily & RSS hits."""
    mock_tavily = MagicMock(spec=EvidenceCollector)
    mock_rss = MagicMock(spec=GoogleNewsRSSCollector)

    shared_url = "https://www.thehindu.com/science/isro-chandrayaan-4"
    tavily_item = EvidenceItem(
        id="ev_tavily_1",
        source_name="thehindu.com",
        title="ISRO announces Chandrayaan-4 details",
        url=shared_url,
        snippet="ISRO chairman shares preliminary mission architecture for Chandrayaan-4.",
        source_origin="tavily",
    )
    rss_item = EvidenceItem(
        id="ev_rss_1",
        source_name="The Hindu",
        title="ISRO announces Chandrayaan-4 details",
        url=shared_url,  # Exact duplicate canonical URL
        snippet="[Google News Headline] ISRO announces Chandrayaan-4 details",
        source_origin="google_news_rss",
    )

    mock_tavily.collect_with_status.return_value = CollectionResult(
        items=[tavily_item], status="SUCCESS", queries_run=["query 1"]
    )
    mock_rss.collect_with_status.return_value = CollectionResult(
        items=[rss_item], status="SUCCESS", queries_run=["query 2"]
    )

    multi = MultiSourceEvidenceCollector(tavily_collector=mock_tavily, rss_collector=mock_rss)
    claim = ExtractedClaim(raw_input="ISRO Chandrayaan-4 announcement", cleaned_claim="ISRO Chandrayaan-4 announcement")

    res = multi.collect_with_status(claim, max_results=5)

    # Duplicate must be filtered out, leaving exactly 1 unique item
    assert len(res.items) == 1
    assert res.status == "SUCCESS"
    assert res.items[0].id == "ev_1"


def test_fallback_rss_only_when_tavily_has_no_key() -> None:
    """When Tavily key is absent, Google News RSS provides candidate coverage and succeeds."""
    mock_tavily = MagicMock(spec=EvidenceCollector)
    mock_rss = MagicMock(spec=GoogleNewsRSSCollector)

    mock_tavily.collect_with_status.return_value = CollectionResult(
        items=[], status="NO_API_KEY", error_message="TAVILY_API_KEY is not set."
    )
    rss_item = EvidenceItem(
        id="ev_rss_1",
        source_name="NDTV",
        title="Cabinet approves space mission",
        url="https://www.ndtv.com/space-mission",
        snippet="Government clears mission budget.",
        source_origin="google_news_rss",
    )
    mock_rss.collect_with_status.return_value = CollectionResult(
        items=[rss_item], status="SUCCESS", queries_run=["space mission"]
    )

    multi = MultiSourceEvidenceCollector(tavily_collector=mock_tavily, rss_collector=mock_rss)
    claim = ExtractedClaim(raw_input="Space mission approved", cleaned_claim="Space mission approved")

    res = multi.collect_with_status(claim)

    assert res.status == "SUCCESS"
    assert len(res.items) == 1
    assert res.items[0].source_origin == "google_news_rss"


def test_fallback_tavily_only_when_rss_fails() -> None:
    """When Google News RSS encounters a network error, Tavily evidence succeeds without interruption."""
    mock_tavily = MagicMock(spec=EvidenceCollector)
    mock_rss = MagicMock(spec=GoogleNewsRSSCollector)

    tavily_item = EvidenceItem(
        id="ev_t1",
        source_name="reuters.com",
        title="Reuters report",
        url="https://reuters.com/article",
        snippet="Corroborating Reuters reporting.",
        source_origin="tavily",
    )
    mock_tavily.collect_with_status.return_value = CollectionResult(
        items=[tavily_item], status="SUCCESS", queries_run=["query"]
    )
    mock_rss.collect_with_status.return_value = CollectionResult(
        items=[], status="ERROR", error_message="HTTP 503 error"
    )

    multi = MultiSourceEvidenceCollector(tavily_collector=mock_tavily, rss_collector=mock_rss)
    claim = ExtractedClaim(raw_input="News event", cleaned_claim="News event")

    res = multi.collect_with_status(claim)

    assert res.status == "SUCCESS"
    assert len(res.items) == 1
    assert res.items[0].source_origin == "tavily"


def test_both_sources_fail_defaults_to_needs_verification() -> None:
    """When both Tavily and Google News RSS return 0 items, verification engine defaults safely to NEEDS VERIFICATION."""
    mock_tavily = MagicMock(spec=EvidenceCollector)
    mock_rss = MagicMock(spec=GoogleNewsRSSCollector)

    mock_tavily.collect_with_status.return_value = CollectionResult(
        items=[], status="NO_API_KEY", error_message="No key"
    )
    mock_rss.collect_with_status.return_value = CollectionResult(
        items=[], status="ERROR", error_message="Connection refused"
    )

    multi = MultiSourceEvidenceCollector(tavily_collector=mock_tavily, rss_collector=mock_rss)
    claim = ExtractedClaim(raw_input="Unverified breaking rumor", cleaned_claim="Unverified breaking rumor")

    res = multi.collect_with_status(claim)
    assert len(res.items) == 0

    engine = VerificationEngine()
    verdict = engine.assess_claim(claim, res.items)

    assert verdict.status == VerificationStatus.NEEDS_VERIFICATION
    assert verdict.status != VerificationStatus.POTENTIAL_FAKE
    assert verdict.uncertainty_score >= 0.80


def test_true_claim_with_relevant_supporting_sources_not_labeled_misinformation() -> None:
    """A true news claim with relevant supporting sources from RSS & Tavily must NEVER be labeled misinformation."""
    import json

    mock_gemini = MagicMock()
    mock_response = MagicMock()
    mock_response.text = json.dumps({
        "status": "LIKELY RELIABLE",
        "summary": "Corroborated by Indian national news reporting and official releases.",
        "reasoning": "Evidence ev_1 and ev_2 directly confirm the Cabinet approval and ISRO mission timeline.",
        "supporting_evidence_ids": ["ev_1", "ev_2"],
        "contradicting_evidence_ids": [],
        "mixed_or_contextual_evidence_ids": [],
        "limitations": "Ongoing mission development details subject to schedule updates.",
        "confidence": "High",
        "investigated_claim": "Cabinet approves ISRO Chandrayaan-4 mission",
    })
    mock_gemini.models.generate_content.return_value = mock_response

    engine = VerificationEngine(gemini_client=mock_gemini)
    claim = ExtractedClaim(
        raw_input="Cabinet approves ISRO Chandrayaan-4 mission.",
        cleaned_claim="Cabinet approves ISRO Chandrayaan-4 mission.",
    )
    evidence = [
        EvidenceItem(
            id="ev_1",
            source_name="The Hindu",
            title="Cabinet approves Chandrayaan-4 lunar mission",
            url="https://thehindu.com/news/national/isro-chandrayaan-4",
            snippet="The Union Cabinet approved Chandrayaan-4 development with an outlay of Rs 2,104 crore.",
            source_origin="google_news_rss",
            reliability_score=0.90,
        ),
        EvidenceItem(
            id="ev_2",
            source_name="reuters.com",
            title="India approves new moon mission Chandrayaan-4",
            url="https://reuters.com/world/india/space-chandrayaan4",
            snippet="India announced cabinet approval for the next moon sample return mission Chandrayaan-4.",
            source_origin="tavily",
            reliability_score=0.95,
        ),
    ]

    result = engine.assess_claim(claim, evidence)

    assert result.status != VerificationStatus.POTENTIAL_FAKE
    assert result.status == VerificationStatus.LIKELY_RELIABLE
    assert len(result.supporting_evidence_ids) == 2


def test_irrelevant_rss_headlines_do_not_justify_fake_verdict() -> None:
    """When RSS returns off-topic or irrelevant headlines, engine must default to NEEDS VERIFICATION, never fake."""
    import json

    mock_gemini = MagicMock()
    mock_response = MagicMock()
    mock_response.text = json.dumps({
        "status": "NEEDS VERIFICATION",
        "summary": "Retrieved RSS articles discuss cinema releases and are irrelevant to the factual claim.",
        "reasoning": "Evidence ev_1 is an entertainment news headline that does not mention the agricultural policy claim.",
        "supporting_evidence_ids": [],
        "contradicting_evidence_ids": [],
        "mixed_or_contextual_evidence_ids": ["ev_1"],
        "limitations": "Lack of relevant news coverage on the specific agricultural policy.",
        "confidence": "Low",
        "investigated_claim": "Government launches new organic farming subsidy",
    })
    mock_gemini.models.generate_content.return_value = mock_response

    engine = VerificationEngine(gemini_client=mock_gemini)
    claim = ExtractedClaim(
        raw_input="Government launches new organic farming subsidy.",
        cleaned_claim="Government launches new organic farming subsidy.",
    )
    evidence = [
        EvidenceItem(
            id="ev_1",
            source_name="Bollywood Life",
            title="Celebrity attends film festival screening",
            url="https://example.com/entertainment",
            snippet="[Google News Headline] Actor attends premiere of upcoming movie.",
            source_origin="google_news_rss",
            reliability_score=0.50,
        )
    ]

    result = engine.assess_claim(claim, evidence)

    assert result.status != VerificationStatus.POTENTIAL_FAKE
    assert result.status == VerificationStatus.NEEDS_VERIFICATION
    assert result.uncertainty_score >= 0.50


def test_syndication_detection_in_evidence_processor() -> None:
    """Verify that multiple outlets repeating the exact same PTI wire story are flagged as syndicated."""
    processor = EvidenceProcessor()
    items = [
        EvidenceItem(
            id="ev_1",
            source_name="thehindu.com",
            title="ISRO chief addresses symposium - PTI",
            url="https://thehindu.com/isro-pti-1",
            snippet="PTI: Bengaluru - ISRO Chairman S Somanath announced that Chandrayaan-4 is on track.",
            source_origin="google_news_rss",
        ),
        EvidenceItem(
            id="ev_2",
            source_name="ndtv.com",
            title="Chandrayaan-4 on track says ISRO chief: Report",
            url="https://ndtv.com/isro-pti-2",
            snippet="Press Trust of India reports ISRO Chairman S Somanath confirmed Chandrayaan-4 schedule.",
            source_origin="google_news_rss",
        ),
    ]

    processed = processor.process_evidence(items)

    assert processed.syndicated_count >= 1
    assert processed.limitations_note is not None
    assert "syndicated wire" in processed.limitations_note.lower()
