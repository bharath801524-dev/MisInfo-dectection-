"""
Mis-Info Verify - Google News RSS Collector Module.

Retrieves candidate news articles from Google News RSS feeds with Indian regional focus
(hl=en-IN, gl=IN, ceid=IN:en). Parses headlines, publisher provenance, publication dates,
and feed metadata using Python standard-library XML parsing without external browser dependencies.
"""

import re
import html
import logging
import urllib.parse
import urllib.request
import urllib.error
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from typing import List, Optional, Set, Dict, Any
from urllib.parse import urlparse

from config import config
from src import ExtractedClaim, EvidenceItem
from src.evidence_collector import CollectionResult

logger = logging.getLogger("mindmesh.google_news_rss_collector")


class GoogleNewsRSSCollectorError(Exception):
    """Exception raised when Google News RSS collection encounters an unrecoverable failure."""
    pass


class GoogleNewsRSSCollector:
    """Collects news evidence items via Google News RSS feeds.

    Features:
    - Bounded execution with configurable network timeout.
    - Safe query URL-encoding with regional parameters (hl=en-IN, gl=IN, ceid=IN:en).
    - Standard library XML parsing with malformed XML resilience.
    - Publisher identification and original source URL extraction from RSS metadata.
    - Explicit redirect URL tagging for news.google.com links.
    - Content snippet labeling ensuring headlines are never treated as full article bodies.
    - Deduplication across multiple query runs.
    """

    def __init__(
        self,
        hl: Optional[str] = None,
        gl: Optional[str] = None,
        ceid: Optional[str] = None,
        timeout: Optional[int] = None,
        enabled: Optional[bool] = None,
    ) -> None:
        self.hl = hl or getattr(config, "google_news_rss_hl", "en-IN")
        self.gl = gl or getattr(config, "google_news_rss_gl", "IN")
        self.ceid = ceid or getattr(config, "google_news_rss_ceid", "IN:en")
        self.timeout = timeout if timeout is not None else getattr(config, "google_news_rss_timeout", 8)
        self.enabled = enabled if enabled is not None else getattr(config, "google_news_rss_enabled", True)
        logger.debug(
            "GoogleNewsRSSCollector initialized (edition: %s-%s, timeout: %ss, enabled: %s)",
            self.hl,
            self.gl,
            self.timeout,
            self.enabled,
        )

    def is_enabled(self) -> bool:
        """Check if RSS retrieval is enabled in configuration."""
        return bool(self.enabled)

    def build_feed_url(self, query: str) -> str:
        """Safely construct a Google News RSS search URL with query parameters."""
        encoded_query = urllib.parse.quote_plus(query.strip())
        return (
            f"https://news.google.com/rss/search?"
            f"q={encoded_query}&hl={self.hl}&gl={self.gl}&ceid={self.ceid}"
        )

    def generate_search_queries(self, claim: ExtractedClaim) -> List[str]:
        """Generate concise, high-yield search queries from an extracted claim and its entities.

        Args:
            claim: ExtractedClaim object.

        Returns:
            List of 1 to 3 targeted search query strings.
        """
        raw_text = claim.cleaned_claim.strip()

        # Remove punctuation for cleaner keyword search
        clean_words = [w for w in re.findall(r"\b[\w-]+\b", raw_text) if len(w) > 1]
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "and", "or", "in", "on", "at",
            "to", "for", "with", "of", "that", "this", "from", "by", "has", "have", "had",
            "about", "into", "over", "after", "it", "its", "news", "viral", "post", "claim",
        }
        filtered_words = [w for w in clean_words if w.lower() not in stopwords]

        queries = []

        # Use getattr to gracefully handle stale Pydantic model instances that
        # may lack newer fields when Streamlit hot-reloads cached modules.
        detected_entities: List[str] = getattr(claim, "detected_entities", []) or []
        detected_locations: List[str] = getattr(claim, "detected_locations", []) or []

        # Query 1: Exact entities if available, combined with salient claim keywords
        if detected_entities:
            entity_str = " ".join(detected_entities[:3])
            remaining = [w for w in filtered_words if w.lower() not in entity_str.lower()]
            query_1 = f"{entity_str} {' '.join(remaining[:4])}".strip()
            if query_1:
                queries.append(query_1)

        # Query 2: Core proposition (concise 6 to 8 key words)
        query_2 = " ".join(filtered_words[:7]).strip()
        if query_2 and query_2 not in queries:
            queries.append(query_2)

        # Query 3: Entity + location marker if available
        if detected_locations and detected_entities:
            query_3 = f"{' '.join(detected_entities[:2])} {' '.join(detected_locations[:1])}"
            if query_3 not in queries:
                queries.append(query_3)

        # Fallback to truncated raw claim if no keywords passed filtering
        if not queries:
            queries.append(raw_text[:80].strip())

        logger.debug("Generated %d RSS search queries: %s", len(queries), queries)
        return queries[:3]

    def _clean_html(self, raw_html: str) -> str:
        """Strip HTML tags and unescape character entities from RSS description fields."""
        if not raw_html:
            return ""
        # Remove HTML tags
        clean_text = re.sub(r"<[^>]+>", " ", raw_html)
        # Unescape HTML entities (e.g., &amp;, &quot;)
        clean_text = html.unescape(clean_text)
        # Collapse whitespace
        return re.sub(r"\s+", " ", clean_text).strip()

    def _normalize_date(self, raw_date_str: Optional[str]) -> Optional[str]:
        """Convert RFC 2822 / 822 RSS publication date into normalized ISO/readable date string."""
        if not raw_date_str:
            return None
        try:
            dt = parsedate_to_datetime(raw_date_str.strip())
            return dt.strftime("%Y-%m-%d %H:%M:%S UTC")
        except Exception:
            return raw_date_str.strip()

    def _extract_domain(self, url: str) -> str:
        """Extract clean root hostname from a URL."""
        if not url:
            return ""
        try:
            netloc = urlparse(url).netloc.lower()
            if netloc.startswith("www."):
                netloc = netloc[4:]
            return netloc
        except Exception:
            return ""

    def parse_rss_xml(self, xml_content: str | bytes, original_query: str) -> List[EvidenceItem]:
        """Parse raw RSS XML bytes or string into standardized EvidenceItem models.

        Args:
            xml_content: Raw XML response body.
            original_query: Search query string that produced this feed.

        Returns:
            List of parsed EvidenceItem objects.
        """
        if not xml_content:
            return []

        if isinstance(xml_content, str):
            xml_bytes = xml_content.encode("utf-8")
        else:
            xml_bytes = xml_content

        try:
            root = ET.fromstring(xml_bytes)
        except ET.ParseError as e:
            logger.warning("Malformed RSS XML received for query '%s': %s", original_query[:40], e)
            return []

        channel = root.find("channel")
        if channel is None:
            logger.warning("No <channel> found in RSS XML.")
            return []

        items: List[EvidenceItem] = []

        for item_elem in channel.findall("item"):
            # 1. Headline / Title
            title_elem = item_elem.find("title")
            raw_title = title_elem.text.strip() if (title_elem is not None and title_elem.text) else "Untitled Article"

            # 2. Publisher name & original publisher URL from <source> tag
            source_elem = item_elem.find("source")
            publisher_name = ""
            publisher_site_url = ""
            if source_elem is not None:
                publisher_name = source_elem.text.strip() if source_elem.text else ""
                publisher_site_url = source_elem.get("url", "").strip()

            # If publisher_name is absent in <source>, infer from title suffix ("Headline - Publisher")
            clean_headline = raw_title
            if not publisher_name and " - " in raw_title:
                parts = raw_title.rsplit(" - ", 1)
                clean_headline = parts[0].strip()
                publisher_name = parts[1].strip()
            elif publisher_name and raw_title.endswith(f" - {publisher_name}"):
                clean_headline = raw_title[:-len(f" - {publisher_name}")].strip()

            if not publisher_name:
                publisher_name = self._extract_domain(publisher_site_url) or "Google News Source"

            # 3. Article Link & Redirect Tagging
            link_elem = item_elem.find("link")
            raw_link = link_elem.text.strip() if (link_elem is not None and link_elem.text) else ""
            if not raw_link and publisher_site_url:
                raw_link = publisher_site_url

            is_redirect = "news.google.com" in raw_link.lower()

            # 4. Publication Date
            pub_date_elem = item_elem.find("pubDate")
            raw_pub_date = pub_date_elem.text.strip() if (pub_date_elem is not None and pub_date_elem.text) else None
            normalized_date = self._normalize_date(raw_pub_date)

            # 5. Description / Snippet
            desc_elem = item_elem.find("description")
            raw_desc = desc_elem.text if (desc_elem is not None and desc_elem.text) else ""
            clean_desc = self._clean_html(raw_desc)

            # Never treat RSS description/headline alone as full article text.
            # Label clearly as an RSS headline excerpt.
            if clean_desc and clean_desc.lower() != clean_headline.lower():
                snippet = f"[Google News Headline] {clean_headline} — Summary excerpt: {clean_desc}"
            else:
                snippet = f"[Google News Headline] {clean_headline} (Reported by {publisher_name})"

            # Construct EvidenceItem
            evidence = EvidenceItem(
                source_name=publisher_name,
                url=raw_link or publisher_site_url or "https://news.google.com",
                title=clean_headline,
                snippet=snippet,
                published_date=normalized_date,
                original_query=original_query,
                relevance_score=0.75,
                reliability_score=0.70,  # Elevated if accredited publisher, calibrated in evidence_processor
                source_type="news",
                source_origin="google_news_rss",
                is_redirect_url=is_redirect,
            )
            items.append(evidence)

        return items

    def _fetch_feed(self, url: str) -> str:
        """Fetch raw XML content from a URL with bounded timeout and safe headers."""
        headers = {
            "User-Agent": "MindMeshVerify/0.2.0 (compatible; news-verification-bot/1.0)",
            "Accept": "application/rss+xml, application/xml, text/xml;q=0.9, */*;q=0.8",
        }
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=self.timeout) as response:
            content_bytes = response.read()
            charset = response.headers.get_content_charset() or "utf-8"
            return content_bytes.decode(charset, errors="replace")

    def collect_with_status(self, claim: ExtractedClaim, max_results: int = 5) -> CollectionResult:
        """Execute Google News RSS collection across generated queries.

        Args:
            claim: ExtractedClaim object.
            max_results: Maximum unique evidence items to return.

        Returns:
            CollectionResult containing items, status, queries run, and any error message.
        """
        if not self.is_enabled():
            logger.info("Google News RSS collector is disabled in configuration.")
            return CollectionResult(
                items=[],
                status="NO_API_KEY",
                error_message="Google News RSS collection is disabled.",
            )

        queries = self.generate_search_queries(claim)
        collected_items: List[EvidenceItem] = []
        queries_run: List[str] = []
        seen_titles: Set[str] = set()
        seen_urls: Set[str] = set()
        last_error: Optional[str] = None

        for query in queries[:2]:  # Top 2 queries to balance speed and coverage
            queries_run.append(query)
            feed_url = self.build_feed_url(query)
            logger.debug("Requesting Google News RSS feed: %s", feed_url)

            try:
                raw_xml = self._fetch_feed(feed_url)
                parsed_items = self.parse_rss_xml(raw_xml, original_query=query)

                for item in parsed_items:
                    # Deduplicate across queries by normalized title and URL
                    norm_title = re.sub(r"[^\w\s]", "", (item.title or "").lower()).strip()
                    norm_url = item.url.lower().strip()

                    if norm_title and norm_title in seen_titles:
                        continue
                    if norm_url and norm_url in seen_urls:
                        continue

                    if norm_title:
                        seen_titles.add(norm_title)
                    if norm_url:
                        seen_urls.add(norm_url)

                    # Stable evidence ID assignment
                    item.id = f"ev_rss_{len(collected_items) + 1}"
                    collected_items.append(item)

                    if len(collected_items) >= max_results:
                        break

            except (urllib.error.HTTPError, urllib.error.URLError) as e:
                last_error = f"HTTP error during RSS fetch: {e}"
                logger.warning("RSS request failed for query '%s': %s", query[:40], e)
            except (TimeoutError, TimeoutException) if "TimeoutException" in globals() else (TimeoutError, OSError) as e:
                last_error = f"Timeout during RSS fetch: {e}"
                logger.warning("RSS request timed out for query '%s': %s", query[:40], e)
            except Exception as e:
                last_error = f"Unexpected error during RSS collection: {e}"
                logger.error("Unexpected RSS error: %s", e)

            if len(collected_items) >= max_results:
                break

        status = "SUCCESS" if collected_items else ("PARTIAL" if queries_run and not last_error else "ERROR")
        if not collected_items and not last_error:
            status = "SUCCESS"  # Empty feed is a successful retrieval of 0 items

        return CollectionResult(
            items=collected_items,
            status=status,
            total_found=len(collected_items),
            queries_run=queries_run,
            error_message=last_error,
        )

    def collect_evidence(self, claim: ExtractedClaim, max_results: int = 5) -> List[EvidenceItem]:
        """Convenience collection method returning a list of EvidenceItem objects."""
        return self.collect_with_status(claim, max_results=max_results).items
