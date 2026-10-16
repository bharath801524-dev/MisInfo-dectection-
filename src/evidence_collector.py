"""
Mis-Info Verify - Evidence Collector Module.

Responsible for querying external fact-checking sources, news agencies, and search engines
via the Tavily Search API. Normalizes results into structured EvidenceItem objects with
stable IDs, deduplication, retry handling, and complete audit metadata.
"""

import time
import logging
import re
from urllib.parse import urlparse
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

from config import config
from src import ExtractedClaim, EvidenceItem

logger = logging.getLogger("mindmesh.evidence_collector")


class CollectionResult(BaseModel):
    """Result of an evidence gathering attempt with explicit status and telemetry."""
    items: List[EvidenceItem] = Field(default_factory=list)
    status: str = Field(..., description="'SUCCESS', 'PARTIAL', 'NO_API_KEY', or 'ERROR'")
    total_found: int = Field(default=0)
    queries_run: List[str] = Field(default_factory=list)
    error_message: Optional[str] = Field(default=None)


class EvidenceCollectorError(Exception):
    """Custom exception raised when evidence collection encounters an unrecoverable failure."""
    pass


_UNSET = object()


class EvidenceCollector:
    """Manages real external evidence retrieval using the Tavily Python SDK.

    Includes bounded exponential retries, URL deduplication, sanitized error reporting,
    and fallback handling when API credentials are absent.
    """

    def __init__(
        self,
        tavily_api_key: Optional[str] = _UNSET,  # type: ignore
        serpapi_api_key: Optional[str] = _UNSET,  # type: ignore
        max_retries: int = 2,
    ) -> None:
        # Use provided keys or fall back to centralized config
        if tavily_api_key is _UNSET:
            self.tavily_key = config.tavily_api_key
        else:
            self.tavily_key = tavily_api_key

        if serpapi_api_key is _UNSET:
            self.serpapi_key = config.serpapi_api_key
        else:
            self.serpapi_key = serpapi_api_key

        self.max_retries = max_retries

        if not self.has_valid_key():
            logger.info("No valid Tavily API key configured. Operating in offline/mock mode.")
        else:
            logger.info("Tavily API credentials detected. Live evidence retrieval enabled.")

    def has_valid_key(self) -> bool:
        """Verify if a functional Tavily key is present (non-empty and not a placeholder)."""
        if not self.tavily_key:
            return False
        key = self.tavily_key.strip()
        return bool(key and not key.startswith("your_"))

    def generate_search_queries(self, claim: ExtractedClaim) -> List[str]:
        """Generate targeted search queries from an extracted claim to maximize fact-check hits.

        Args:
            claim: ExtractedClaim object.

        Returns:
            List of search query strings suitable for search engines or fact-checking APIs.
        """
        base_claim = claim.cleaned_claim.strip()
        # Truncate extremely long inputs for search engine limits
        truncated_claim = base_claim[:150].strip()

        queries = [
            f'"{truncated_claim}"',
            f"{truncated_claim} fact check",
            f"{truncated_claim} news reporting",
        ]

        if claim.detected_entities:
            entities_str = " ".join(claim.detected_entities[:3])
            queries.append(f"{entities_str} news report")

        logger.debug("Generated %d search queries for claim.", len(queries))
        return queries

    def _extract_domain(self, url: str) -> str:
        """Extract clean hostname from URL for publisher identification."""
        try:
            netloc = urlparse(url).netloc.lower()
            if netloc.startswith("www."):
                netloc = netloc[4:]
            return netloc or "Web Source"
        except Exception:
            return "Web Source"

    def _query_tavily_with_retry(self, client: Any, query: str, max_results: int) -> Dict[str, Any]:
        """Execute a Tavily search call with bounded retry for transient network errors."""
        last_exception = None
        for attempt in range(self.max_retries + 1):
            try:
                logger.debug("Executing Tavily search for query '%s' (attempt %d)", query[:50], attempt + 1)
                response = client.search(
                    query=query,
                    max_results=max_results,
                    search_depth="basic",
                    include_answer=False,
                    include_raw_content=False,
                    include_images=True,
                )
                return response if isinstance(response, dict) else {}
            except Exception as e:
                last_exception = e
                err_str = str(e).lower()
                # If invalid key or unauthorized, do not retry
                if "unauthorized" in err_str or "invalid api key" in err_str or "401" in err_str:
                    logger.error("Tavily authentication failure: %s", e)
                    raise EvidenceCollectorError(f"Tavily authentication error: {e}")

                if attempt < self.max_retries:
                    wait_time = 1.0 * (2 ** attempt)
                    logger.warning("Tavily query failed (%s). Retrying in %.1fs...", e, wait_time)
                    time.sleep(wait_time)
                else:
                    logger.error("Tavily query failed after %d retries: %s", self.max_retries, e)

        raise EvidenceCollectorError(f"Tavily search failed after {self.max_retries} retries: {last_exception}")

    def collect_with_status(self, claim: ExtractedClaim, max_results: int = 5) -> CollectionResult:
        """Retrieve external evidence items using the configured Tavily API key.

        Returns an explicit CollectionResult detailing outcome, queries, items, and any errors.
        """
        if not self.has_valid_key():
            logger.warning("Search API key (TAVILY_API_KEY) is not configured. Returning NO_API_KEY status.")
            return CollectionResult(
                items=[],
                status="NO_API_KEY",
                error_message="TAVILY_API_KEY is not set. Add your key to .env to enable live search.",
            )

        try:
            from tavily import TavilyClient
            client = TavilyClient(api_key=self.tavily_key)
        except ImportError:
            logger.error("tavily-python package is not installed.")
            return CollectionResult(
                items=[],
                status="ERROR",
                error_message="tavily-python package is not installed.",
            )

        queries = self.generate_search_queries(claim)
        seen_urls = set()
        collected_items: List[EvidenceItem] = []
        queries_run = []
        error_msg = None

        # Execute top 2 targeted queries to balance latency and comprehensive coverage
        for q in queries[:2]:
            queries_run.append(q)
            try:
                response = self._query_tavily_with_retry(client, q, max_results=max_results)
                raw_results = response.get("results", [])

                for idx, r in enumerate(raw_results, start=1):
                    url = r.get("url", "").strip()
                    if not url or url.lower() in seen_urls:
                        continue

                    seen_urls.add(url.lower())
                    source_name = self._extract_domain(url)
                    snippet = r.get("content", "").strip()
                    title = r.get("title", "").strip()
                    pub_date = r.get("published_date")
                    score = r.get("score")
                    rel_score = float(score) if score is not None else 0.5

                    # Extract real news/article image if returned by search
                    item_images = r.get("images", []) or []
                    chosen_image = None
                    if isinstance(item_images, list):
                        for img in item_images:
                            if isinstance(img, str) and img.startswith(("http://", "https://")):
                                chosen_image = img
                                break
                    if not chosen_image:
                        gen_images = response.get("images", []) or []
                        if idx - 1 < len(gen_images):
                            cand = gen_images[idx - 1]
                            if isinstance(cand, str) and cand.startswith(("http://", "https://")):
                                chosen_image = cand

                    # Stable evidence ID
                    ev_id = f"ev_{len(collected_items) + 1}"

                    item = EvidenceItem(
                        id=ev_id,
                        source_name=source_name,
                        title=title if title else source_name,
                        url=url,
                        snippet=snippet if snippet else "No excerpt provided.",
                        published_date=str(pub_date) if pub_date else None,
                        original_query=q,
                        relevance_score=min(max(rel_score, 0.0), 1.0),
                        reliability_score=0.5,  # Processed/scored further in evidence_processor
                        source_type="fact_check" if "fact" in url or "fact" in snippet.lower() else "web",
                        source_origin="tavily",
                        image_url=chosen_image,
                    )
                    collected_items.append(item)

                    if len(collected_items) >= max_results:
                        break

            except EvidenceCollectorError as e:
                error_msg = str(e)
                logger.warning("Error running query '%s': %s", q, e)
                # Continue to next query if possible

            if len(collected_items) >= max_results:
                break

        status = "SUCCESS" if collected_items else ("PARTIAL" if queries_run and not error_msg else "ERROR")
        if not collected_items and error_msg:
            status = "ERROR"

        return CollectionResult(
            items=collected_items,
            status=status,
            total_found=len(collected_items),
            queries_run=queries_run,
            error_message=error_msg,
        )

    def collect_evidence(self, claim: ExtractedClaim, max_results: int = 5) -> List[EvidenceItem]:
        """Backward-compatible collection method returning a list of EvidenceItem objects.

        If external APIs are not configured or fail, returns an empty list without fabricating results.
        """
        result = self.collect_with_status(claim, max_results=max_results)
        return result.items


class MultiSourceEvidenceCollector:
    """Orchestrates evidence gathering across multiple independent collectors:
    1. Tavily Search API (broad web search, fact-check registries)
    2. Google News RSS (regional/national news coverage with Indian edition focus)

    Features:
    - Merges results while preserving exact source provenance ('tavily' vs 'google_news_rss').
    - Cross-source deduplication by canonical URL and normalized headlines.
    - Graceful fallback: If Tavily is unconfigured or fails, RSS provides candidate coverage.
      If RSS fails or is disabled, Tavily provides evidence.
      If both fail or return empty, returns safe status leading to NEEDS VERIFICATION.
    """

    def __init__(
        self,
        tavily_collector: Optional[EvidenceCollector] = None,
        rss_collector: Optional[Any] = None,
    ) -> None:
        self.tavily_collector = tavily_collector or EvidenceCollector()
        if rss_collector is not None:
            self.rss_collector = rss_collector
        else:
            from src.google_news_rss_collector import GoogleNewsRSSCollector
            self.rss_collector = GoogleNewsRSSCollector()

    def collect_with_status(
        self,
        claim: ExtractedClaim,
        max_results: int = 8,
    ) -> CollectionResult:
        """Run both collectors, merge items, preserve provenance, and deduplicate."""
        all_items: List[EvidenceItem] = []
        queries_run: List[str] = []
        errors: List[str] = []

        # 1. Collect from Tavily
        tavily_res = self.tavily_collector.collect_with_status(claim, max_results=max_results)
        queries_run.extend(tavily_res.queries_run)
        if tavily_res.items:
            for it in tavily_res.items:
                it.source_origin = "tavily"
                all_items.append(it)
        elif tavily_res.status == "ERROR" and tavily_res.error_message:
            errors.append(f"Tavily: {tavily_res.error_message}")

        # 2. Collect from Google News RSS
        rss_res = self.rss_collector.collect_with_status(claim, max_results=max_results)
        queries_run.extend(rss_res.queries_run)
        if rss_res.items:
            for it in rss_res.items:
                it.source_origin = "google_news_rss"
                all_items.append(it)
        elif rss_res.status == "ERROR" and rss_res.error_message:
            errors.append(f"Google News RSS: {rss_res.error_message}")

        # 3. Cross-source Deduplication (URL and normalized headline/snippet)
        seen_urls = set()
        seen_snippets = set()
        deduped: List[EvidenceItem] = []

        for item in all_items:
            clean_url = item.url.strip().lower()
            norm_text = re.sub(r"[^\w\s]", "", (item.title or item.snippet).lower())[:60].strip()

            if clean_url and clean_url in seen_urls:
                continue
            if norm_text and len(norm_text) > 15 and norm_text in seen_snippets:
                continue

            if clean_url:
                seen_urls.add(clean_url)
            if norm_text:
                seen_snippets.add(norm_text)

            deduped.append(item)

        # 4. Re-index IDs sequentially
        for idx, item in enumerate(deduped, start=1):
            item.id = f"ev_{idx}"

        final_items = deduped[:max_results]

        # 5. Determine combined status
        if final_items:
            status = "SUCCESS"
        elif not all_items and errors and len(errors) == 2:
            status = "ERROR"
        elif tavily_res.status == "NO_API_KEY" and not rss_res.items and not errors:
            status = "SUCCESS"  # Valid search, but no corroborating evidence found
        else:
            status = "PARTIAL" if queries_run else "SUCCESS"

        error_summary = " | ".join(errors) if errors else None

        return CollectionResult(
            items=final_items,
            status=status,
            total_found=len(final_items),
            queries_run=queries_run,
            error_message=error_summary,
        )

    def collect_evidence(self, claim: ExtractedClaim, max_results: int = 8) -> List[EvidenceItem]:
        """Convenience method returning merged list of EvidenceItem objects."""
        return self.collect_with_status(claim, max_results=max_results).items
