"""
Mis-Info Verify - Evidence Processor Module.

Cleans, deduplicates, and cross-compares retrieved evidence from multiple sources.
Computes source diversity, agreement metrics, domain credibility tiers, and
prepares structured evidence bundles for AI verification analysis.
"""

import re
import logging
from typing import List, Dict, Set, Optional, Any
from urllib.parse import urlparse
from pydantic import BaseModel, Field

from src import EvidenceItem, ExtractedClaim

logger = logging.getLogger("mindmesh.evidence_processor")

# Baseline domain credibility weightings (heuristic tiering for accredited news/fact-checkers)
# Used as reference heuristics, not definitive truth guarantees.
TRUSTED_DOMAINS: Dict[str, float] = {
    # Global accredited news & scientific reference
    "reuters.com": 0.95,
    "apnews.com": 0.95,
    "snopes.com": 0.90,
    "politifact.com": 0.90,
    "factcheck.org": 0.90,
    "bbc.com": 0.90,
    "bbc.co.uk": 0.90,
    "who.int": 0.98,
    "cdc.gov": 0.98,
    "nature.com": 0.95,
    "science.org": 0.95,
    # Accredited Indian national & regional news outlets
    "thehindu.com": 0.90,
    "indianexpress.com": 0.90,
    "hindustantimes.com": 0.85,
    "ndtv.com": 0.85,
    "timesofindia.indiatimes.com": 0.85,
    "indiatimes.com": 0.85,
    "indiatoday.in": 0.85,
    "pib.gov.in": 0.98,
    "boomlive.in": 0.90,
    "altnews.in": 0.90,
    "thequint.com": 0.85,
    "news18.com": 0.80,
    "firstpost.com": 0.80,
    "livemint.com": 0.85,
    "business-standard.com": 0.85,
    "deccanherald.com": 0.85,
}


class ProcessedEvidence(BaseModel):
    """Aggregated, deduplicated, and scored evidence analysis."""
    unique_items: List[EvidenceItem] = Field(default_factory=list)
    unique_domains: Set[str] = Field(default_factory=set)
    domain_diversity_score: float = Field(default=0.0, ge=0.0, le=1.0)
    average_source_reliability: float = Field(default=0.5, ge=0.0, le=1.0)

    # Phase 2 Traceability & Transparency Extensions
    all_collected_items: List[EvidenceItem] = Field(default_factory=list, description="All retrieved items before filtering")
    independent_sources_count: int = Field(default=0, description="Count of distinct independent top-level publisher domains")
    duplicate_count: int = Field(default=0, description="Number of duplicate items removed")
    syndicated_count: int = Field(default=0, description="Count of items identified as syndicated wire stories")
    limitations_note: Optional[str] = Field(default=None, description="Explicit notes on evidence gaps or limitations")


class EvidenceProcessor:
    """Processes, filters, and analyzes evidence items across distinct sources."""

    def __init__(self, domain_weights: Optional[Dict[str, float]] = None) -> None:
        self.domain_weights = domain_weights or TRUSTED_DOMAINS
        logger.debug("EvidenceProcessor initialized with %d known domain weights.", len(self.domain_weights))

    def extract_domain(self, url: str) -> str:
        """Extract clean root domain from a given URL string."""
        if not url:
            return ""
        try:
            netloc = urlparse(url).netloc.lower()
            if netloc.startswith("www."):
                netloc = netloc[4:]
            return netloc
        except Exception:
            return ""

    def extract_publisher_domain(self, item: EvidenceItem) -> str:
        """Extract the effective publisher domain, resolving Google News redirects to the publisher domain."""
        raw_domain = self.extract_domain(item.url)
        if raw_domain == "news.google.com" and item.source_name:
            clean_name = re.sub(r"[^\w]", "", item.source_name.lower())
            for trusted_domain in self.domain_weights:
                root = trusted_domain.split(".")[0]
                if root in clean_name or clean_name in root:
                    return trusted_domain
            return f"{clean_name}.publisher" if clean_name else "news.google.com"
        return raw_domain

    def estimate_source_credibility(self, url: str, source_name: Optional[str] = None) -> float:
        """Estimate baseline credibility score of a source URL or publisher name."""
        domain = self.extract_domain(url)
        for trusted_domain, score in self.domain_weights.items():
            if domain == trusted_domain or domain.endswith("." + trusted_domain):
                return score

        # Check publisher name if URL is a redirect/aggregator
        if source_name:
            clean_name = re.sub(r"[^\w]", "", source_name.lower())
            for trusted_domain, score in self.domain_weights.items():
                root = trusted_domain.split(".")[0]
                if root and (root in clean_name or clean_name in root):
                    return score

        # Default baseline for unknown domain (0.5 = uncalibrated, not automatically fake)
        return 0.5

    def normalize_snippet(self, text: str) -> str:
        """Normalize snippet text for near-duplicate comparison."""
        cleaned = re.sub(r"[^\w\s]", "", text.lower())
        return re.sub(r"\s+", " ", cleaned).strip()

    def deduplicate(self, items: List[EvidenceItem]) -> List[EvidenceItem]:
        """Deduplicate evidence items by canonical URL and normalized content snippets."""
        seen_urls = set()
        seen_snippets = set()
        deduped = []

        for item in items:
            clean_url = item.url.strip().lower()
            norm_snip = self.normalize_snippet(item.snippet)[:80]  # First 80 characters normalized

            if clean_url and clean_url in seen_urls:
                continue
            if norm_snip and len(norm_snip) > 20 and norm_snip in seen_snippets:
                continue

            if clean_url:
                seen_urls.add(clean_url)
            if norm_snip:
                seen_snippets.add(norm_snip)

            deduped.append(item)

        return deduped

    def compute_claim_relevance(self, claim: ExtractedClaim, item: EvidenceItem) -> float:
        """Compute a heuristic keyword overlap relevance score between claim and evidence snippet."""
        claim_words = set(re.findall(r"\w+", claim.cleaned_claim.lower()))
        # Filter out common short stopwords
        stopwords = {"the", "a", "an", "is", "are", "was", "were", "and", "or", "in", "on", "at", "to", "for", "with", "of", "that"}
        key_claim_words = {w for w in claim_words if w not in stopwords and len(w) > 2}

        if not key_claim_words:
            return 0.5

        snippet_words = set(re.findall(r"\w+", item.snippet.lower()))
        overlap = len(key_claim_words.intersection(snippet_words))
        return min(overlap / len(key_claim_words), 1.0)

    def process_evidence(self, items: List[EvidenceItem]) -> ProcessedEvidence:
        """Analyze, score, and aggregate a collection of evidence items.

        Args:
            items: Raw list of EvidenceItem instances.

        Returns:
            ProcessedEvidence object containing metrics, deduplicated sources, and audit data.
        """
        if not items:
            logger.info("No evidence items to process. Returning zeroed metrics.")
            return ProcessedEvidence(
                unique_items=[],
                unique_domains=set(),
                domain_diversity_score=0.0,
                average_source_reliability=0.0,
                all_collected_items=[],
                independent_sources_count=0,
                duplicate_count=0,
                limitations_note="No evidence items were retrieved. Verification cannot be confirmed.",
            )

        deduped = self.deduplicate(items)
        duplicate_count = len(items) - len(deduped)
        domains: Set[str] = set()
        total_reliability = 0.0

        for item in deduped:
            domain = self.extract_publisher_domain(item)
            if domain:
                domains.add(domain)
            score = self.estimate_source_credibility(item.url, source_name=item.source_name)
            item.reliability_score = score
            total_reliability += score

        avg_reliability = total_reliability / len(deduped) if deduped else 0.5
        # Diversity score: higher if evidence comes from multiple distinct domains (up to 5)
        diversity_score = min(len(domains) / 5.0, 1.0)

        # Detect syndicated wire republishing (e.g. PTI, ANI)
        syndicated_wires = ["pti", "press trust of india", "ani", "asian news international", "reuters news agency"]
        syndicated_items = [
            item for item in deduped
            if any(wire in item.snippet.lower() or wire in (item.title or "").lower() for wire in syndicated_wires)
        ]
        syndicated_count = len(syndicated_items)

        # Distinguish raw count vs independent corroboration
        limitations = None
        if syndicated_count > 1:
            limitations = "Multiple retrieved items republish syndicated wire reports (e.g. PTI/ANI); headline repetition does not constitute independent corroboration."
        elif len(domains) == 1 and len(deduped) > 1:
            limitations = "All retrieved evidence originates from a single publisher domain. Independent corroboration is limited."
        elif len(domains) < 3 and len(deduped) > 0:
            limitations = "Evidence was gathered from fewer than 3 distinct sources; broader confirmation recommended."

        logger.info(
            "Processed %d items from %d unique domains (avg reliability: %.2f, syndicated: %d)",
            len(deduped),
            len(domains),
            avg_reliability,
            syndicated_count,
        )

        return ProcessedEvidence(
            unique_items=deduped,
            unique_domains=domains,
            domain_diversity_score=diversity_score,
            average_source_reliability=avg_reliability,
            all_collected_items=items,
            independent_sources_count=len(domains),
            duplicate_count=duplicate_count,
            syndicated_count=syndicated_count,
            limitations_note=limitations,
        )

    def prepare_bundle_for_gemini(
        self,
        claim: ExtractedClaim,
        processed: ProcessedEvidence,
    ) -> List[Dict[str, Any]]:
        """Format processed evidence into a clean, compact structured payload for the Gemini model.

        Ensures each item has a stable ID, publisher, URL, date, and concise excerpt.
        """
        bundle = []
        for item in processed.unique_items:
            bundle.append({
                "id": item.id,
                "title": item.title or item.source_name,
                "source": item.source_name,
                "url": item.url,
                "published_date": item.published_date or "Date not specified",
                "snippet": item.snippet[:400],
                "origin": item.source_origin,
            })
        return bundle
