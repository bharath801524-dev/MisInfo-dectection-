"""
Mis-Info Verify - Core Source Package.

This package contains the core verification pipeline components:
- claim_extractor: Extracts core factual claims from user input.
- evidence_collector: Gathers external evidence and corroborating sources.
- evidence_processor: Cleans, parses, and compares evidence across sources.
- verification_engine: Assesses factual veracity, consistency, and uncertainty.
- temporal_tracker: Monitors how claim verification evolves over time.
- database: Handles local persistence of claims, evidence, and audit logs.
- report_generator: Creates human-readable verification reports and explanations.
"""

from enum import Enum
from typing import List, Optional
from datetime import datetime, timezone
import uuid
from pydantic import BaseModel, Field


class VerificationStatus(str, Enum):
    """The three official assessment outcomes for Mis-Info Verify."""
    LIKELY_RELIABLE = "LIKELY RELIABLE"
    NEEDS_VERIFICATION = "NEEDS VERIFICATION"
    POTENTIAL_FAKE = "POTENTIAL FAKE / MISINFORMATION"


class ExtractedClaim(BaseModel):
    """Represents a factual claim extracted from raw input text or URLs."""
    raw_input: str = Field(..., description="Original input text or URL")
    cleaned_claim: str = Field(..., description="Normalized factual statement")
    detected_entities: List[str] = Field(default_factory=list, description="Key entities identified")
    source_url: Optional[str] = Field(default=None, description="URL if input was a link")
    additional_claims: List[str] = Field(default_factory=list, description="Secondary or discrete factual propositions")
    detected_dates: List[str] = Field(default_factory=list, description="Time or date markers identified in claim")
    detected_locations: List[str] = Field(default_factory=list, description="Geographic or location markers identified")


class EvidenceItem(BaseModel):
    """Represents a piece of evidence retrieved from an external source."""
    id: str = Field(default_factory=lambda: f"ev_{uuid.uuid4().hex[:6]}", description="Unique stable evidence identifier")
    source_name: str = Field(..., description="Publisher or domain name")
    url: str = Field(..., description="Canonical link to the source document")
    snippet: str = Field(..., description="Relevant excerpt or headline")
    title: Optional[str] = Field(default=None, description="Page or article title")
    published_date: Optional[str] = Field(default=None, description="Publication timestamp or date string")
    retrieval_timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO timestamp when evidence was retrieved",
    )
    original_query: Optional[str] = Field(default=None, description="Search query that retrieved this item")
    relevance_score: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Provider-supplied relevance score")
    reliability_score: float = Field(default=0.5, ge=0.0, le=1.0, description="Estimated domain reliability score")
    source_type: str = Field(default="web", description="Type of source (e.g. news, fact_check, academic, web)")
    source_origin: str = Field(default="tavily", description="Provenance origin: 'tavily', 'google_news_rss', etc.")
    is_redirect_url: bool = Field(default=False, description="True if URL is a redirect/intermediate link")
    image_url: Optional[str] = Field(default=None, description="Featured image or news photo URL")


class VerificationResult(BaseModel):
    """Final assessment model containing the outcome, explanation, and evidence."""
    claim: ExtractedClaim
    status: VerificationStatus
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Model confidence (0.0 to 1.0)")
    uncertainty_score: float = Field(..., ge=0.0, le=1.0, description="Estimated uncertainty level")
    explanation: str = Field(..., description="Clear explanation of the reasoning and evidence")
    evidence_items: List[EvidenceItem] = Field(default_factory=list, description="Supporting or refuting evidence")
    source_urls: List[str] = Field(default_factory=list, description="All consulted source links")

    # Phase 2 Explainability & Citation Fields
    summary: Optional[str] = Field(default=None, description="Concise executive summary of verification verdict")
    reasoning: Optional[str] = Field(default=None, description="In-depth reasoning linking claims to evidence citations")
    supporting_evidence_ids: List[str] = Field(default_factory=list, description="Evidence IDs directly supporting claim")
    contradicting_evidence_ids: List[str] = Field(default_factory=list, description="Evidence IDs contradicting claim")
    mixed_or_contextual_evidence_ids: List[str] = Field(default_factory=list, description="Evidence IDs adding context or mixed data")
    limitations: Optional[str] = Field(default=None, description="Caveats, evidence gaps, or missing information")
    investigated_claim: Optional[str] = Field(default=None, description="The specific proposition examined")
    model_used: Optional[str] = Field(default=None, description="AI model or fallback logic used for verification")


__all__ = [
    "VerificationStatus",
    "ExtractedClaim",
    "EvidenceItem",
    "VerificationResult",
]
