"""
Mis-Info Verify - Claim Extractor Module.

Extracts verifiable factual claims, core propositions, key entities, dates,
and locations from user-submitted social media posts, headlines, articles, or URLs.
Includes robust SSRF protection, timeout enforcement, and deterministic fallbacks.
"""

import re
import socket
import logging
import ipaddress
from urllib.parse import urlparse
from typing import List, Optional, Tuple
import requests

from src import ExtractedClaim

logger = logging.getLogger("mindmesh.claim_extractor")

# Maximum permitted response size for remote URL retrieval (1 MB)
MAX_URL_RESPONSE_BYTES = 1024 * 1024
DEFAULT_HTTP_TIMEOUT = 5.0  # seconds


class ClaimExtractorError(Exception):
    """Custom exception raised when claim extraction fails or input is invalid."""
    pass


class ClaimExtractor:
    """Extracts factual claims, entities, dates, and locations from raw input text or URLs.

    Provides robust text cleaning, URL safety/SSRF filtering, heuristic proposition
    segmentation, and deterministic fallback parsing.
    """

    def __init__(self) -> None:
        logger.debug("ClaimExtractor initialized.")

    def is_url(self, text: str) -> bool:
        """Check if the provided text string is an HTTP/HTTPS URL."""
        if not text:
            return False
        text = text.strip()
        try:
            parsed = urlparse(text)
            return bool(parsed.scheme in ("http", "https") and parsed.netloc)
        except Exception:
            return False

    def validate_url_safety(self, url: str) -> Tuple[bool, str]:
        """Validate URL against SSRF, internal networks, loopbacks, and dangerous protocols.

        Args:
            url: Target URL string.

        Returns:
            Tuple of (is_safe: bool, reason: str).
        """
        try:
            parsed = urlparse(url.strip())
            if parsed.scheme.lower() not in ("http", "https"):
                return False, f"Unsupported URL scheme: {parsed.scheme}. Only HTTP and HTTPS are allowed."

            hostname = parsed.hostname
            if not hostname:
                return False, "URL does not contain a valid hostname."

            # Disallow localhost and local domain aliases
            if hostname.lower() in ("localhost", "127.0.0.1", "::1", "0.0.0.0") or hostname.lower().endswith(".local"):
                return False, f"Access to localhost/local domain ({hostname}) is restricted."

            # Resolve DNS and check each IP address
            addr_info = socket.getaddrinfo(hostname, None)
            for item in addr_info:
                ip_str = item[4][0]
                ip = ipaddress.ip_address(ip_str)
                if (
                    ip.is_private
                    or ip.is_loopback
                    or ip.is_link_local
                    or ip.is_reserved
                    or ip.is_unspecified
                ):
                    return False, f"URL resolves to a restricted private or internal IP address: {ip_str}"

            return True, "URL is safe for retrieval."
        except socket.gaierror:
            return False, f"Could not resolve domain: {url}"
        except Exception as e:
            return False, f"URL safety validation failed: {e}"

    def fetch_url_content(self, url: str) -> str:
        """Safely fetch headline and textual body content from a remote URL.

        Enforces SSRF prevention, response size caps, content-type checks, and timeouts.
        Does not execute scripts or evaluate remote HTML.
        """
        is_safe, reason = self.validate_url_safety(url)
        if not is_safe:
            logger.warning("SSRF/Safety check blocked URL '%s': %s", url, reason)
            raise ClaimExtractorError(f"URL access blocked: {reason}")

        try:
            headers = {"User-Agent": "MindMeshVerify/1.0 (Evidence-First Misinformation Research)"}
            with requests.get(url, headers=headers, timeout=DEFAULT_HTTP_TIMEOUT, stream=True) as resp:
                resp.raise_for_status()

                # Verify content type is text or HTML
                content_type = resp.headers.get("Content-Type", "").lower()
                if "text/html" not in content_type and "text/plain" not in content_type:
                    raise ClaimExtractorError(
                        f"Unsupported remote content type: {content_type}. Only HTML or plain text can be inspected."
                    )

                # Read body content with byte cap to prevent memory exhaustion
                raw_bytes = bytearray()
                for chunk in resp.iter_content(chunk_size=8192):
                    raw_bytes.extend(chunk)
                    if len(raw_bytes) > MAX_URL_RESPONSE_BYTES:
                        logger.warning("Remote page exceeded max size (%d bytes); truncating.", MAX_URL_RESPONSE_BYTES)
                        break

                text = raw_bytes.decode("utf-8", errors="replace")

                # Extract title tag if present
                title_match = re.search(r"<title[^>]*>(.*?)</title>", text, re.IGNORECASE | re.DOTALL)
                page_title = title_match.group(1).strip() if title_match else ""

                # Strip HTML script, style, and markup tags
                no_scripts = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", text, flags=re.IGNORECASE | re.DOTALL)
                clean_body = re.sub(r"<[^>]+>", " ", no_scripts)
                clean_body = re.sub(r"\s+", " ", clean_body).strip()

                if page_title:
                    return f"{page_title} - {clean_body[:400]}"
                return clean_body[:500] if clean_body else ""

        except requests.exceptions.Timeout:
            raise ClaimExtractorError(f"Timed out after {DEFAULT_HTTP_TIMEOUT}s while retrieving URL content.")
        except requests.exceptions.RequestException as e:
            raise ClaimExtractorError(f"Network error while retrieving URL: {e}")

    def clean_text(self, text: str) -> str:
        """Sanitize raw text by removing extraneous whitespace, control characters,

        and standard social media artifacts (e.g. hashtags, RT markers).
        """
        if not text:
            return ""

        # Normalize multiple spaces and newlines
        cleaned = re.sub(r"\s+", " ", text).strip()
        # Remove leading 'RT @user:' prefixes common in tweets
        cleaned = re.sub(r"^rt\s+@[a-zA-Z0-9_]+:\s*", "", cleaned, flags=re.IGNORECASE)
        return cleaned

    def extract_entities_heuristic(self, text: str) -> List[str]:
        """Extract candidate proper nouns, acronyms, and capitalized entities."""
        words = text.split()
        entities = []
        for word in words:
            clean_word = re.sub(r"[^\w\s]", "", word)
            if clean_word and clean_word[0].isupper() and len(clean_word) > 1:
                if clean_word not in entities:
                    entities.append(clean_word)
        return entities

    def extract_dates(self, text: str) -> List[str]:
        """Detect date and temporal markers in the text (e.g. years, months, relative dates)."""
        date_patterns = [
            r"\b(19\d{2}|20\d{2})\b",  # Years 1900-2099
            r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b",
            r"\b(yesterday|today|last week|breaking|recently)\b",
        ]
        found = []
        for pat in date_patterns:
            matches = re.findall(pat, text, re.IGNORECASE)
            for m in matches:
                clean_m = m.strip().title()
                if clean_m not in found:
                    found.append(clean_m)
        return found

    def extract_locations(self, text: str) -> List[str]:
        """Detect potential location references using prepositional heuristic."""
        # Find capitalized words following 'in', 'at', 'near', 'from'
        loc_pattern = r"\b(?:in|at|near|from)\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)\b"
        matches = re.findall(loc_pattern, text)
        locations = []
        for loc in matches:
            clean_loc = loc.strip()
            if clean_loc not in locations and clean_loc not in ("The", "This", "That"):
                locations.append(clean_loc)
        return locations

    def segment_claims(self, text: str) -> Tuple[str, List[str]]:
        """Separate compound statements into primary claim and secondary propositions."""
        # Split sentences by period, exclamation, or semicolon
        sentences = [s.strip() for s in re.split(r"[.!?;\n]+", text) if len(s.strip()) > 15]
        if not sentences:
            return text, []

        primary_claim = sentences[0]
        additional_claims = sentences[1:4]  # Keep up to 3 secondary claims
        return primary_claim, additional_claims

    def extract(self, raw_input: str) -> ExtractedClaim:
        """Extract a structured claim from raw input text or URL.

        Args:
            raw_input: Raw string containing social media post, headline, or URL.

        Returns:
            ExtractedClaim model containing the normalized claim and metadata.

        Raises:
            ClaimExtractorError: If input is empty or cannot be parsed.
        """
        if not raw_input or not raw_input.strip():
            logger.error("Attempted to extract claim from empty input.")
            raise ClaimExtractorError("Input text cannot be empty or whitespace only.")

        raw_trimmed = raw_input.strip()
        source_url: Optional[str] = None
        additional_claims: List[str] = []

        if self.is_url(raw_trimmed):
            source_url = raw_trimmed
            domain = urlparse(raw_trimmed).netloc
            logger.info("URL detected as input: %s", raw_trimmed)

            # Attempt safe retrieval
            try:
                retrieved_content = self.fetch_url_content(raw_trimmed)
                if retrieved_content:
                    cleaned_claim = self.clean_text(retrieved_content[:300])
                else:
                    cleaned_claim = f"Verification requested for content at: {raw_trimmed}"
            except ClaimExtractorError as ce:
                logger.warning("URL content retrieval failed or was blocked: %s", ce)
                cleaned_claim = f"Verification requested for content at: {raw_trimmed}"

            entities = [domain]
        else:
            cleaned_text = self.clean_text(raw_trimmed)
            primary_claim, secondaries = self.segment_claims(cleaned_text)
            cleaned_claim = primary_claim
            additional_claims = secondaries
            entities = self.extract_entities_heuristic(cleaned_text)
            logger.info("Extracted claim with %d candidate entities.", len(entities))

        dates = self.extract_dates(raw_trimmed)
        locations = self.extract_locations(raw_trimmed)

        return ExtractedClaim(
            raw_input=raw_trimmed,
            cleaned_claim=cleaned_claim,
            detected_entities=entities,
            source_url=source_url,
            additional_claims=additional_claims,
            detected_dates=dates,
            detected_locations=locations,
        )
