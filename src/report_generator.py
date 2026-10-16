"""
Mis-Info Verify - Report Generator Module.

Formats structured verification outcomes into clear, transparent, human-readable
markdown summaries and exportable JSON audit records. Displays categorized evidence
(supporting, contradicting, contextual), explicit limitations, and citation links.
"""

import json
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List

from config import config
from src import VerificationResult, EvidenceItem

logger = logging.getLogger("mindmesh.report_generator")


class ReportGenerator:
    """Produces structured markdown and dictionary representations of verification results."""

    def __init__(self, output_dir: Optional[str] = None) -> None:
        self.output_dir = Path(output_dir or config.outputs_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.debug("ReportGenerator initialized with output dir: %s", self.output_dir)

    def _render_evidence_section(self, title: str, items: List[EvidenceItem]) -> List[str]:
        """Render a formatted list of evidence items for markdown."""
        lines = [f"### {title}"]
        if not items:
            lines.append("_No evidence identified in this category._\n")
            return lines

        for item in items:
            date_str = f" | Published: {item.published_date}" if item.published_date else ""
            item_title = getattr(item, "title", None) or item.source_name
            ev_id = getattr(item, "id", "ref")
            lines.extend([
                f"- **[{ev_id}] [{item_title}]({item.url})** ({item.source_name}{date_str})",
                f"  > \"{item.snippet}\"",
                "",
            ])
        return lines

    def generate_markdown(self, result: VerificationResult) -> str:
        """Format a VerificationResult as a clean, categorized markdown report.

        Args:
            result: The VerificationResult model instance.

        Returns:
            Rendered markdown string.
        """
        status_badges = {
            "LIKELY RELIABLE": "🟢 **LIKELY RELIABLE**",
            "NEEDS VERIFICATION": "🟡 **NEEDS VERIFICATION**",
            "POTENTIAL FAKE / MISINFORMATION": "🔴 **POTENTIAL FAKE / MISINFORMATION**",
        }
        badge = status_badges.get(result.status.value, f"**{result.status.value}**")

        evidence_by_id: Dict[str, EvidenceItem] = {
            getattr(item, "id", ""): item for item in result.evidence_items
        }

        # Resolve categorized evidence items
        supporting = [evidence_by_id[eid] for eid in result.supporting_evidence_ids if eid in evidence_by_id]
        contradicting = [evidence_by_id[eid] for eid in result.contradicting_evidence_ids if eid in evidence_by_id]
        contextual = [evidence_by_id[eid] for eid in result.mixed_or_contextual_evidence_ids if eid in evidence_by_id]

        lines = [
            "# Verification Report: Mis-Info Verify",
            f"**Audit Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
            f"**Verification Engine:** {result.model_used or 'Mis-Info Hybrid Engine'}",
            "",
            "## Claim Analyzed",
            f"> **Statement:** {result.claim.cleaned_claim}",
            "",
            f"**Original Input Text:** `{result.claim.raw_input}`",
        ]

        if result.claim.source_url:
            lines.append(f"**Referenced URL:** [{result.claim.source_url}]({result.claim.source_url})")

        if result.claim.detected_entities:
            lines.append(f"**Entities Identified:** {', '.join(result.claim.detected_entities)}")

        lines.extend([
            "",
            "## Assessment Outcome",
            f"### Verdict: {badge}",
            f"- **Confidence Level:** {result.confidence_score * 100:.1f}%",
            f"- **Estimated Uncertainty:** {result.uncertainty_score * 100:.1f}%",
            "",
            "### Executive Summary",
            result.summary or result.explanation,
            "",
            "### Detailed Reasoning & Analysis",
            result.reasoning or result.explanation,
            "",
            "## Evidence Audit by Category",
        ])

        if result.evidence_items:
            # If categorized lists are available, render by category
            if supporting or contradicting or contextual:
                lines.extend(self._render_evidence_section("Supporting Evidence", supporting))
                lines.extend(self._render_evidence_section("Contradicting Evidence", contradicting))
                lines.extend(self._render_evidence_section("Contextual / Mixed Evidence", contextual))
            else:
                # Fallback to general evidence listing
                lines.extend(self._render_evidence_section("Retrieved Evidence", result.evidence_items))
        else:
            lines.extend([
                "_No external corroborating or refuting evidence retrieved._",
                "_Note: Under evidence-first rules, absence of evidence never implies falsity._",
                "",
            ])

        lines.extend([
            "## Limitations & Evidence Gaps",
            result.limitations or "No explicit limitations recorded.",
            "",
            "---",
            "*Mis-Info Verify adheres to strict evidence-first verification. Never treat unverified claims as definitively false without verified contradictory evidence.*",
        ])

        return "\n".join(lines)

    def export_report(self, result: VerificationResult, filename_stem: str = "verification_report") -> Path:
        """Save both a markdown and JSON copy of the report into the outputs directory.

        Returns:
            Path to the generated markdown report.
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        md_file = self.output_dir / f"{filename_stem}_{timestamp}.md"
        json_file = self.output_dir / f"{filename_stem}_{timestamp}.json"

        md_content = self.generate_markdown(result)
        md_file.write_text(md_content, encoding="utf-8")

        # Serializing result safely
        json_file.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        logger.info("Exported verification reports to %s and %s", md_file.name, json_file.name)
        return md_file
