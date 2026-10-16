"""
Mis-Info Verify - Streamlit Web Application (Phase 2).

Evidence-first misinformation verification portal for social media claims,
headlines, and URLs with live Tavily search and Google Gemini reasoning.
"""

import re
import html
import textwrap
from typing import Optional
import urllib.request
from urllib.parse import urlparse
import streamlit as st
from config import config, logger
from src import VerificationStatus, VerificationResult, EvidenceItem
from src.claim_extractor import ClaimExtractor, ClaimExtractorError
from src.evidence_collector import MultiSourceEvidenceCollector
from src.evidence_processor import EvidenceProcessor
from src.verification_engine import VerificationEngine
from src.database import DatabaseManager
from src.report_generator import ReportGenerator

# Set Streamlit page configuration
st.set_page_config(
    page_title=f"{config.app_name} | Evidence-First Misinformation Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished White with Plum theme
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"], .stApp {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        background-color: #FFFFFF !important;
        color: #2D142A !important;
    }

    [data-testid="stSidebar"] {
        background-color: #FAF4F9 !important;
        border-right: 1px solid #EBDCE8 !important;
    }

    [data-testid="stSidebar"] * {
        color: #3B1B38 !important;
    }

    .main-header {
        font-size: 2.35rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
        color: #4A154B !important;
        letter-spacing: -0.5px;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #6B4368 !important;
        margin-bottom: 1.5rem;
        font-weight: 400;
    }

    h1, h2, h3, h4, [data-testid="stMarkdownContainer"] h3 {
        color: #4A154B !important;
        font-weight: 700 !important;
    }

    .badge-card {
        padding: 1.1rem 1.25rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        border: 1px solid;
        box-shadow: 0 4px 14px rgba(92, 29, 84, 0.04);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .badge-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 18px rgba(92, 29, 84, 0.08);
    }
    .badge-reliable {
        background-color: #F4FBF7 !important;
        border-color: #A3E6C5 !important;
        border-left: 5px solid #16A34A !important;
    }
    .badge-needs-verification {
        background-color: #FFFDF5 !important;
        border-color: #FDE68A !important;
        border-left: 5px solid #D97706 !important;
    }
    .badge-fake {
        background-color: #FDF4F6 !important;
        border-color: #FECDD3 !important;
        border-left: 5px solid #E11D48 !important;
    }
    .status-title {
        font-weight: 700;
        font-size: 1.05rem;
        margin-bottom: 0.2rem;
    }

    /* Primary Button - Imperial Plum */
    .stButton > button[kind="primary"], .stButton > button {
        background: linear-gradient(135deg, #5C1D54 0%, #43123D 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.65rem 1.75rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        box-shadow: 0 4px 14px rgba(92, 29, 84, 0.25) !important;
        transition: all 0.2s ease-in-out !important;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #481341 0%, #300A2C 100%) !important;
        box-shadow: 0 6px 20px rgba(92, 29, 84, 0.35) !important;
        transform: translateY(-1px) !important;
    }

    /* Input text area styling */
    .stTextArea textarea {
        background-color: #FFFFFF !important;
        color: #2D142A !important;
        border: 1.5px solid #D8C3D5 !important;
        border-radius: 10px !important;
        font-size: 0.98rem !important;
        padding: 0.85rem !important;
    }
    .stTextArea textarea:focus {
        border-color: #5C1D54 !important;
        box-shadow: 0 0 0 3px rgba(92, 29, 84, 0.15) !important;
    }
    .stTextArea label {
        color: #4A154B !important;
        font-weight: 600 !important;
    }

    /* Metrics with plum accents */
    [data-testid="stMetric"] {
        background-color: #FAF4F9 !important;
        border: 1px solid #EBDCE8 !important;
        border-radius: 10px !important;
        padding: 0.85rem 1rem !important;
    }
    [data-testid="stMetricLabel"] {
        color: #6E4B6C !important;
        font-weight: 600 !important;
    }
    [data-testid="stMetricValue"] {
        color: #4A154B !important;
        font-weight: 800 !important;
    }

    .evidence-item-box {
        background-color: #FFFFFF !important;
        border: 1px solid #EBDCE8 !important;
        border-radius: 8px !important;
        padding: 0.95rem 1.1rem !important;
        margin-bottom: 0.85rem !important;
        box-shadow: 0 2px 8px rgba(92, 29, 84, 0.03) !important;
        transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
    }
    .evidence-item-box:hover {
        border-color: #CBA4C7 !important;
        box-shadow: 0 4px 14px rgba(92, 29, 84, 0.07) !important;
    }
    .evidence-item-box a {
        color: #5C1D54 !important;
        text-decoration: none !important;
    }
    .evidence-item-box a:hover {
        color: #380C34 !important;
        text-decoration: underline !important;
    }

    .evidence-tag {
        display: inline-block;
        padding: 0.25rem 0.55rem;
        font-size: 0.75rem;
        font-weight: 700;
        border-radius: 6px;
        margin-right: 0.5rem;
        letter-spacing: 0.3px;
    }
    .tag-support {
        background-color: #DCFCE7 !important;
        color: #15803D !important;
        border: 1px solid #86EFAC !important;
    }
    .tag-contradict {
        background-color: #FFE4E6 !important;
        color: #BE123C !important;
        border: 1px solid #FDA4AF !important;
    }
    .tag-context {
        background-color: #F4ECF3 !important;
        color: #5C1D54 !important;
        border: 1px solid #DFC0DC !important;
    }
    .tag-tavily {
        background-color: #EFF6FF !important;
        color: #1D4ED8 !important;
        border: 1px solid #BFDBFE !important;
    }
    .tag-rss {
        background-color: #FDF2F8 !important;
        color: #9D174D !important;
        border: 1px solid #FBCFE8 !important;
    }
    .tag-justification {
        background-color: #FAF5FF !important;
        color: #6B21A8 !important;
        border: 1px solid #E9D5FF !important;
    }

    /* Sidebar Brand & Badges */
    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 0.35rem;
    }
    .sidebar-title {
        font-size: 1.3rem;
        font-weight: 800;
        color: #4A154B !important;
    }
    .sidebar-meta {
        display: flex;
        gap: 0.4rem;
        margin-bottom: 0.9rem;
    }
    .plum-chip {
        display: inline-block;
        background-color: #5C1D54;
        color: #FFFFFF !important;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 0.15rem 0.55rem;
        border-radius: 12px;
    }
    .plum-chip-outline {
        display: inline-block;
        background-color: transparent;
        color: #5C1D54 !important;
        border: 1px solid #5C1D54;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 0.15rem 0.55rem;
        border-radius: 12px;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1.5px solid #EBDCE8 !important;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px 6px 0 0;
        color: #6E4B6C !important;
        font-weight: 600 !important;
        padding: 8px 16px;
    }
    .stTabs [aria-selected="true"] {
        color: #5C1D54 !important;
        border-bottom: 3px solid #5C1D54 !important;
        background-color: #FAF4F9 !important;
    }

    .stDownloadButton > button {
        background: #FFFFFF !important;
        color: #5C1D54 !important;
        border: 1.5px solid #5C1D54 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }
    .stDownloadButton > button:hover {
        background: #FAF4F9 !important;
        border-color: #3E0E38 !important;
        color: #3E0E38 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_db_manager() -> DatabaseManager:
    """Initialize and cache the database manager instance."""
    return DatabaseManager()


def render_sidebar(db: DatabaseManager) -> None:
    """Render the sidebar with brand info, audit history, and core rules (system status omitted)."""
    st.sidebar.markdown(
        f"""
        <div class="sidebar-brand">
            <span style="font-size: 1.5rem;">🛡️</span>
            <span class="sidebar-title">Mis-Info Verify</span>
        </div>
        <div class="sidebar-meta">
            <span class="plum-chip">v{config.app_version}</span>
            <span class="plum-chip-outline">{config.app_env}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.divider()
    st.sidebar.subheader("Recent Verification Audits")
    try:
        recent = db.list_recent_verifications(limit=5)
        if recent:
            for item in recent:
                st.sidebar.markdown(
                    f"**{item['status']}**  \n"
                    f"<small>{item['cleaned_claim'][:50]}...</small>  \n"
                    f"<small style='color: #6B4368;'>{item['created_at']}</small>",
                    unsafe_allow_html=True,
                )
                st.sidebar.markdown("---")
        else:
            st.sidebar.caption("No verifications saved yet.")
    except Exception as e:
        logger.error("Error reading recent verifications: %s", e)
        st.sidebar.caption("Unable to load history.")

    st.sidebar.info(
        "**Core Rule:** Never label a claim false merely because evidence was not found. "
        "Absence of evidence defaults to *Needs Verification* with high uncertainty."
    )


def _clean_text(text: Optional[str]) -> str:
    """Remove HTML tags, unescape HTML entities, and collapse extra whitespace from snippet/title."""
    if not text:
        return ""
    clean = re.sub(r"<[^>]+>", " ", text)
    clean = html.unescape(clean)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean


@st.cache_data(show_spinner=False, ttl=7200)
def _resolve_article_image(url: str) -> Optional[str]:
    """Attempt to extract the article's og:image or featured image meta tag."""
    if not url or not url.startswith(("http://", "https://")):
        return None
    if "news.google.com" in url:
        return None
    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            },
        )
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            content_type = resp.headers.get("content-type", "")
            if "text/html" not in content_type:
                return None
            chunk = resp.read(65536).decode("utf-8", errors="ignore")
            m = re.search(r'<meta[^>]+property=[\'"]og:image[\'"][^>]+content=[\'"]([^\'"]+)[\'"]', chunk, re.IGNORECASE)
            if not m:
                m = re.search(r'<meta[^>]+content=[\'"]([^\'"]+)[\'"][^>]+property=[\'"]og:image[\'"]', chunk, re.IGNORECASE)
            if not m:
                m = re.search(r'<meta[^>]+name=[\'"]twitter:image[\'"][^>]+content=[\'"]([^\'"]+)[\'"]', chunk, re.IGNORECASE)
            if m:
                img_url = m.group(1).strip()
                if img_url.startswith("//"):
                    img_url = "https:" + img_url
                elif img_url.startswith("/"):
                    parsed = urlparse(url)
                    img_url = f"{parsed.scheme}://{parsed.netloc}{img_url}"
                return img_url
    except Exception:
        return None
    return None


def render_evidence_cards(
    title: str,
    tag_class: str,
    tag_label: str,
    items: list[EvidenceItem],
    is_verdict_justification: bool = False,
) -> None:
    """Render a section of categorized evidence cards with news images, publisher details, and direct links."""
    st.markdown(f"#### {title} ({len(items)})")
    if not items:
        st.caption("No evidence items in this category.")
        return

    for item in items:
        with st.container(border=True):
            # 1. Badges & Source Origin (unindented HTML prevents markdown code block interpretation)
            origin_is_rss = getattr(item, "source_origin", "") == "google_news_rss"
            origin_label = "Google News RSS" if origin_is_rss else "Tavily Search"
            origin_class = "tag-rss" if origin_is_rss else "tag-tavily"
            redirect_note = " • _Google News Redirect_" if getattr(item, "is_redirect_url", False) else ""
            date_display = f" • {item.published_date}" if item.published_date else ""

            badges_html = (
                f"<span class='evidence-tag {tag_class}'>{tag_label}</span>"
                f"<span class='evidence-tag {origin_class}'>{origin_label}</span>"
            )
            if is_verdict_justification:
                badges_html += "<span class='evidence-tag tag-justification'>VERDICT CITATION</span>"

            st.markdown(badges_html, unsafe_allow_html=True)

            # 2. Extract clean title, snippet, and resolve article image
            item_title = _clean_text(item.title if item.title else item.source_name)
            clean_snippet = _clean_text(item.snippet)
            image_url = getattr(item, "image_url", None) or _resolve_article_image(item.url)

            # 3. Card layout with or without image
            if image_url:
                col_img, col_body = st.columns([1, 3], gap="medium")
                with col_img:
                    try:
                        st.image(image_url, use_container_width=True)
                    except Exception:
                        st.markdown(f"📰 **{item.source_name}**")
                with col_body:
                    st.markdown(f"##### [{item_title}]({item.url})")
                    st.caption(f"Published by **{item.source_name}**{date_display}{redirect_note}")
                    st.markdown(f"> {clean_snippet}")
                    st.link_button(f"🔗 Read Full Source Article on {item.source_name} ↗", item.url)
            else:
                st.markdown(f"##### [{item_title}]({item.url})")
                st.caption(f"Published by **{item.source_name}**{date_display}{redirect_note}")
                st.markdown(f"> {clean_snippet}")
                st.link_button(f"🔗 Read Full Source Article on {item.source_name} ↗", item.url)


def main() -> None:
    """Main Streamlit application entry point."""
    db = get_db_manager()
    render_sidebar(db)

    st.markdown('<div class="main-header">Mis-Info Verify</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Evidence-first social media claim analysis and explainable verification engine.</div>',
        unsafe_allow_html=True,
    )

    # Explanation cards for the 3 assessment outcomes
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """
            <div class="badge-card badge-reliable">
                <div class="status-title" style="color: #065F46;">🟢 LIKELY RELIABLE</div>
                <small>Multiple independent, credible sources corroborate the core factual claim.</small>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class="badge-card badge-needs-verification">
                <div class="status-title" style="color: #92400E;">🟡 NEEDS VERIFICATION</div>
                <small>Evidence is inconclusive, missing, or conflicting. Not assumed false.</small>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """
            <div class="badge-card badge-fake">
                <div class="status-title" style="color: #991B1B;">🔴 POTENTIAL FAKE / MISINFO</div>
                <small>Direct counter-evidence or fact-checker debunks identified.</small>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Submit Claim for Verification")
    user_input = st.text_area(
        label="Enter a claim, social media post text, headline, or article URL:",
        placeholder="e.g. 'Breaking: James Webb Telescope discovers water vapor on habitable exoplanet' or paste an article URL...",
        height=120,
        help="Paste a claim or link to inspect.",
    )

    col_btn, col_info = st.columns([1, 4])
    with col_btn:
        verify_clicked = st.button("Verify Claim", type="primary", use_container_width=True)

    if verify_clicked:
        if not user_input or not user_input.strip():
            st.warning("⚠️ Please enter a social media post, claim, headline, or URL first.")
            return

        with st.status("Executing Mis-Info Verification Pipeline...", expanded=True) as status_box:
            # 1. Claim Extraction
            st.write("🔍 **Stage 1:** Extracting core claim, entities, and temporal markers...")
            extractor = ClaimExtractor()
            try:
                extracted = extractor.extract(user_input)
            except ClaimExtractorError as e:
                status_box.update(label="Claim extraction failed", state="error")
                st.error(f"Error during claim extraction: {e}")
                return
            except Exception as e:
                status_box.update(label="Unexpected extraction error", state="error")
                logger.exception("Extraction error: %s", e)
                st.error("An unexpected error occurred while processing the input.")
                return

            # 2. Evidence Collection (Tavily Search + Google News RSS)
            st.write("🌐 **Stage 2:** Querying Tavily Search & Google News RSS feeds...")
            try:
                collector = MultiSourceEvidenceCollector()
                collection_result = collector.collect_with_status(extracted, max_results=8)
                raw_evidence = collection_result.items
                if collection_result.error_message:
                    st.caption(f"ℹ️ Search notice: {collection_result.error_message}")
            except Exception as _e:
                logger.exception("Evidence collection error: %s", _e)
                raw_evidence = []

            # 3. Evidence Processing & Deduplication
            st.write("📊 **Stage 3:** Normalizing sources, deduplicating, and assessing domain diversity...")
            try:
                processor = EvidenceProcessor()
                processed_bundle = processor.process_evidence(raw_evidence)
            except Exception as _e:
                logger.exception("Evidence processing error: %s", _e)
                status_box.update(label="Evidence processing failed", state="error")
                st.error("An unexpected error occurred during evidence processing. Please try again.")
                return

            # 4. Verification Synthesis (Gemini or Heuristics)
            st.write("🧠 **Stage 4:** Synthesizing evidence and computing uncertainty...")
            try:
                engine = VerificationEngine(processor=processor)
                result = engine.assess_claim(extracted, processed_bundle.unique_items)
            except Exception as _e:
                logger.exception("Verification engine error: %s", _e)
                status_box.update(label="Verification synthesis failed", state="error")
                st.error(
                    "The verification engine encountered a network error (the AI API connection was interrupted). "
                    "This is usually temporary — please click **Verify Claim** again to retry."
                )
                return

            # 5. Database Persistence
            st.write("💾 **Stage 5:** Persisting audit record to local SQLite database...")
            try:
                db.save_full_pipeline_audit(result)
            except Exception as e:
                logger.error("Failed to persist verification audit: %s", e)

            status_box.update(label="Verification Pipeline Complete!", state="complete", expanded=False)

        # ----------------- UI RESULTS DISPLAY -----------------
        st.divider()

        # Display Extracted Claim Box
        st.subheader("1. Extracted Claim Proposition")
        col_c1, col_c2 = st.columns([3, 2])
        with col_c1:
            st.markdown(f"**Cleaned Claim:** {result.claim.cleaned_claim}")
            if result.claim.source_url:
                st.markdown(f"**Source URL:** [{result.claim.source_url}]({result.claim.source_url})")
            if result.claim.additional_claims:
                st.markdown(f"**Secondary Propositions:** {', '.join(result.claim.additional_claims)}")
        with col_c2:
            _entities = getattr(result.claim, "detected_entities", []) or []
            _dates = getattr(result.claim, "detected_dates", []) or []
            _locations = getattr(result.claim, "detected_locations", []) or []
            if _entities:
                st.markdown(f"**Entities:** {', '.join(_entities)}")
            if _dates:
                st.markdown(f"**Temporal Markers:** {', '.join(_dates)}")
            if _locations:
                st.markdown(f"**Locations:** {', '.join(_locations)}")

        # Display Assessment Verdict
        st.subheader("2. Assessment Verdict & Explainability")
        badge_style = {
            VerificationStatus.LIKELY_RELIABLE: ("badge-reliable", "🟢 LIKELY RELIABLE", "#065F46"),
            VerificationStatus.NEEDS_VERIFICATION: ("badge-needs-verification", "🟡 NEEDS VERIFICATION", "#92400E"),
            VerificationStatus.POTENTIAL_FAKE: ("badge-fake", "🔴 POTENTIAL FAKE / MISINFORMATION", "#991B1B"),
        }.get(result.status, ("badge-needs-verification", result.status.value, "#334155"))

        st.markdown(
            textwrap.dedent(
                f"""
                <div class="badge-card {badge_style[0]}">
                    <div class="status-title" style="color: {badge_style[2]};">{badge_style[1]}</div>
                    <div style="margin-top: 0.3rem; font-size: 1rem; color: #1E293B;">
                        <strong>Summary:</strong> {result.summary or result.explanation}
                    </div>
                </div>
                """
            ).strip(),
            unsafe_allow_html=True,
        )

        # Metrics: Confidence, Uncertainty, Engine
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Confidence Level", f"{result.confidence_score * 100:.0f}%")
        col_m2.metric("Estimated Uncertainty", f"{result.uncertainty_score * 100:.0f}%")
        col_m3.metric("Evidence Sources", len(result.evidence_items))
        col_m4.metric("Verification Engine", result.model_used.split("(")[0].strip() if result.model_used else "Hybrid")

        st.markdown(f"**Detailed Reasoning:** {result.reasoning or result.explanation}")

        # Display Categorized Evidence Sections
        st.subheader("3. Evidence Audit & Sources")
        evidence_by_id = {item.id: item for item in result.evidence_items}

        supporting = [evidence_by_id[eid] for eid in result.supporting_evidence_ids if eid in evidence_by_id]
        contradicting = [evidence_by_id[eid] for eid in result.contradicting_evidence_ids if eid in evidence_by_id]
        contextual = [evidence_by_id[eid] for eid in result.mixed_or_contextual_evidence_ids if eid in evidence_by_id]
        cited_in_verdict = supporting + contradicting

        if result.evidence_items:
            tab0, tab1, tab2, tab3, tab4 = st.tabs([
                f"Active Verdict Citations ({len(cited_in_verdict)})",
                f"Supporting ({len(supporting)})",
                f"Contradicting ({len(contradicting)})",
                f"Contextual ({len(contextual)})",
                f"All Discovered Sources ({len(result.evidence_items)})",
            ])
            with tab0:
                st.caption("Evidence items explicitly cited by the AI verification engine to justify this verdict.")
                render_evidence_cards("Evidence Directly Justifying Verdict", "tag-justification", "CITED", cited_in_verdict, is_verdict_justification=True)
            with tab1:
                render_evidence_cards("Corroborating Evidence", "tag-support", "SUPPORT", supporting)
            with tab2:
                render_evidence_cards("Contradicting / Debunk Evidence", "tag-contradict", "CONTRADICT", contradicting)
            with tab3:
                render_evidence_cards("Contextual / Neutral Evidence", "tag-context", "CONTEXT", contextual)
            with tab4:
                st.caption("All candidate news headlines and web sources discovered during multi-source retrieval.")
                render_evidence_cards("All Gathered Discovered Sources", "tag-context", "SOURCE", result.evidence_items)
        else:
            st.info(
                "No external evidence was gathered for this claim. Under Mis-Info Verify's evidence-first "
                "principles, this results in **NEEDS VERIFICATION** with elevated uncertainty."
            )

        # Display Limitations Section
        st.subheader("4. Limitations & Caveats")
        limitations_msg = result.limitations or "No explicit limitations noted for this check."
        st.markdown(
            f"""
            > ⚠️ **Methodology Note:** {limitations_msg}  \n
            > *Mis-Info Verify does not guarantee infallible truth. Unverified claims must not be labeled false without primary counter-evidence.*
            """
        )

        # Export Report Section
        st.subheader("5. Export Audit Report")
        reporter = ReportGenerator()
        md_report = reporter.generate_markdown(result)
        json_report = result.model_dump_json(indent=2)

        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.download_button(
                label="📥 Download Markdown Report (.md)",
                data=md_report,
                file_name=f"misinfo_report_{result.status.name.lower()}.md",
                mime="text/markdown",
                use_container_width=True,
            )
        with col_d2:
            st.download_button(
                label="📥 Download JSON Audit (.json)",
                data=json_report,
                file_name=f"misinfo_report_{result.status.name.lower()}.json",
                mime="application/json",
                use_container_width=True,
            )


if __name__ == "__main__":
    main()
