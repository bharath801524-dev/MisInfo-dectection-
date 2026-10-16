"""
Unit tests for Mis-Info Verify Verification Engine module.
"""

from src import VerificationStatus, ExtractedClaim, EvidenceItem
from src.verification_engine import VerificationEngine


def test_empty_evidence_never_labels_claim_as_fake() -> None:
    """CRITICAL REQUIREMENT: Never label a claim false merely because evidence was not found.

    Absence of evidence must return NEEDS_VERIFICATION with high uncertainty.
    """
    engine = VerificationEngine()
    claim = ExtractedClaim(
        raw_input="Alien spacecraft landed in Antarctica yesterday.",
        cleaned_claim="Alien spacecraft landed in Antarctica yesterday.",
    )

    result = engine.assess_claim(claim, evidence_items=[])

    # Must be NEEDS_VERIFICATION, not POTENTIAL_FAKE
    assert result.status == VerificationStatus.NEEDS_VERIFICATION
    assert result.uncertainty_score >= 0.8
    assert "No corroborating or refuting external evidence was found" in result.explanation
    assert len(result.evidence_items) == 0
    assert len(result.source_urls) == 0


def test_refuting_evidence_flags_potential_fake() -> None:
    engine = VerificationEngine()
    claim = ExtractedClaim(
        raw_input="Drinking chlorine cures the common cold.",
        cleaned_claim="Drinking chlorine cures the common cold.",
    )
    evidence = [
        EvidenceItem(
            source_name="FactCheck Org",
            url="https://factcheck.org/chlorine-claim",
            snippet="Health experts debunk claims that chlorine cures cold; this claim is false and dangerous.",
            reliability_score=0.9,
        ),
        EvidenceItem(
            source_name="Snopes",
            url="https://snopes.com/fact-check/chlorine-cold",
            snippet="Rating: FALSE. This viral claim has been repeatedly debunked by medical bodies.",
            reliability_score=0.9,
        ),
    ]

    result = engine.assess_claim(claim, evidence)

    assert result.status == VerificationStatus.POTENTIAL_FAKE
    assert result.confidence_score >= 0.7
    assert result.uncertainty_score <= 0.3
    assert len(result.source_urls) == 2
    assert "flag this claim as inaccurate, debunked, or misleading" in result.explanation


def test_corroborating_evidence_flags_likely_reliable() -> None:
    engine = VerificationEngine()
    claim = ExtractedClaim(
        raw_input="James Webb Telescope captures clearest view of Neptune's rings.",
        cleaned_claim="James Webb Telescope captures clearest view of Neptune's rings.",
    )
    evidence = [
        EvidenceItem(
            source_name="Reuters",
            url="https://reuters.com/science/jwst-neptune",
            snippet="NASA has confirmed that the James Webb Space Telescope verified clearest view of Neptune rings.",
            reliability_score=0.95,
        ),
        EvidenceItem(
            source_name="AP News",
            url="https://apnews.com/article/jwst-neptune",
            snippet="Astronomers accurate observations confirmed unprecedented clarity of Neptune ring system.",
            reliability_score=0.95,
        ),
    ]

    result = engine.assess_claim(claim, evidence)

    assert result.status == VerificationStatus.LIKELY_RELIABLE
    assert result.confidence_score >= 0.7
    assert result.uncertainty_score <= 0.3
    assert len(result.source_urls) == 2
    assert "Corroborating evidence was identified" in result.explanation


def test_gemini_valid_structured_response() -> None:
    """Mock Gemini structured response and verify proper mapping into VerificationResult."""
    import json
    from unittest.mock import MagicMock

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = json.dumps({
        "status": "LIKELY RELIABLE",
        "summary": "Corroborated by space agency press releases.",
        "reasoning": "Evidence ev_1 directly confirms the exoplanet atmospheric water observation.",
        "supporting_evidence_ids": ["ev_1"],
        "contradicting_evidence_ids": [],
        "mixed_or_contextual_evidence_ids": [],
        "limitations": "Spectroscopy requires secondary instrument confirmation.",
        "confidence": "High",
        "investigated_claim": "JWST detects water on exoplanet",
    })
    mock_client.models.generate_content.return_value = mock_response

    engine = VerificationEngine(gemini_client=mock_client)
    claim = ExtractedClaim(raw_input="JWST detects water on exoplanet", cleaned_claim="JWST detects water on exoplanet")
    evidence = [
        EvidenceItem(
            id="ev_1",
            source_name="nature.com",
            url="https://nature.com/articles/exoplanet-water",
            snippet="JWST spectroscopy confirms atmospheric water signatures.",
        )
    ]

    result = engine.assess_claim(claim, evidence)

    assert result.status == VerificationStatus.LIKELY_RELIABLE
    assert result.summary == "Corroborated by space agency press releases."
    assert result.supporting_evidence_ids == ["ev_1"]
    assert "Google Gemini" in result.model_used


def test_citation_integrity_rejects_hallucinated_ids() -> None:
    """CRITICAL REQUIREMENT: If the model references unknown evidence IDs, they must be rejected server-side."""
    import json
    from unittest.mock import MagicMock

    mock_client = MagicMock()
    mock_response = MagicMock()
    # Model hallucinated 'ev_fake_99' and 'ev_ghost_1'
    mock_response.text = json.dumps({
        "status": "LIKELY RELIABLE",
        "summary": "Supported by sources.",
        "reasoning": "Model cites real ev_1 and hallucinated ev_fake_99.",
        "supporting_evidence_ids": ["ev_1", "ev_fake_99"],
        "contradicting_evidence_ids": ["ev_ghost_1"],
        "mixed_or_contextual_evidence_ids": [],
        "limitations": "None.",
        "confidence": "Medium",
        "investigated_claim": "Sample claim",
    })
    mock_client.models.generate_content.return_value = mock_response

    engine = VerificationEngine(gemini_client=mock_client)
    claim = ExtractedClaim(raw_input="Sample claim", cleaned_claim="Sample claim")
    evidence = [
        EvidenceItem(
            id="ev_1",
            source_name="reuters.com",
            url="https://reuters.com/news/1",
            snippet="Real retrieved snippet.",
        )
    ]

    result = engine.assess_claim(claim, evidence)

    # Hallucinated IDs must be stripped!
    assert "ev_fake_99" not in result.supporting_evidence_ids
    assert result.supporting_evidence_ids == ["ev_1"]
    assert "ev_ghost_1" not in result.contradicting_evidence_ids
    # Intercepted anomalies must be logged into limitations
    assert "Citation Integrity Audit" in result.limitations
    assert "Rejected unknown" in result.limitations


def test_gemini_malformed_response_triggers_deterministic_fallback() -> None:
    """If Gemini returns invalid JSON, engine must fall back safely without crashing."""
    from unittest.mock import MagicMock

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "This is plain text, not valid JSON schema output."
    mock_client.models.generate_content.return_value = mock_response

    engine = VerificationEngine(gemini_client=mock_client)
    claim = ExtractedClaim(raw_input="Mars rover status update", cleaned_claim="Mars rover status update")
    evidence = [
        EvidenceItem(
            id="ev_1",
            source_name="reuters.com",
            url="https://reuters.com/news/1",
            snippet="NASA verified and confirmed mission objectives.",
        )
    ]

    result = engine.assess_claim(claim, evidence)

    # Fallback to deterministic heuristic baseline
    assert result.status == VerificationStatus.LIKELY_RELIABLE
    assert "Baseline" in result.model_used or "Fallback" in result.model_used


# ==============================================================================
# SECTION 5: MOCKED REGRESSION TESTS FOR VERDICT ACCURACY & SAFE DEFAULTS
# ==============================================================================


def test_regression_case_1_true_claim_with_supporting_evidence() -> None:
    """Case 1: A true claim with relevant supporting evidence must NOT receive a misinformation verdict."""
    import json
    from unittest.mock import MagicMock

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = json.dumps({
        "status": "LIKELY RELIABLE",
        "summary": "Corroborated by independent scientific reporting and peer-reviewed releases.",
        "reasoning": "Evidence ev_jwst_1 and ev_jwst_2 confirm atmospheric analysis of exoplanet LHS 475 b.",
        "supporting_evidence_ids": ["ev_jwst_1", "ev_jwst_2"],
        "contradicting_evidence_ids": [],
        "mixed_or_contextual_evidence_ids": [],
        "limitations": "Requires ongoing spectroscopic validation.",
        "confidence": "High",
        "investigated_claim": "JWST confirms Earth-sized exoplanet LHS 475 b",
    })
    mock_client.models.generate_content.return_value = mock_response

    engine = VerificationEngine(gemini_client=mock_client)
    claim = ExtractedClaim(
        raw_input="James Webb Telescope confirms Earth-sized exoplanet LHS 475 b.",
        cleaned_claim="James Webb Telescope confirms Earth-sized exoplanet LHS 475 b.",
    )
    evidence = [
        EvidenceItem(
            id="ev_jwst_1",
            source_name="nasa.gov",
            url="https://nasa.gov/mission_pages/webb/lhs475b",
            snippet="NASA Webb confirms Earth-sized exoplanet LHS 475 b using NIRSpec observations.",
            reliability_score=0.98,
        ),
        EvidenceItem(
            id="ev_jwst_2",
            source_name="nature.com",
            url="https://nature.com/articles/exoplanet-lhs475b",
            snippet="Researchers report James Webb Space Telescope confirmation of LHS 475 b.",
            reliability_score=0.95,
        ),
    ]

    result = engine.assess_claim(claim, evidence)

    # Must NOT receive a misinformation verdict
    assert result.status != VerificationStatus.POTENTIAL_FAKE
    assert result.status == VerificationStatus.LIKELY_RELIABLE
    assert result.confidence_score >= 0.70
    assert result.uncertainty_score <= 0.30
    assert len(result.supporting_evidence_ids) == 2


def test_regression_case_2_false_claim_with_contradictory_evidence() -> None:
    """Case 2: A false claim with credible contradictory evidence flags POTENTIAL FAKE."""
    import json
    from unittest.mock import MagicMock

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = json.dumps({
        "status": "POTENTIAL FAKE / MISINFORMATION",
        "summary": "Debunked by credible public health agencies and fact-checkers.",
        "reasoning": "Evidence ev_fc_1 explicitly refutes the claim, identifying it as a dangerous medical hoax.",
        "supporting_evidence_ids": [],
        "contradicting_evidence_ids": ["ev_fc_1"],
        "mixed_or_contextual_evidence_ids": [],
        "limitations": "Widely circulated on social platforms despite official debunks.",
        "confidence": "High",
        "investigated_claim": "Drinking concentrated bleach cures viral infections",
    })
    mock_client.models.generate_content.return_value = mock_response

    engine = VerificationEngine(gemini_client=mock_client)
    claim = ExtractedClaim(
        raw_input="Drinking concentrated bleach cures viral infections.",
        cleaned_claim="Drinking concentrated bleach cures viral infections.",
    )
    evidence = [
        EvidenceItem(
            id="ev_fc_1",
            source_name="factcheck.org",
            url="https://factcheck.org/bleach-cures-claim",
            snippet="Rating: FALSE. Drinking bleach does not cure viral infections and can cause fatal poisoning.",
            reliability_score=0.92,
        )
    ]

    result = engine.assess_claim(claim, evidence)

    assert result.status == VerificationStatus.POTENTIAL_FAKE
    assert result.confidence_score >= 0.70
    assert result.uncertainty_score <= 0.30
    assert "ev_fc_1" in result.contradicting_evidence_ids


def test_regression_case_3_true_claim_with_irrelevant_search_results() -> None:
    """Case 3: A true claim where search results are irrelevant must default to NEEDS VERIFICATION."""
    import json
    from unittest.mock import MagicMock

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = json.dumps({
        "status": "NEEDS VERIFICATION",
        "summary": "Retrieved articles do not contain information verifying or refuting the bilateral trade accord.",
        "reasoning": "Evidence ev_irr_1 and ev_irr_2 discuss smartphone shipments and tech company earnings, which are irrelevant to the international treaty claim.",
        "supporting_evidence_ids": [],
        "contradicting_evidence_ids": [],
        "mixed_or_contextual_evidence_ids": ["ev_irr_1", "ev_irr_2"],
        "limitations": "Search results yielded unrelated corporate news rather than diplomatic or government sources.",
        "confidence": "Low",
        "investigated_claim": "Country A and Country B sign bilateral semiconductor accord",
    })
    mock_client.models.generate_content.return_value = mock_response

    engine = VerificationEngine(gemini_client=mock_client)
    claim = ExtractedClaim(
        raw_input="Country A and Country B sign bilateral semiconductor accord.",
        cleaned_claim="Country A and Country B sign bilateral semiconductor accord.",
    )
    evidence = [
        EvidenceItem(
            id="ev_irr_1",
            source_name="techcrunch.com",
            url="https://techcrunch.com/smartphone-sales",
            snippet="Smartphone shipments declined by 3 percent this quarter across major vendors.",
            reliability_score=0.75,
        ),
        EvidenceItem(
            id="ev_irr_2",
            source_name="marketwatch.com",
            url="https://marketwatch.com/semiconductor-index",
            snippet="Semiconductor stock index rises slightly ahead of major earnings releases.",
            reliability_score=0.70,
        ),
    ]

    result = engine.assess_claim(claim, evidence)

    # Must NOT label as fake simply because search results were off-topic
    assert result.status != VerificationStatus.POTENTIAL_FAKE
    assert result.status == VerificationStatus.NEEDS_VERIFICATION
    assert result.uncertainty_score >= 0.50


def test_regression_case_4_claim_with_conflicting_sources() -> None:
    """Case 4: A claim with conflicting sources must default to NEEDS VERIFICATION."""
    import json
    from unittest.mock import MagicMock

    mock_client = MagicMock()
    mock_response = MagicMock()
    # Even if model tried to mark POTENTIAL FAKE, server-side conflict resolution must intercept
    mock_response.text = json.dumps({
        "status": "POTENTIAL FAKE / MISINFORMATION",
        "summary": "Conflicting local reports regarding the city council plastic ban.",
        "reasoning": "Source ev_conf_1 says ban was enacted; source ev_conf_2 says mayor halted the rollout.",
        "supporting_evidence_ids": ["ev_conf_1"],
        "contradicting_evidence_ids": ["ev_conf_2"],
        "mixed_or_contextual_evidence_ids": [],
        "limitations": "Direct conflict between local news reports.",
        "confidence": "Medium",
        "investigated_claim": "City council bans plastic bags starting next week",
    })
    mock_client.models.generate_content.return_value = mock_response

    engine = VerificationEngine(gemini_client=mock_client)
    claim = ExtractedClaim(
        raw_input="City council bans plastic bags starting next week.",
        cleaned_claim="City council bans plastic bags starting next week.",
    )
    evidence = [
        EvidenceItem(
            id="ev_conf_1",
            source_name="localgazette.com",
            url="https://localgazette.com/plastic-ban-approved",
            snippet="City council votes 5-2 to implement plastic bag ban starting next Monday.",
            reliability_score=0.80,
        ),
        EvidenceItem(
            id="ev_conf_2",
            source_name="tribunenews.com",
            url="https://tribunenews.com/plastic-ban-delayed",
            snippet="Mayor announces stay on plastic bag ban rollout pending merchant association appeals.",
            reliability_score=0.80,
        ),
    ]

    result = engine.assess_claim(claim, evidence)

    # Material contradiction between credible sources must default to NEEDS VERIFICATION
    assert result.status == VerificationStatus.NEEDS_VERIFICATION
    assert result.uncertainty_score >= 0.50


def test_regression_case_5_empty_evidence_response() -> None:
    """Case 5: An empty evidence response must strictly return NEEDS VERIFICATION with high uncertainty."""
    engine = VerificationEngine()
    claim = ExtractedClaim(
        raw_input="Deep-sea expedition discovers glowing octopuses at 10,000 meters depth.",
        cleaned_claim="Deep-sea expedition discovers glowing octopuses at 10,000 meters depth.",
    )

    result = engine.assess_claim(claim, evidence_items=[])

    assert result.status == VerificationStatus.NEEDS_VERIFICATION
    assert result.uncertainty_score >= 0.80
    assert result.confidence_score <= 0.30
    assert len(result.evidence_items) == 0
    assert "No corroborating or refuting external evidence was found" in result.explanation


def test_regression_case_6_gemini_response_with_invalid_evidence_ids() -> None:
    """Case 6: Gemini response referencing hallucinated/invalid IDs defaults safely to NEEDS VERIFICATION."""
    import json
    from unittest.mock import MagicMock

    mock_client = MagicMock()
    mock_response = MagicMock()
    # Model claimed POTENTIAL FAKE but only cited non-existent IDs
    mock_response.text = json.dumps({
        "status": "POTENTIAL FAKE / MISINFORMATION",
        "summary": "Debunked by non-existent sources.",
        "reasoning": "Claim disproven by ev_phantom_1 and ev_phantom_2.",
        "supporting_evidence_ids": [],
        "contradicting_evidence_ids": ["ev_phantom_1", "ev_phantom_2"],
        "mixed_or_contextual_evidence_ids": [],
        "limitations": "Model relied on hallucinatory identifiers.",
        "confidence": "High",
        "investigated_claim": "Secret tunnel network discovered beneath lake",
    })
    mock_client.models.generate_content.return_value = mock_response

    engine = VerificationEngine(gemini_client=mock_client)
    claim = ExtractedClaim(
        raw_input="Secret tunnel network discovered beneath lake.",
        cleaned_claim="Secret tunnel network discovered beneath lake.",
    )
    evidence = [
        EvidenceItem(
            id="ev_valid_1",
            source_name="geology.org",
            url="https://geology.org/lake-survey",
            snippet="Subterranean geological mapping reveals natural karst limestone formations beneath lake.",
            reliability_score=0.85,
        )
    ]

    result = engine.assess_claim(claim, evidence)

    # Hallucinated IDs are stripped, so no valid contradicting evidence exists -> Must default to NEEDS VERIFICATION
    assert result.status == VerificationStatus.NEEDS_VERIFICATION
    assert len(result.contradicting_evidence_ids) == 0
    assert "Citation Integrity Audit" in result.limitations
    assert "Rejected unknown" in result.limitations
    assert result.uncertainty_score >= 0.50


def test_regression_case_7_api_error_or_malformed_model_response() -> None:
    """Case 7: API error or malformed model output defaults safely to NEEDS VERIFICATION on ambiguous input."""
    from unittest.mock import MagicMock

    # Scenario A: Generation raises an exception (e.g. rate limit, 503, network drop)
    mock_client_error = MagicMock()
    mock_client_error.models.generate_content.side_effect = Exception("503 Service Unavailable: Model Overloaded")

    engine_error = VerificationEngine(gemini_client=mock_client_error)
    claim = ExtractedClaim(
        raw_input="Ancient artifact found during metro construction.",
        cleaned_claim="Ancient artifact found during metro construction.",
    )
    # Neutral/unverified evidence that does not meet the strict heuristic threshold for debunk
    evidence = [
        EvidenceItem(
            id="ev_metro_1",
            source_name="metroblog.com",
            url="https://metroblog.com/construction-diary",
            snippet="Workers pause excavation in sector 4 after uncovering metallic object.",
            reliability_score=0.60,
        )
    ]

    result_error = engine_error.assess_claim(claim, evidence)

    # Must fall back gracefully to NEEDS VERIFICATION, never crash and never produce a fake verdict
    assert result_error.status == VerificationStatus.NEEDS_VERIFICATION
    assert result_error.status != VerificationStatus.POTENTIAL_FAKE
    assert "Gemini API unavailable" in result_error.limitations

    # Scenario B: Generation returns malformed non-JSON output
    mock_client_malformed = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = "<<<MALFORMED RESPONSE: NOT JSON>>>"
    mock_client_malformed.models.generate_content.return_value = mock_resp

    engine_malformed = VerificationEngine(gemini_client=mock_client_malformed)
    result_malformed = engine_malformed.assess_claim(claim, evidence)

    assert result_malformed.status == VerificationStatus.NEEDS_VERIFICATION
    assert result_malformed.status != VerificationStatus.POTENTIAL_FAKE
    assert "Gemini API unavailable" in result_malformed.limitations


