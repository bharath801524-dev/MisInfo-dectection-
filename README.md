# Mis-Info Verify 🛡️

**Mis-Info Verify** is an evidence-first misinformation verification application designed to detect, contextualize, and explain potential misinformation across social media posts, headlines, claims, and URLs using **Tavily Search API** and **Google Gemini AI**.

---

## 🎯 Project Objective & Core Philosophy

When evaluating claims in the digital age, traditional systems often mistakenly label unverified claims as "false" simply because immediate internet evidence was not found. **Mis-Info Verify adheres strictly to an evidence-first principle:**

> **Core Rule:** Never label a claim false merely because evidence was not found. Absence of evidence results in **NEEDS VERIFICATION** with elevated uncertainty, never an automatic verdict of falsehood.

### Three Assessment Outcomes:
1. 🟢 **LIKELY RELIABLE** — Multiple independent, trusted sources corroborate the factual claim without unresolved material contradiction.
2. 🟡 **NEEDS VERIFICATION** — Evidence is absent, inconclusive, conflicting, outdated, or sparse. (Default outcome when evidence is lacking).
3. 🔴 **POTENTIAL FAKE / MISINFORMATION** — Credible fact-checking bodies or authoritative primary sources explicitly refute or debunk the claim.

---

## 🔄 Phase 2 End-to-End Architecture & Data Flow

```text
USER INPUT (Text or URL)
       │
       ▼
1. CLAIM EXTRACTION (src/claim_extractor.py)
   ├── SSRF Protection: Blocks localhost, private IPs, loopbacks
   ├── Text Cleaning & Tweet Sanitization
   └── Entity, Date, Location, & Proposition Segmentation
       │
       ▼
2. REAL EVIDENCE COLLECTION (src/evidence_collector.py)
   ├── Targeted Search Query Generation
   ├── Tavily Search API (Bounded Exponential Retries)
   └── Canonical URL Deduplication & Normalization
       │
       ▼
3. EVIDENCE PROCESSING & QUALITY CHECKS (src/evidence_processor.py)
   ├── Near-Duplicate Snippet Pruning
   ├── Source Diversity vs Single-Domain Checks
   └── Accredited Publisher Credibility Tiering
       │
       ▼
4. EXPLAINABLE AI REASONING (src/verification_engine.py)
   ├── Untrusted Input Sandboxing (Prompt-Injection Defense)
   ├── Google Gemini API (Structured JSON Schema Analysis)
   ├── Strict Server-Side Citation Integrity Validation
   └── Deterministic Fallback if Offline or API Unavailable
       │
       ▼
5. PERSISTENCE & AUDIT LOGGING (src/database.py)
   └── SQLite Storage: Claims, Verifications, & Evidence Linkages
       │
       ▼
6. EXPLAINABLE INTERFACE & EXPORTS (app.py & src/report_generator.py)
   ├── Streamlit Dashboard: Verdict Cards, Reasonings, & Limitations
   ├── Categorized Evidence Tabs: Supporting, Contradicting, Contextual
   └── Instant Report Export: Markdown (.md) and JSON (.json)
```

---

## 🔑 Required API Accounts & Configuration

Mis-Info Verify integrates with two external developer services:

| Service | Purpose | Get Free API Key |
| :--- | :--- | :--- |
| **Tavily Search API** | Real-time fact-checking and web source retrieval | [app.tavily.com](https://app.tavily.com) |
| **Google Gemini API** | Multi-source evidence synthesis & structured reasoning | [aistudio.google.com](https://aistudio.google.com) |

> ⚠️ **API Usage Note:** External API requests may incur costs or be subject to provider rate limits. The application implements exponential backoff retries and graceful offline fallbacks to minimize unnecessary calls.

### Configuring Local Environment Variables
Create a local `.env` file from the provided `.env.example` template:

```powershell
Copy-Item .env.example .env
```

Open `.env` and insert your credentials:

```env
APP_ENV=development
APP_DEBUG=true
LOG_LEVEL=INFO

DATABASE_PATH=data/mindmesh.db
OUTPUT_DIR=outputs

# Phase 2 Search & AI Verification APIs
TAVILY_API_KEY=tvly-xxxxxxxxxxxxxxxxxxxx
GEMINI_API_KEY=AIzaxxxxxxxxxxxxxxxxxxxx
GEMINI_MODEL=gemini-2.5-flash
```

*(If API keys are left unset, the application launches safely in offline heuristic mode).*

---

## 💻 Windows Setup & Installation

### 1. Prerequisites
- **Operating System:** Windows 10/11
- **Python:** Python 3.12 (Supported 64-bit release)
- **Terminal:** PowerShell or Command Prompt

### 2. Open PowerShell in Project Root
```powershell
cd D:\MISINFO_dect
```

### 3. Create and Activate Virtual Environment
```powershell
# Create virtual environment
python -m venv .venv

# Activate the virtual environment
.\.venv\Scripts\Activate.ps1
```

### 4. Install Dependencies
```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

## 🧪 Running the Test Suite

Run the automated test suite (36 tests including all Phase 1 regression tests, unit tests, and mocked end-to-end integration tests):

```powershell
python -m pytest tests -v
```

### Optional: Manual Live API Smoke Test
To test live network calls against actual Tavily and Gemini servers (disabled by default to avoid quota usage):

```powershell
$env:RUN_LIVE_API_TESTS="1"
python -m pytest tests/test_integration_pipeline.py -k "test_manual_live_api_smoke_test" -v
$env:RUN_LIVE_API_TESTS=""
```

---

## 🚀 Launching the Streamlit Application

Start the interactive verification dashboard:
```powershell
streamlit run app.py
```

The application will start on `http://localhost:8501` and open in your default browser.

---

## ⚖️ Limitations & Safe Interpretation

1. **Not Infallible Truth:** Web search results and LLM reasoning represent probabilistic evidence assessments, not absolute factual certainty.
2. **Search Horizon:** Breaking events within the last few minutes may not yet be indexed by search providers.
3. **Citation Integrity Enforcement:** If an AI model references an invented evidence ID, the server-side validator strips the reference and logs an audit notice in the limitations section.
4. **Offline Resilience:** If API quotas are exceeded or internet connectivity is lost, the engine falls back to heuristic baseline evaluation without fabricating data.
