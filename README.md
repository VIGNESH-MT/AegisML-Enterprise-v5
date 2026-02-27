# AegisML Enterprise
### Production ML Reliability & Drift Assurance Engine

> **Enterprise-grade AI governance, drift detection, and risk scoring — ready for UK, EU, UAE and US markets.**

---

## What Is AegisML?

AegisML continuously monitors your machine learning models in production, detecting when they start to fail — **before your customers or regulators notice**.

It answers three critical questions about any ML model:

1. **Is it drifting?** Has the data your model sees changed since training?
2. **Is it miscalibrated?** Can you trust its confidence scores?
3. **What's the risk?** Should this model be deployed, monitored, or shut down?

---

## Why AegisML — The Business Case

| Problem | Without AegisML | With AegisML |
|---------|----------------|--------------|
| Model drift | Discovered after customer complaints | Detected automatically, before impact |
| Regulatory audit | Manual evidence gathering (weeks) | One-click PDF audit report (minutes) |
| EU AI Act compliance | Unknown exposure | Article 9/13/15/17 mapped automatically |
| Model failure | Silent degradation | Slack/webhook alert within minutes |
| Governance documentation | Scattered spreadsheets | SHA-256 integrity-stamped JSON reports |

---

## Market Positioning & Pricing

### Target Markets

| Region | Sectors | Price | Notes |
|--------|---------|-------|-------|
| 🇬🇧 **UK** (London, Edinburgh, Manchester) | Fintech, Insurance, NHS, Retail Banking | £15,000–£18,000/year | FCA Consumer Duty + ICO compliant |
| 🇪🇺 **EU** (Germany, Netherlands, Denmark, Ireland, France, Austria, Finland) | Enterprise software, Finance, Healthcare | €16,000–€22,000/year | EU AI Act Article 9 mandatory for high-risk AI |
| 🇦🇪 **UAE** (Dubai, Abu Dhabi, Sharjah) + 🇸🇦 **Saudi Arabia** | DIFC Finance, Government AI, Vision 2030 | $25,000–$40,000/year | Relationship selling, POC-first |
| 🇺🇸 **USA** (New York, San Francisco, Chicago) | Mid-market SaaS, Healthcare AI, Regional banks | $20,000–$30,000/year | NIST AI RMF aligned |

### Recommended Pricing Tiers

| Tier | Name | Annual Price (USD) | Included |
|------|------|--------------------|---------|
| **Starter** | AegisML Core | **$9,999/year** | 5 models, email alerts, basic dashboard |
| **Professional** | AegisML Pro | **$19,999/year** | 20 models, Slack/PagerDuty, 99.9% SLA |
| **Enterprise** | AegisML Enterprise | **$39,999/year** | Unlimited models, on-premise, dedicated support |
| **Sovereign** | AegisML Sovereign | **$65,000+/year** | Air-gapped, Arabic UI, government compliance |

### Key Sales Angles by Region

**UK (FCA / ICO):**
> *"The FCA's Consumer Duty requires you to evidence your AI produces fair outcomes. AegisML gives you that evidence — automatically, with every audit run."*

**EU (EU AI Act):**
> *"EU AI Act Article 9 requires a risk management system for high-risk AI. AegisML IS that system — automatically generating the documentation you'll need for conformity assessment."*

**UAE / Saudi Arabia:**
> *"AegisML helps you demonstrate AI governance maturity to DIFC regulators and align with UAE National AI Strategy 2031 principles of transparency and accountability."*

**USA (NIST / Healthcare):**
> *"NIST AI RMF GOVERN, MAP, MEASURE, MANAGE — AegisML operationalises all four functions, giving your compliance team audit-ready evidence without the manual effort."*

---

## Technical Features

### Core Drift Detection
- **PSI (Population Stability Index)** — industry-standard distribution shift metric
- **KS Test (Kolmogorov-Smirnov)** — statistical significance of drift
- **Feature-level drift** — per-feature PSI heatmap
- **Prediction drift** — KL divergence + PSI on model outputs

### Calibration Analysis
- **ECE (Expected Calibration Error)** — overall confidence reliability
- **MCE (Maximum Calibration Error)** — worst single-bin error
- **3D Calibration Surface** — interactive Plotly visualisation
- **Reliability Diagram** — visual trust assessment

### Risk Scoring
- **0–100 composite risk score** — calibration (35%), Brier (25%), accuracy (20%), drift (20%)
- **Four tiers:** LOW · MEDIUM · HIGH · CRITICAL
- **Deployment verdict** with specific remediation recommendations

### Bias & Fairness
- **Statistical Parity Difference** — group outcome rate disparity
- **Disparate Impact Ratio** — EEOC 4/5ths rule compliance
- **Equal Opportunity Difference** — TPR parity across groups
- **Average Odds Difference** — combined fairness metric

### Compliance & Governance
- **EU AI Act** Articles 9, 13, 14, 15, 17 — automated assessment
- **UK AI Framework** (DSIT 2024)
- **FCA Consumer Duty** — PRIN 12 compliance check
- **ICO UK GDPR Article 22** — automated decision-making
- **UAE TDRA AI Ethics Principles**
- **NIST AI RMF 1.0** — GOVERN/MAP/MEASURE/MANAGE

### Reporting
- **PDF audit reports** in 13 languages (English, German, French, Spanish, Italian, Polish, Romanian, Dutch, Russian, Ukrainian, Hindi, Urdu, Arabic)
- **SHA-256 integrity hash** — tamper-evident reports
- **JSON structured output** — CI/CD pipeline integration

---

## Installation

### Requirements
- Python 3.11+
- Docker (optional, recommended for production)

### Quick Start

```bash
# Clone the repository
git clone https://github.com/your-org/AegisML-Enterprise.git
cd AegisML-Enterprise

# Install dependencies
pip install -r requirements.txt

# Generate test datasets
python create_test_data.py

# Start the FastAPI backend
uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload

# In a new terminal, start the Streamlit dashboard
streamlit run streamlit_app.py
```

### Docker (Production)

```bash
# Copy and configure environment
cp .env.example .env
# Edit .env with your SECRET_KEY, SLACK_WEBHOOK_URL, etc.

# Build and start all services
docker compose up -d

# API:       http://localhost:8000
# Dashboard: http://localhost:8501
# API Docs:  http://localhost:8000/docs
```

---

## API Usage

### Authentication

All API endpoints require either:

**Option A — API Key (header):**
```
X-AegisML-API-Key: aegis-demo-key-uk-001
```

**Option B — JWT Bearer:**
```
Authorization: Bearer <your-jwt-token>
```

### Example: Audit Upload

```bash
curl -X POST "http://localhost:8000/audit-upload" \
  -H "X-AegisML-API-Key: aegis-demo-key-uk-001" \
  -F "train_file=@examples/stable_train.csv" \
  -F "prod_file=@examples/drifted_prod.csv" \
  -F "column_name=credit_score" \
  -F "language=English" \
  -F "model_name=CreditRiskModel-v2" \
  --output audit_report.pdf
```

### Example: Quick JSON Risk Check (CI/CD)

```bash
curl -X POST "http://localhost:8000/audit-quick" \
  -H "X-AegisML-API-Key: aegis-demo-key-uk-001" \
  -F "train_file=@examples/stable_train.csv" \
  -F "prod_file=@examples/stable_prod.csv" \
  -F "column_name=credit_score"
```

**Response:**
```json
{
  "request_id": "a1b2c3d4-...",
  "risk_tier": "LOW",
  "risk_score": 12.4,
  "psi": 0.031,
  "ks_pvalue": 0.743,
  "status": "PASS",
  "required_action": "Routine monitoring — no immediate action required."
}
```

---

## Project Structure

```
AegisML-Enterprise/
├── api_server.py              # FastAPI backend — JWT auth, rate limiting, webhooks
├── streamlit_app.py           # Streamlit dashboard — 3D visualisations
├── create_test_data.py        # Synthetic dataset generator
├── test_core.py               # Full test suite (pytest)
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Production container
├── docker-compose.yml         # Full stack deployment
├── .env.example               # Environment configuration template
│
├── reliability/               # Core ML reliability engine
│   ├── __init__.py
│   ├── metrics.py             # Accuracy, F1, AUC-ROC, Brier, MCC, log_loss
│   ├── calibration.py         # ECE, MCE, reliability diagram, 3D surface
│   ├── drift.py               # PSI, KL divergence, feature drift heatmap
│   ├── risk.py                # Weighted risk scoring + EU AI Act mapping
│   └── report.py              # Professional PDF report generator
│
└── verification/              # Enterprise verification pipeline
    ├── __init__.py
    ├── verification_runner.py # Full test suite orchestrator
    ├── bias.py                # SPD, DIR, EOD, AOD — fairness metrics
    ├── governance.py          # EU AI Act, UK AI, FCA, ICO, UAE, NIST mapping
    ├── metrics.py             # PSI, KS Test
    ├── risk_scoring.py        # PSI/KS/Bias composite risk score
    ├── config.py              # Deterministic seeds, constants
    ├── pdf_report.py          # 13-language PDF generator
    └── scenarios/             # Test scenario data generators
```

---

## Roadmap

- [ ] Real-time streaming drift detection (Kafka integration)
- [ ] Model registry integration (MLflow, SageMaker)
- [ ] Automated retraining trigger API
- [ ] White-label dashboard for resellers
- [ ] Kubernetes Helm chart
- [ ] Prometheus metrics endpoint
- [ ] SCIM/SSO enterprise auth (Okta, Azure AD)

---

## Compliance Summary

| Standard | Coverage | Status |
|----------|---------|--------|
| EU AI Act 2024 | Articles 9, 13, 14, 15, 17 | ✅ Automated assessment |
| UK AI Framework (DSIT) | 6 core principles | ✅ Covered |
| FCA Consumer Duty (PRIN 12) | Financial AI fairness | ✅ Sector check |
| ICO UK GDPR Article 22 | Automated decision-making | ✅ Covered |
| NIST AI RMF 1.0 | GOVERN/MAP/MEASURE/MANAGE | ✅ All 4 functions |
| UAE TDRA AI Ethics | 5 core principles | ✅ Covered |
| ISO 27001 | Audit trail + integrity hash | ✅ SOC2-ready logging |

---

## Support & Licensing

**Enterprise Support:** enterprise@aegisml.io
**Documentation:** https://docs.aegisml.io
**Status Page:** https://status.aegisml.io

Licensed under the AegisML Commercial Enterprise Licence.
Unauthorised redistribution prohibited.

---

*Built with Python, FastAPI, Streamlit, ReportLab, and Plotly.*
*Aligned to EU AI Act (2024), UK DSIT AI Framework, FCA Consumer Duty, and NIST AI RMF 1.0.*
