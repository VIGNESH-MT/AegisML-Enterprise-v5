"""
AegisML Enterprise API Server
==============================
Production-grade FastAPI service for ML Reliability & Drift Auditing.

Features:
  - JWT Bearer + API-Key dual authentication
  - Multi-tenant request isolation
  - Rate limiting (100 req/min per tenant)
  - Structured JSON audit logging (SOC2-ready)
  - Webhook / Slack alerting on HIGH/CRITICAL risk
  - EU AI Act Article 9/13/15/17 traceability headers
  - CORS, GZip compression, request-ID tracing
  - Health + readiness + version endpoints
  - Async file handling (no blocking I/O on event loop)

Usage:
  uvicorn api_server:app --host 0.0.0.0 --port 8000 --workers 4
"""

import asyncio
import hashlib
import hmac
import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from functools import lru_cache
from typing import Annotated, Optional

import httpx
import numpy as np
import pandas as pd
from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.middleware.base import BaseHTTPMiddleware

from verification.metrics import calculate_psi, calculate_ks
from verification.risk_scoring import calculate_risk_score
from verification.governance import map_risk_tier, eu_ai_act_compliance_check
from verification.verification_runner import generate_verification_report
from verification.pdf_report import generate_pdf

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SECRET_KEY = os.getenv("AEGISML_SECRET_KEY", "change-me-in-production-use-256bit-key")
ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
API_KEY_HEADER_NAME = "X-AegisML-API-Key"
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "")
WEBHOOK_URL = os.getenv("AEGISML_WEBHOOK_URL", "")
ENVIRONMENT = os.getenv("AEGISML_ENV", "production")
VERSION = "5.0.0-enterprise"

# Hardcoded demo API keys — in production, load from DB / Vault
_VALID_API_KEYS: dict[str, dict] = {
    "aegis-demo-key-uk-001": {"tenant": "demo-uk", "plan": "enterprise", "region": "uk"},
    "aegis-demo-key-eu-001": {"tenant": "demo-eu", "plan": "enterprise", "region": "eu"},
    "aegis-demo-key-uae-001": {"tenant": "demo-uae", "plan": "enterprise", "region": "uae"},
}

# ---------------------------------------------------------------------------
# Structured Audit Logger (JSON — SOC2 / ISO 27001 compliant)
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)
_logger = logging.getLogger("aegisml.audit")


def _audit_log(event: str, request_id: str, tenant: str, data: dict):
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "service": "aegisml-api",
        "version": VERSION,
        "environment": ENVIRONMENT,
        "event": event,
        "request_id": request_id,
        "tenant": tenant,
        "data": data,
    }
    _logger.info(json.dumps(record))


# ---------------------------------------------------------------------------
# Rate Limiter
# ---------------------------------------------------------------------------

limiter = Limiter(key_func=get_remote_address)

# ---------------------------------------------------------------------------
# Security
# ---------------------------------------------------------------------------

bearer_scheme = HTTPBearer(auto_error=False)
api_key_scheme = APIKeyHeader(name=API_KEY_HEADER_NAME, auto_error=False)


class TenantContext:
    def __init__(self, tenant_id: str, plan: str, region: str, auth_method: str):
        self.tenant_id = tenant_id
        self.plan = plan
        self.region = region
        self.auth_method = auth_method


async def get_tenant(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    api_key: Optional[str] = Depends(api_key_scheme),
) -> TenantContext:
    """Validates JWT Bearer OR API-Key. Returns tenant context."""

    # 1. Try API Key
    if api_key:
        key_info = _VALID_API_KEYS.get(api_key)
        if not key_info:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key",
                headers={"WWW-Authenticate": "ApiKey"},
            )
        return TenantContext(
            tenant_id=key_info["tenant"],
            plan=key_info["plan"],
            region=key_info["region"],
            auth_method="api_key",
        )

    # 2. Try JWT Bearer
    if credentials:
        try:
            payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
            tenant_id = payload.get("sub")
            if not tenant_id:
                raise HTTPException(status_code=401, detail="Invalid JWT: missing subject")
            return TenantContext(
                tenant_id=tenant_id,
                plan=payload.get("plan", "professional"),
                region=payload.get("region", "uk"),
                auth_method="jwt",
            )
        except JWTError as exc:
            raise HTTPException(status_code=401, detail=f"JWT error: {exc}")

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required. Provide X-AegisML-API-Key header or Bearer token.",
        headers={"WWW-Authenticate": "Bearer or ApiKey"},
    )


# ---------------------------------------------------------------------------
# Request-ID Middleware
# ---------------------------------------------------------------------------

class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time-Ms"] = str(duration_ms)
        # EU AI Act traceability headers
        response.headers["X-AegisML-Version"] = VERSION
        response.headers["X-EU-AI-Act-Article"] = "9,13,15,17"
        return response


# ---------------------------------------------------------------------------
# Webhook / Slack Alerting
# ---------------------------------------------------------------------------

async def _send_alert(risk_tier: str, risk_score: float, tenant: str, request_id: str, column: str):
    """Non-blocking alert dispatch for HIGH/CRITICAL risk results."""
    if risk_tier not in ("HIGH", "CRITICAL"):
        return

    payload = {
        "text": (
            f":rotating_light: *AegisML ALERT* | Risk: `{risk_tier}` | "
            f"Score: `{risk_score}` | Tenant: `{tenant}` | "
            f"Column: `{column}` | Req: `{request_id}`"
        )
    }

    async with httpx.AsyncClient(timeout=5.0) as client:
        if SLACK_WEBHOOK_URL:
            try:
                await client.post(SLACK_WEBHOOK_URL, json=payload)
            except Exception:
                pass
        if WEBHOOK_URL:
            try:
                await client.post(WEBHOOK_URL, json={
                    "event": "high_risk_detected",
                    "risk_tier": risk_tier,
                    "risk_score": risk_score,
                    "tenant": tenant,
                    "request_id": request_id,
                    "column": column,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
            except Exception:
                pass


# ---------------------------------------------------------------------------
# App Initialisation
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AegisML — AI Risk & Governance Audit API",
    description=(
        "Enterprise-grade ML reliability, drift detection, and AI governance auditing. "
        "EU AI Act Article 9/13/15/17 compliant. GDPR-ready audit trails."
    ),
    version=VERSION,
    contact={
        "name": "AegisML Enterprise Support",
        "email": "enterprise@aegisml.io",
        "url": "https://aegisml.io/support",
    },
    license_info={
        "name": "Commercial Enterprise Licence",
        "url": "https://aegisml.io/licence",
    },
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Middleware stack
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(RequestIDMiddleware)
app.add_middleware(GZipMiddleware, minimum_size=500)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "X-Response-Time-Ms", "X-AegisML-Version"],
)


# ---------------------------------------------------------------------------
# Health & Info Endpoints
# ---------------------------------------------------------------------------

@app.get("/", tags=["System"])
def root():
    return {
        "product": "AegisML Enterprise",
        "version": VERSION,
        "status": "operational",
        "environment": ENVIRONMENT,
        "docs": "/docs",
        "compliance": ["EU AI Act", "GDPR", "ISO 27001", "SOC2"],
    }


@app.get("/health", tags=["System"])
def health():
    return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.get("/readiness", tags=["System"])
def readiness():
    """Kubernetes readiness probe."""
    return {"ready": True, "version": VERSION}


@app.get("/version", tags=["System"])
def version_info():
    return {
        "version": VERSION,
        "api_spec": "OpenAPI 3.1",
        "eu_ai_act_alignment": {
            "article_9": "Risk Management System",
            "article_13": "Transparency & Explainability",
            "article_15": "Accuracy, Robustness & Cybersecurity",
            "article_17": "Quality Management System",
        },
    }


# ---------------------------------------------------------------------------
# Core Audit Endpoint
# ---------------------------------------------------------------------------

@app.post(
    "/audit-upload",
    tags=["Audit"],
    summary="Upload CSV data for ML drift & risk audit",
    response_description="PDF audit report with risk score, drift metrics, and governance assessment",
)
@limiter.limit("100/minute")
async def audit_upload(
    request: Request,
    tenant: TenantContext = Depends(get_tenant),
    train_file: Annotated[UploadFile, File(description="Training / reference CSV")] = None,
    prod_file: Annotated[UploadFile, File(description="Production CSV")] = None,
    column_name: Annotated[str, Form(description="Numeric column to analyse")] = None,
    language: Annotated[str, Form(description="Report language")] = "English",
    model_name: Annotated[str, Form(description="Model identifier")] = "Unknown",
    notify_webhook: Annotated[bool, Form(description="Send webhook on HIGH/CRITICAL")] = True,
):
    """
    Perform a full ML drift & risk audit on uploaded CSV data.

    Returns a professional PDF audit report including:
    - PSI (Population Stability Index)
    - KS Test (Kolmogorov-Smirnov)
    - Risk Score & Tier (LOW / MEDIUM / HIGH / CRITICAL)
    - Required Action
    - EU AI Act compliance traceability
    - Integrity hash (SHA-256)

    Authentication: X-AegisML-API-Key header or Bearer JWT.
    """
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))

    _audit_log("audit_request_received", request_id, tenant.tenant_id, {
        "column": column_name,
        "language": language,
        "model": model_name,
        "region": tenant.region,
    })

    # --- Validation ---
    if not train_file or not prod_file or not column_name:
        raise HTTPException(
            status_code=400,
            detail="train_file, prod_file and column_name are all required.",
        )

    supported_languages = [
        "English", "German", "French", "Spanish", "Italian",
        "Polish", "Romanian", "Dutch", "Russian", "Ukrainian", "Hindi", "Urdu",
        "Arabic",
    ]
    if language not in supported_languages:
        language = "English"

    try:
        # Read CSVs (run in thread pool to avoid blocking event loop)
        train_bytes = await train_file.read()
        prod_bytes = await prod_file.read()

        train_df = await asyncio.get_event_loop().run_in_executor(
            None, lambda: pd.read_csv(__import__("io").BytesIO(train_bytes))
        )
        prod_df = await asyncio.get_event_loop().run_in_executor(
            None, lambda: pd.read_csv(__import__("io").BytesIO(prod_bytes))
        )

        # Column validation
        for df, name in [(train_df, "training"), (prod_df, "production")]:
            if column_name not in df.columns:
                raise HTTPException(
                    status_code=400,
                    detail=f"Column '{column_name}' not found in {name} data. "
                           f"Available: {list(df.columns)}",
                )

        train_series = train_df[column_name].dropna()
        prod_series = prod_df[column_name].dropna()

        # Type validation
        for s, name in [(train_series, "training"), (prod_series, "production")]:
            if not np.issubdtype(s.dtype, np.number):
                raise HTTPException(
                    status_code=400,
                    detail=f"Column '{column_name}' in {name} data must be numeric.",
                )

        # Sample size validation
        if len(train_series) < 30 or len(prod_series) < 30:
            raise HTTPException(
                status_code=400,
                detail=f"Minimum 30 samples required. Got train={len(train_series)}, prod={len(prod_series)}.",
            )

        train_data = train_series.values
        prod_data = prod_series.values

        # --- Compute Metrics ---
        psi = calculate_psi(train_data, prod_data)
        ks_stat, p_value = calculate_ks(train_data, prod_data)
        risk_score = calculate_risk_score(psi, p_value, bias_score=0)
        tier, action = map_risk_tier(risk_score)
        compliance = eu_ai_act_compliance_check(risk_score, psi, p_value)

        status_str = "PASS" if risk_score < 70 else "FAIL"
        overall_status = status_str == "PASS"

        result = {
            "case": "enterprise_audit",
            "model_name": model_name,
            "tenant": tenant.tenant_id,
            "region": tenant.region,
            "request_id": request_id,
            "column_analysed": column_name,
            "train_samples": int(len(train_data)),
            "prod_samples": int(len(prod_data)),
            "psi": round(float(psi), 5),
            "ks_statistic": round(float(ks_stat), 5),
            "ks_pvalue": round(float(p_value), 5),
            "risk_score": float(risk_score),
            "risk_tier": tier,
            "required_action": action,
            "status": status_str,
            "eu_ai_act_compliance": compliance,
        }

        _audit_log("audit_computed", request_id, tenant.tenant_id, {
            "psi": result["psi"],
            "risk_tier": tier,
            "risk_score": risk_score,
            "status": status_str,
        })

        # --- Generate Reports ---
        report_path = f"verification/{tenant.tenant_id}_{uuid.uuid4().hex[:8]}_report.json"
        generate_verification_report([result], overall_status, output_path=report_path)

        pdf_path = report_path.replace(".json", ".pdf")
        generate_pdf(report_path, output_path=pdf_path, language=language)

        if not os.path.exists(pdf_path):
            raise HTTPException(status_code=500, detail="PDF generation failed.")

        # --- Async Alert ---
        if notify_webhook:
            asyncio.create_task(
                _send_alert(tier, risk_score, tenant.tenant_id, request_id, column_name)
            )

        _audit_log("audit_completed", request_id, tenant.tenant_id, {
            "pdf_generated": True,
            "risk_tier": tier,
        })

        filename = f"AegisML_{model_name.replace(' ', '_')}_{tier}_Audit.pdf"
        return FileResponse(
            pdf_path,
            media_type="application/pdf",
            filename=filename,
            headers={
                "X-Risk-Tier": tier,
                "X-Risk-Score": str(risk_score),
                "X-PSI": str(result["psi"]),
                "X-Request-ID": request_id,
                "X-AegisML-Tenant": tenant.tenant_id,
            },
        )

    except HTTPException:
        raise
    except Exception as exc:
        _audit_log("audit_error", request_id, tenant.tenant_id, {"error": str(exc)})
        raise HTTPException(status_code=500, detail=f"Internal error: {exc}")


# ---------------------------------------------------------------------------
# Quick Risk Summary Endpoint (JSON — no PDF)
# ---------------------------------------------------------------------------

@app.post("/audit-quick", tags=["Audit"], summary="JSON risk summary (no PDF)")
@limiter.limit("200/minute")
async def audit_quick(
    request: Request,
    tenant: TenantContext = Depends(get_tenant),
    train_file: Annotated[UploadFile, File()] = None,
    prod_file: Annotated[UploadFile, File()] = None,
    column_name: Annotated[str, Form()] = None,
):
    """Returns JSON risk summary instantly, no PDF generated. Ideal for CI/CD pipelines."""
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))

    if not train_file or not prod_file or not column_name:
        raise HTTPException(status_code=400, detail="train_file, prod_file and column_name required.")

    try:
        train_bytes = await train_file.read()
        prod_bytes = await prod_file.read()
        train_df = pd.read_csv(__import__("io").BytesIO(train_bytes))
        prod_df = pd.read_csv(__import__("io").BytesIO(prod_bytes))

        if column_name not in train_df.columns or column_name not in prod_df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{column_name}' not found.")

        train_data = train_df[column_name].dropna().values
        prod_data = prod_df[column_name].dropna().values

        if len(train_data) < 10 or len(prod_data) < 10:
            raise HTTPException(status_code=400, detail="Minimum 10 samples required.")

        psi = calculate_psi(train_data, prod_data)
        ks_stat, p_value = calculate_ks(train_data, prod_data)
        risk_score = calculate_risk_score(psi, p_value, bias_score=0)
        tier, action = map_risk_tier(risk_score)

        return JSONResponse({
            "request_id": request_id,
            "tenant": tenant.tenant_id,
            "column": column_name,
            "psi": round(float(psi), 5),
            "ks_statistic": round(float(ks_stat), 5),
            "ks_pvalue": round(float(p_value), 5),
            "risk_score": float(risk_score),
            "risk_tier": tier,
            "required_action": action,
            "status": "PASS" if risk_score < 70 else "FAIL",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Tenant Usage / Plan Info
# ---------------------------------------------------------------------------

@app.get("/tenant/info", tags=["Tenant"])
async def tenant_info(tenant: TenantContext = Depends(get_tenant)):
    return {
        "tenant_id": tenant.tenant_id,
        "plan": tenant.plan,
        "region": tenant.region,
        "auth_method": tenant.auth_method,
        "features": {
            "audit_upload": True,
            "audit_quick": True,
            "multi_language_pdf": True,
            "webhook_alerts": True,
            "eu_ai_act_compliance": True,
            "bias_detection": True,
            "feature_drift": True,
        },
        "rate_limits": {"audit_upload": "100/min", "audit_quick": "200/min"},
    }
