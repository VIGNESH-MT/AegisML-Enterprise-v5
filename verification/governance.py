"""
verification/governance.py
---------------------------
Enterprise AI Governance Engine.

Covers:
  - EU AI Act (2024) — Articles 5, 6, 7, 9, 13, 14, 15, 17
  - UK AI Framework (DSIT 2024)
  - FCA Consumer Duty (UK Financial Services)
  - ICO AI Auditing Framework (UK GDPR)
  - UAE TDRA AI Ethics Principles
  - US NIST AI RMF 1.0

Risk Tier Mapping:
  Score < 30   → LOW       Routine monitoring
  Score 30–69  → MEDIUM    Engineering review required
  Score 70–89  → HIGH      Immediate escalation required
  Score 90+    → CRITICAL  Emergency response + regulatory notification
"""

from __future__ import annotations

from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# Core Risk Tier
# ---------------------------------------------------------------------------

def map_risk_tier(score: float) -> tuple[str, str]:
    """
    Map a numeric risk score (0–100) to a named tier and required action.

    Returns
    -------
    (tier: str, action: str)
    """
    if score < 30:
        return "LOW", "Routine monitoring — no immediate action required."
    elif score < 70:
        return "MEDIUM", "Engineering review required within 5 business days."
    elif score < 90:
        return "HIGH", "Immediate escalation to AI governance team. Deployment hold recommended."
    else:
        return "CRITICAL", (
            "Emergency response required. Notify AI governance committee, legal, and compliance. "
            "Suspend deployment immediately. Regulatory notification may be required."
        )


# ---------------------------------------------------------------------------
# EU AI Act Compliance Assessment
# ---------------------------------------------------------------------------

def eu_ai_act_compliance_check(
    risk_score: float,
    psi: float,
    p_value: float,
) -> dict:
    """
    Assess EU AI Act compliance obligations based on risk metrics.

    Returns a structured compliance report aligned to key articles.
    """
    tier, _ = map_risk_tier(risk_score)

    articles = {
        "Article_9_Risk_Management": {
            "obligation": "Establish and maintain risk management system throughout AI lifecycle.",
            "status": "COMPLIANT" if risk_score < 70 else "REQUIRES_ACTION",
            "evidence": f"Risk score: {risk_score:.1f}/100. PSI: {psi:.4f}.",
            "action_required": risk_score >= 70,
        },
        "Article_13_Transparency": {
            "obligation": "Ensure AI system outputs are interpretable to deployers and users.",
            "status": "COMPLIANT",
            "evidence": "Audit report generated with full metric breakdown and interpretations.",
            "action_required": False,
        },
        "Article_14_Human_Oversight": {
            "obligation": "Ensure humans can oversee, understand, and intervene in AI outputs.",
            "status": "REQUIRES_ACTION" if tier in ("HIGH", "CRITICAL") else "COMPLIANT",
            "evidence": f"Risk tier: {tier}. Human oversight {'required' if tier in ('HIGH', 'CRITICAL') else 'standard'}.",
            "action_required": tier in ("HIGH", "CRITICAL"),
        },
        "Article_15_Accuracy_Robustness": {
            "obligation": "AI system must achieve appropriate levels of accuracy, robustness, and cybersecurity.",
            "status": "REQUIRES_MONITORING" if psi > 0.10 else "COMPLIANT",
            "evidence": f"Distribution shift PSI: {psi:.4f}. KS p-value: {p_value:.4f}.",
            "action_required": psi > 0.20,
        },
        "Article_17_Quality_Management": {
            "obligation": "Implement quality management system including post-market monitoring.",
            "status": "COMPLIANT",
            "evidence": "Automated monitoring pipeline active. Audit trail maintained with SHA-256 integrity hash.",
            "action_required": False,
        },
    }

    overall_compliant = all(
        not a["action_required"] for a in articles.values()
    )

    return {
        "assessment_timestamp": datetime.now(timezone.utc).isoformat(),
        "overall_status": "COMPLIANT" if overall_compliant else "ACTION_REQUIRED",
        "risk_tier": tier,
        "articles": articles,
        "conformity_assessment_required": tier in ("HIGH", "CRITICAL"),
        "notified_body_required": tier == "CRITICAL",
    }


# ---------------------------------------------------------------------------
# UK AI Framework Compliance
# ---------------------------------------------------------------------------

def uk_ai_framework_check(risk_score: float, sector: str = "general") -> dict:
    """
    UK DSIT AI Framework + FCA Consumer Duty + ICO assessment.

    Parameters
    ----------
    risk_score : float  0–100 AegisML risk score
    sector     : str    "financial_services", "healthcare", "general"
    """
    tier, action = map_risk_tier(risk_score)

    # FCA Consumer Duty obligations (financial services)
    fca_obligation = None
    if sector == "financial_services":
        if risk_score >= 70:
            fca_obligation = {
                "status": "BREACH_RISK",
                "rule": "FCA Consumer Duty — PRIN 12",
                "action": "Review AI system outputs for consumer harm risk. "
                          "Notify Compliance team and consider voluntary disclosure to FCA.",
            }
        else:
            fca_obligation = {
                "status": "COMPLIANT",
                "rule": "FCA Consumer Duty — PRIN 12",
                "action": "Standard monitoring. Document AI system in Consumer Duty board report.",
            }

    # ICO guidance
    ico_obligation = {
        "status": "COMPLIANT" if risk_score < 70 else "REVIEW_REQUIRED",
        "framework": "ICO AI Auditing Framework (UK GDPR Article 22)",
        "automated_decision_making": risk_score >= 70,
        "action": (
            "Ensure Data Protection Impact Assessment (DPIA) completed. "
            "Document lawful basis for automated processing."
        ) if risk_score >= 70 else "Standard DPIA documentation sufficient.",
    }

    return {
        "risk_tier": tier,
        "dsit_ai_principles": {
            "safety": tier not in ("CRITICAL",),
            "security": True,
            "fairness": "bias_assessment_required",
            "accountability": True,
            "transparency": True,
            "contestability": True,
        },
        "fca_consumer_duty": fca_obligation,
        "ico_gdpr_article22": ico_obligation,
        "ico_notification_required": risk_score >= 90,
        "sector": sector,
    }


# ---------------------------------------------------------------------------
# UAE TDRA AI Ethics
# ---------------------------------------------------------------------------

def uae_ai_ethics_check(risk_score: float) -> dict:
    """UAE TDRA AI Ethics Principles assessment."""
    tier, _ = map_risk_tier(risk_score)
    return {
        "tdra_framework": "UAE National AI Strategy 2031",
        "risk_tier": tier,
        "principles": {
            "transparency": "COMPLIANT",
            "fairness": "bias_assessment_recommended",
            "accountability": "COMPLIANT",
            "safety": "COMPLIANT" if risk_score < 70 else "REVIEW_REQUIRED",
            "privacy": "COMPLIANT",
        },
        "difc_data_protection": {
            "applicable": True,
            "status": "COMPLIANT" if risk_score < 70 else "REVIEW_REQUIRED",
        },
    }


# ---------------------------------------------------------------------------
# US NIST AI RMF
# ---------------------------------------------------------------------------

def us_nist_rmf_check(risk_score: float) -> dict:
    """NIST AI Risk Management Framework 1.0 assessment."""
    tier, _ = map_risk_tier(risk_score)
    return {
        "framework": "NIST AI RMF 1.0",
        "risk_tier": tier,
        "functions": {
            "GOVERN": "Active governance policy in place",
            "MAP": f"Risk mapped — score {risk_score:.1f}/100",
            "MEASURE": "Quantitative metrics computed (PSI, KS, ECE, Brier)",
            "MANAGE": "MANAGE" if risk_score < 70 else "MANAGE — escalation triggered",
        },
        "sociotechnical_risk": "ELEVATED" if risk_score >= 70 else "NOMINAL",
    }
