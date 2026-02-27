"""
reliability/report.py
----------------------
Enterprise PDF report generator.

Sections:
  1. Cover Page
  2. Executive Summary
  3. Model Performance Metrics
  4. Calibration Analysis
  5. Distribution Drift Analysis
  6. Risk Assessment + EU AI Act Compliance
  7. Recommendations & Deployment Verdict

Preserves all original visual design; adds EU AI Act, UK AI Framework,
new metrics (AUC-ROC, MCC), and structured compliance section.
"""

from __future__ import annotations

import io
from datetime import datetime
from typing import Optional

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    HRFlowable,
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.pdfgen import canvas as rl_canvas
from reportlab.platypus.flowables import Flowable

# ---------------------------------------------------------------------------
# Colour Palette
# ---------------------------------------------------------------------------

BRAND_DARK   = colors.HexColor("#1a1a2e")
BRAND_MID    = colors.HexColor("#16213e")
BRAND_ACCENT = colors.HexColor("#0f3460")
BRAND_BLUE   = colors.HexColor("#2980b9")
RISK_LOW     = colors.HexColor("#27ae60")
RISK_MED     = colors.HexColor("#f39c12")
RISK_HIGH    = colors.HexColor("#e67e22")
RISK_CRIT    = colors.HexColor("#e74c3c")
LIGHT_GREY   = colors.HexColor("#f4f6f8")
MID_GREY     = colors.HexColor("#bdc3c7")
TEXT_DARK    = colors.HexColor("#2c3e50")

RISK_COLOR_MAP = {
    "Low":      RISK_LOW,
    "Medium":   RISK_MED,
    "High":     RISK_HIGH,
    "Critical": RISK_CRIT,
}

# ---------------------------------------------------------------------------
# Canvas with header/footer
# ---------------------------------------------------------------------------

class _ReportCanvas(rl_canvas.Canvas):
    def __init__(self, *args, model_name="ML Model", **kwargs):
        super().__init__(*args, **kwargs)
        self._model_name = model_name
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        n_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self._draw_header_footer(n_pages)
            rl_canvas.Canvas.showPage(self)
        rl_canvas.Canvas.save(self)

    def _draw_header_footer(self, page_count):
        page_num = self._pageNumber
        width, height = A4

        self.setFillColor(BRAND_DARK)
        self.rect(0, height - 1.2 * cm, width, 1.2 * cm, fill=1, stroke=0)
        self.setFillColor(colors.white)
        self.setFont("Helvetica-Bold", 9)
        self.drawString(1 * cm, height - 0.8 * cm, "AegisML Enterprise | ML Reliability & Distribution Shift Auditor")
        self.setFont("Helvetica", 8)
        self.drawRightString(width - 1 * cm, height - 0.8 * cm, self._model_name)

        self.setFillColor(MID_GREY)
        self.setFont("Helvetica", 7.5)
        self.drawString(1 * cm, 0.6 * cm,
                        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} | Confidential | AegisML Enterprise v5.0.0")
        self.drawRightString(width - 1 * cm, 0.6 * cm, f"Page {page_num} of {page_count}")
        self.setStrokeColor(MID_GREY)
        self.setLineWidth(0.5)
        self.line(1 * cm, 1.1 * cm, width - 1 * cm, 1.1 * cm)


# ---------------------------------------------------------------------------
# Style factory
# ---------------------------------------------------------------------------

def _styles():
    custom = {
        "section_heading": ParagraphStyle(
            "section_heading", fontName="Helvetica-Bold", fontSize=14,
            textColor=BRAND_DARK, spaceBefore=14, spaceAfter=6,
        ),
        "subsection_heading": ParagraphStyle(
            "subsection_heading", fontName="Helvetica-Bold", fontSize=11,
            textColor=BRAND_ACCENT, spaceBefore=8, spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "body", fontName="Helvetica", fontSize=10, textColor=TEXT_DARK,
            spaceAfter=6, leading=15, alignment=TA_JUSTIFY,
        ),
        "caption": ParagraphStyle(
            "caption", fontName="Helvetica-Oblique", fontSize=8.5,
            textColor=colors.HexColor("#7f8c8d"), alignment=TA_CENTER, spaceAfter=4,
        ),
        "verdict": ParagraphStyle(
            "verdict", fontName="Helvetica", fontSize=10, textColor=TEXT_DARK,
            spaceAfter=6, leading=15, alignment=TA_JUSTIFY,
            leftIndent=10, rightIndent=10,
        ),
        "bullet": ParagraphStyle(
            "bullet", fontName="Helvetica", fontSize=9.5, textColor=TEXT_DARK,
            spaceAfter=4, leading=14, leftIndent=14, bulletIndent=4,
        ),
    }
    return custom


# ---------------------------------------------------------------------------
# Helper builders
# ---------------------------------------------------------------------------

def _section_header(story, styles, title: str, section_num: int = None):
    label = f"{section_num}. {title}" if section_num else title
    story.append(HRFlowable(width="100%", thickness=1, color=BRAND_ACCENT, spaceAfter=4))
    story.append(Paragraph(label, styles["section_heading"]))


def _kv_table(data: list, col_widths=None):
    if col_widths is None:
        col_widths = [7 * cm, 10 * cm]
    rows = [[k, v] for k, v in data]
    t = Table(rows, colWidths=col_widths, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("FONTNAME",      (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME",      (1, 0), (1, -1), "Helvetica"),
        ("FONTSIZE",      (0, 0), (-1, -1), 9.5),
        ("TEXTCOLOR",     (0, 0), (0, -1), BRAND_ACCENT),
        ("TEXTCOLOR",     (1, 0), (1, -1), TEXT_DARK),
        ("ROWBACKGROUNDS",(0, 0), (-1, -1), [LIGHT_GREY, colors.white]),
        ("GRID",          (0, 0), (-1, -1), 0.3, MID_GREY),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
    ]))
    return t


def _risk_badge_table(risk_level: str):
    color = RISK_COLOR_MAP.get(risk_level, RISK_HIGH)
    t = Table([[f" RISK LEVEL: {risk_level.upper()} "]], colWidths=[8 * cm], hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), color),
        ("FONTNAME",      (0, 0), (-1, -1), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 13),
        ("TEXTCOLOR",     (0, 0), (-1, -1), colors.white),
        ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
        ("TOPPADDING",    (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    return t


def _metrics_table(metrics: dict, calibration: dict):
    cm_raw = metrics.get("confusion_matrix", [[0, 0], [0, 0]])
    if cm_raw and len(cm_raw) == 2:
        tn = cm_raw[0][0]; fp = cm_raw[0][1]; fn = cm_raw[1][0]; tp = cm_raw[1][1]
        cm_str = f"TP={tp}, TN={tn}, FP={fp}, FN={fn}"
    else:
        cm_str = "N/A"

    auc = metrics.get("auc_roc")
    mcc = metrics.get("mcc")

    data = [
        ["Metric", "Value", "Interpretation"],
        ["Accuracy",             f"{metrics.get('accuracy', 0):.2%}",     _interp_accuracy(metrics.get('accuracy', 0))],
        ["F1 Score (weighted)",  f"{metrics.get('f1_weighted', metrics.get('f1', 0)):.4f}", _interp_f1(metrics.get('f1_weighted', 0))],
        ["Precision",            f"{metrics.get('precision', 0):.4f}",    "Positive predictive value"],
        ["Recall",               f"{metrics.get('recall', 0):.4f}",       "Sensitivity / True Positive Rate"],
        ["AUC-ROC",              f"{auc:.4f}" if auc else "N/A",          "Discrimination ability (1.0 = perfect)"],
        ["MCC",                  f"{mcc:.4f}" if mcc else "N/A",          "Matthews Correlation Coefficient"],
        ["Brier Score",          f"{metrics.get('brier_score', 0):.4f}",  _interp_brier(metrics.get('brier_score', 0))],
        ["ECE",                  f"{calibration.get('ece', 0):.4f}",      _interp_ece(calibration.get('ece', 0))],
        ["MCE",                  f"{calibration.get('mce', 0):.4f}",      "Maximum per-bin calibration error"],
        ["Overconfidence Gap",   f"{calibration.get('overconfidence_gap', 0):.4f}", "Positive = overconfident on average"],
        ["Confusion Matrix",     cm_str, ""],
    ]

    col_widths = [5.5 * cm, 3 * cm, 8.5 * cm]
    t = Table(data, colWidths=col_widths, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), BRAND_DARK),
        ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 9),
        ("FONTNAME",      (0, 1), (0, -1), "Helvetica-Bold"),
        ("TEXTCOLOR",     (0, 1), (0, -1), BRAND_ACCENT),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [LIGHT_GREY, colors.white]),
        ("GRID",          (0, 0), (-1, -1), 0.3, MID_GREY),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 7),
        ("ALIGN",         (1, 1), (1, -1), "CENTER"),
    ]))
    return t


def _component_risk_table(component_scores: dict):
    data = [["Component", "Score", "Level"]]
    for comp, info in component_scores.items():
        data.append([comp.replace("_", " ").title(), str(info["score"]), info["level"]])

    col_widths = [7 * cm, 3 * cm, 4.5 * cm]
    t = Table(data, colWidths=col_widths, hAlign="LEFT")
    style = [
        ("BACKGROUND",    (0, 0), (-1, 0), BRAND_DARK),
        ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 9.5),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [LIGHT_GREY, colors.white]),
        ("GRID",          (0, 0), (-1, -1), 0.3, MID_GREY),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("ALIGN",         (1, 1), (2, -1), "CENTER"),
    ]
    for i, (comp, info) in enumerate(component_scores.items(), start=1):
        c = RISK_COLOR_MAP.get(info["level"], RISK_HIGH)
        style.extend([
            ("TEXTCOLOR", (2, i), (2, i), c),
            ("FONTNAME",  (2, i), (2, i), "Helvetica-Bold"),
        ])
    t.setStyle(TableStyle(style))
    return t


# ---------------------------------------------------------------------------
# Main report function
# ---------------------------------------------------------------------------

def generate_report(
    output_path: str,
    model_name: str,
    metrics: dict,
    calibration: dict,
    risk: dict,
    drift: Optional[dict] = None,
    prediction_drift: Optional[dict] = None,
    reliability_diagram_bytes: Optional[bytes] = None,
    confidence_hist_bytes: Optional[bytes] = None,
    drift_chart_bytes: Optional[bytes] = None,
    n_samples: int = 0,
) -> str:
    report_date = datetime.now().strftime("%B %d, %Y")
    doc = SimpleDocTemplate(
        output_path, pagesize=A4,
        topMargin=1.8 * cm, bottomMargin=1.8 * cm,
        leftMargin=1.5 * cm, rightMargin=1.5 * cm,
        title=f"AegisML Reliability Report — {model_name}",
        author="AegisML Enterprise v5.0.0",
    )

    styles = _styles()
    story  = []
    risk_level = risk.get("overall_risk_level", "Unknown")

    # ── 1. Executive Summary ──────────────────────────────────────────────
    _section_header(story, styles, "Executive Summary", 1)
    story.append(_risk_badge_table(risk_level))
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        f"This report presents the reliability assessment of <b>{model_name}</b> "
        f"evaluated on <b>{n_samples:,}</b> production samples as of <b>{report_date}</b>. "
        "The audit covers five dimensions: predictive performance, probability calibration, "
        "distribution drift, algorithmic fairness, and overall deployment risk.",
        styles["body"],
    ))

    # EU AI Act summary
    eu_cat = risk.get("eu_ai_act_category", "")
    uk_fw  = risk.get("uk_ai_framework", "")
    if eu_cat:
        story.append(Paragraph(
            f"<b>EU AI Act Category:</b> {eu_cat}",
            styles["body"],
        ))
    if uk_fw:
        story.append(Paragraph(
            f"<b>UK AI Framework:</b> {uk_fw}",
            styles["body"],
        ))

    auc_str = f"{metrics.get('auc_roc'):.4f}" if metrics.get('auc_roc') is not None else "N/A"
    summary_kv = [
        ("Model Name",               model_name),
        ("Evaluation Samples",       f"{n_samples:,}"),
        ("Overall Risk Level",       risk_level),
        ("Weighted Risk Score",      f"{risk.get('weighted_score', 0):.3f} / 3.0"),
        ("Accuracy",                 f"{metrics.get('accuracy', 0):.2%}"),
        ("AUC-ROC",                  auc_str),
        ("ECE (Calibration Error)",  f"{calibration.get('ece', 0):.4f}"),
        ("Brier Score",              f"{metrics.get('brier_score', 0):.4f}"),
        ("Distribution Drift (PSI)", f"{drift.get('overall_drift_score', 0):.4f}" if drift else "N/A"),
        ("Human Oversight Required", "YES" if risk.get("requires_human_oversight") else "No"),
        ("Report Date",              report_date),
    ]
    story.append(_kv_table(summary_kv))

    # ── 2. Performance Metrics ────────────────────────────────────────────
    story.append(PageBreak())
    _section_header(story, styles, "Model Performance Metrics", 2)
    story.append(Paragraph(
        "The following metrics evaluate the model's predictive accuracy on the production dataset. "
        "Accuracy alone is an insufficient measure of model reliability — calibration quality, "
        "AUC-ROC, MCC, and probabilistic accuracy (Brier score) are equally critical.",
        styles["body"],
    ))
    story.append(Spacer(1, 8))
    story.append(_metrics_table(metrics, calibration))

    # ── 3. Calibration Analysis ───────────────────────────────────────────
    story.append(PageBreak())
    _section_header(story, styles, "Calibration Analysis", 3)
    story.append(Paragraph(
        "Calibration measures whether a model's predicted probabilities reflect true empirical frequencies. "
        "A perfectly calibrated model predicts 70% probability means the event occurs 70% of the time. "
        "Poor calibration leads to systematically over- or under-confident decisions.",
        styles["body"],
    ))

    if reliability_diagram_bytes:
        story.append(Spacer(1, 8))
        story.append(Paragraph("Reliability Diagram", styles["subsection_heading"]))
        img = Image(io.BytesIO(reliability_diagram_bytes), width=14 * cm, height=12 * cm)
        story.append(img)
        story.append(Paragraph("Figure 1: Reliability Diagram — Red = overconfident, Green = underconfident", styles["caption"]))

    if confidence_hist_bytes:
        story.append(Spacer(1, 8))
        story.append(Paragraph("Confidence Score Distribution", styles["subsection_heading"]))
        img2 = Image(io.BytesIO(confidence_hist_bytes), width=14 * cm, height=8 * cm)
        story.append(img2)
        story.append(Paragraph("Figure 2: Confidence Score Histogram with KDE overlay", styles["caption"]))

    # ── 4. Distribution Drift ─────────────────────────────────────────────
    story.append(PageBreak())
    _section_header(story, styles, "Distribution Shift Analysis", 4)

    if drift is None and prediction_drift is None:
        story.append(Paragraph(
            "No reference dataset provided. Upload a reference (training) CSV to enable drift analysis.",
            styles["body"],
        ))
    else:
        story.append(Paragraph(
            "Distribution shift occurs when production data differs from the training distribution. "
            "PSI (Population Stability Index) quantifies shift magnitude; "
            "KL Divergence measures information-theoretic distance.",
            styles["body"],
        ))

        if prediction_drift:
            story.append(Paragraph("Prediction Probability Drift", styles["subsection_heading"]))
            story.append(_kv_table([
                ("KL Divergence", f"{prediction_drift.get('kl_divergence', 0):.4f} — {prediction_drift.get('kl_interpretation', '')}"),
                ("PSI (Predicted Probabilities)", f"{prediction_drift.get('psi', 0):.4f} — {prediction_drift.get('psi_interpretation', '')}"),
            ]))

        if drift and drift.get("n_features", 0) > 0:
            story.append(Spacer(1, 8))
            story.append(Paragraph("Feature-Level Drift", styles["subsection_heading"]))
            story.append(_kv_table([
                ("Features Analysed",     str(drift.get("n_features", 0))),
                ("Overall Drift Score",   f"{drift.get('overall_drift_score', 0):.4f} — {drift.get('overall_interpretation', '')}"),
            ]))

        if drift_chart_bytes:
            story.append(Spacer(1, 10))
            img3 = Image(io.BytesIO(drift_chart_bytes), width=14 * cm, height=9 * cm)
            story.append(img3)
            story.append(Paragraph(
                "Figure 3: Feature PSI — Green <0.10 (stable), Orange 0.10–0.20 (moderate), Red >0.20 (significant)",
                styles["caption"],
            ))

    # ── 5. Risk Assessment ────────────────────────────────────────────────
    story.append(PageBreak())
    _section_header(story, styles, "Risk Assessment", 5)
    story.append(_risk_badge_table(risk_level))
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        f"Weighted risk score: <b>{risk.get('weighted_score', 0):.3f}</b> / 3.0. "
        f"EU AI Act Category: <b>{risk.get('eu_ai_act_category', 'N/A')}</b>.",
        styles["body"],
    ))
    story.append(Spacer(1, 8))
    story.append(Paragraph("Component Risk Breakdown", styles["subsection_heading"]))
    story.append(_component_risk_table(risk.get("component_scores", {})))

    # Compliance obligations
    story.append(Spacer(1, 10))
    story.append(Paragraph("Regulatory Compliance", styles["subsection_heading"]))
    story.append(_kv_table([
        ("EU AI Act Category",          risk.get("eu_ai_act_category", "N/A")),
        ("UK AI Framework",             risk.get("uk_ai_framework", "N/A")),
        ("Human Oversight Required",    "YES" if risk.get("requires_human_oversight") else "No"),
        ("Regulatory Notification",     "REQUIRED" if risk.get("requires_regulatory_notification") else "Not required"),
    ]))

    # ── 6. Recommendations & Verdict ─────────────────────────────────────
    story.append(PageBreak())
    _section_header(story, styles, "Recommendations & Deployment Verdict", 6)
    story.append(Paragraph("Recommendations", styles["subsection_heading"]))
    for rec in risk.get("recommendations", []):
        clean_rec = rec.replace("⚠", "WARNING:").replace("🚨", "CRITICAL:").replace(
            "ℹ", "INFO:").replace("✅", "OK:").replace("📋", "NOTE:")
        story.append(Paragraph(f"• {clean_rec}", styles["bullet"]))
        story.append(Spacer(1, 3))

    story.append(Spacer(1, 10))
    story.append(Paragraph("Deployment Verdict", styles["subsection_heading"]))
    story.append(HRFlowable(width="100%", thickness=1,
                             color=RISK_COLOR_MAP.get(risk_level, RISK_HIGH), spaceAfter=6))

    verdict_text = risk.get("deployment_verdict", "No verdict available.")
    for line in verdict_text.split(". "):
        if line.strip():
            story.append(Paragraph(line.strip() + ".", styles["verdict"]))

    story.append(Spacer(1, 16))
    story.append(HRFlowable(width="100%", thickness=2, color=BRAND_DARK, spaceAfter=8))
    story.append(Paragraph(
        f"AegisML Enterprise v5.0.0 | {report_date} | EU AI Act Aligned | Confidential",
        styles["caption"],
    ))

    def make_canvas(filename, **kwargs):
        return _ReportCanvas(filename, pagesize=A4, model_name=model_name)

    doc.build(story, canvasmaker=make_canvas)
    return output_path


# ---------------------------------------------------------------------------
# Metric interpreters
# ---------------------------------------------------------------------------

def _interp_accuracy(v):
    if v >= 0.90: return "Excellent"
    if v >= 0.80: return "Good"
    if v >= 0.70: return "Acceptable"
    return "Poor — requires investigation"

def _interp_f1(v):
    if v >= 0.90: return "Excellent"
    if v >= 0.80: return "Good"
    if v >= 0.70: return "Moderate"
    return "Poor"

def _interp_brier(v):
    if v <= 0.10: return "Excellent probabilistic accuracy"
    if v <= 0.17: return "Good"
    if v <= 0.22: return "Moderate"
    return "Poor — near-random probabilistic predictions"

def _interp_ece(v):
    if v <= 0.05: return "Well calibrated"
    if v <= 0.10: return "Moderate miscalibration"
    if v <= 0.15: return "Poor calibration"
    return "Severely miscalibrated"
