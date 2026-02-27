"""
verification/pdf_report.py
---------------------------
Enterprise audit PDF generator — 13 language support.

Languages: English, German, French, Spanish, Italian, Polish,
           Romanian, Dutch, Russian, Ukrainian, Hindi, Urdu, Arabic

Features:
  - Branded cover page
  - Executive summary table
  - Technical risk metrics table
  - EU AI Act compliance section
  - Integrity hash (tamper-evident)
  - Professional typography and colour scheme
"""

from __future__ import annotations

import json
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, inch
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# ---------------------------------------------------------------------------
# Colour Palette
# ---------------------------------------------------------------------------

BRAND_DARK   = colors.HexColor("#0D1B2A")
BRAND_ACCENT = colors.HexColor("#1B4F8A")
BRAND_LIGHT  = colors.HexColor("#4A90D9")
RISK_LOW     = colors.HexColor("#27AE60")
RISK_MED     = colors.HexColor("#F39C12")
RISK_HIGH    = colors.HexColor("#E67E22")
RISK_CRIT    = colors.HexColor("#C0392B")
LIGHT_GREY   = colors.HexColor("#F4F6F8")
MID_GREY     = colors.HexColor("#BDC3C7")
TEXT_DARK    = colors.HexColor("#2C3E50")

RISK_COLOUR_MAP = {
    "LOW":      RISK_LOW,
    "MEDIUM":   RISK_MED,
    "HIGH":     RISK_HIGH,
    "CRITICAL": RISK_CRIT,
}

# ---------------------------------------------------------------------------
# Language Dictionary — 13 Languages
# ---------------------------------------------------------------------------

LANGUAGE_CONTENT: dict[str, dict] = {
    "English": {
        "title": "AI Risk & Governance Audit Report",
        "subtitle": "Production ML Reliability & Drift Assurance",
        "executive_summary": "Executive Summary",
        "technical_section": "Technical Risk Assessment",
        "compliance_section": "Regulatory Compliance",
        "integrity_section": "Report Integrity",
        "verified": "The AI system has passed all drift, bias, and governance validation checks.",
        "failed": "The AI system exhibits significant risk indicators requiring immediate investigation.",
        "status_label": "System Status",
        "risk_tier_label": "Risk Tier",
        "generated_label": "Generated",
        "confidential": "CONFIDENTIAL — For Authorised Recipients Only",
    },
    "German": {
        "title": "KI Risiko- und Governance-Prüfbericht",
        "subtitle": "Produktions-ML Zuverlässigkeit & Drift-Assurance",
        "executive_summary": "Zusammenfassung für Führungsebene",
        "technical_section": "Technische Risikoanalyse",
        "compliance_section": "Regulatorische Konformität",
        "integrity_section": "Berichtsintegrität",
        "verified": "Das KI-System hat alle Drift-, Bias- und Governance-Prüfungen bestanden.",
        "failed": "Das KI-System weist erhebliche Risikoindikatoren auf, die sofortige Untersuchung erfordern.",
        "status_label": "Systemstatus",
        "risk_tier_label": "Risikostufe",
        "generated_label": "Erstellt",
        "confidential": "VERTRAULICH — Nur für autorisierte Empfänger",
    },
    "French": {
        "title": "Rapport d'Audit de Risque et Gouvernance IA",
        "subtitle": "Fiabilité ML en Production & Assurance de Dérive",
        "executive_summary": "Résumé Exécutif",
        "technical_section": "Évaluation Technique des Risques",
        "compliance_section": "Conformité Réglementaire",
        "integrity_section": "Intégrité du Rapport",
        "verified": "Le système d'IA a réussi tous les contrôles de dérive, de biais et de gouvernance.",
        "failed": "Le système d'IA présente des indicateurs de risque significatifs nécessitant une investigation immédiate.",
        "status_label": "Statut Système",
        "risk_tier_label": "Niveau de Risque",
        "generated_label": "Généré le",
        "confidential": "CONFIDENTIEL — Destinataires Autorisés Uniquement",
    },
    "Spanish": {
        "title": "Informe de Auditoría de Riesgo y Gobernanza IA",
        "subtitle": "Fiabilidad ML en Producción y Garantía de Desviación",
        "executive_summary": "Resumen Ejecutivo",
        "technical_section": "Evaluación Técnica de Riesgos",
        "compliance_section": "Cumplimiento Regulatorio",
        "integrity_section": "Integridad del Informe",
        "verified": "El sistema de IA ha superado todas las validaciones de desviación, sesgo y gobernanza.",
        "failed": "El sistema de IA presenta indicadores de riesgo significativos que requieren investigación inmediata.",
        "status_label": "Estado del Sistema",
        "risk_tier_label": "Nivel de Riesgo",
        "generated_label": "Generado el",
        "confidential": "CONFIDENCIAL — Solo para destinatarios autorizados",
    },
    "Italian": {
        "title": "Rapporto di Audit su Rischio e Governance IA",
        "subtitle": "Affidabilità ML in Produzione e Garanzia di Deriva",
        "executive_summary": "Sintesi Esecutiva",
        "technical_section": "Valutazione Tecnica del Rischio",
        "compliance_section": "Conformità Normativa",
        "integrity_section": "Integrità del Rapporto",
        "verified": "Il sistema IA ha superato tutte le verifiche di deriva, bias e governance.",
        "failed": "Il sistema IA mostra indicatori di rischio significativi che richiedono indagine immediata.",
        "status_label": "Stato del Sistema",
        "risk_tier_label": "Livello di Rischio",
        "generated_label": "Generato il",
        "confidential": "RISERVATO — Solo per destinatari autorizzati",
    },
    "Polish": {
        "title": "Raport Audytu Ryzyka i Nadzoru AI",
        "subtitle": "Niezawodność ML w Produkcji i Zapewnienie Dryfu",
        "executive_summary": "Podsumowanie Wykonawcze",
        "technical_section": "Techniczna Ocena Ryzyka",
        "compliance_section": "Zgodność Regulacyjna",
        "integrity_section": "Integralność Raportu",
        "verified": "System AI przeszedł wszystkie walidacje dryfu, biasu i nadzoru.",
        "failed": "System AI wykazuje istotne wskaźniki ryzyka wymagające natychmiastowego dochodzenia.",
        "status_label": "Status Systemu",
        "risk_tier_label": "Poziom Ryzyka",
        "generated_label": "Wygenerowano",
        "confidential": "POUFNE — Tylko dla upoważnionych odbiorców",
    },
    "Romanian": {
        "title": "Raport de Audit privind Riscul și Guvernanța AI",
        "subtitle": "Fiabilitatea ML în Producție și Asigurarea Derivei",
        "executive_summary": "Rezumat Executiv",
        "technical_section": "Evaluare Tehnică a Riscului",
        "compliance_section": "Conformitate Reglementară",
        "integrity_section": "Integritatea Raportului",
        "verified": "Sistemul AI a trecut toate validările de derivă, bias și guvernanță.",
        "failed": "Sistemul AI prezintă indicatori semnificativi de risc care necesită investigație imediată.",
        "status_label": "Starea Sistemului",
        "risk_tier_label": "Nivel de Risc",
        "generated_label": "Generat la",
        "confidential": "CONFIDENTIAL — Numai pentru destinatari autorizați",
    },
    "Dutch": {
        "title": "AI Risico- en Governance Auditrapport",
        "subtitle": "ML Betrouwbaarheid in Productie & Drift Assurance",
        "executive_summary": "Managementsamenvatting",
        "technical_section": "Technische Risicobeoordeling",
        "compliance_section": "Regelgevende Naleving",
        "integrity_section": "Rapportintegriteit",
        "verified": "Het AI-systeem heeft alle validaties voor drift, bias en governance doorstaan.",
        "failed": "Het AI-systeem vertoont significante risicofactoren die onmiddellijk onderzoek vereisen.",
        "status_label": "Systeemstatus",
        "risk_tier_label": "Risiconiveau",
        "generated_label": "Gegenereerd op",
        "confidential": "VERTROUWELIJK — Alleen voor bevoegde ontvangers",
    },
    "Russian": {
        "title": "Otchet po auditu riskov i upravleniya II",
        "subtitle": "Nadezhnost ML v produkcii i obespechenie drejfa",
        "executive_summary": "Ispolnitelnoe rezyume",
        "technical_section": "Tehnicheskaya ocenka riska",
        "compliance_section": "Regulyatornoe sootvetstvie",
        "integrity_section": "Celostnost otcheta",
        "verified": "Sistema II uspeshno proshla vse proverki drejfa, smeshcheniya i upravleniya.",
        "failed": "Sistema II demonstriruet znachitelnye riski, trebuyushchie nemedlennogo rassledovaniya.",
        "status_label": "Status sistemy",
        "risk_tier_label": "Uroven riska",
        "generated_label": "Sgenerirovan",
        "confidential": "KONFIDENCIALNO — Tolko dlya avtorizovannyh poluchatelej",
    },
    "Ukrainian": {
        "title": "Zvit z audytu ryzykiv ta upravlinnya SHI",
        "subtitle": "Nadijnist ML u vyrobnytstvi ta zabezpechennya dreyfingu",
        "executive_summary": "Vykonavche rezyume",
        "technical_section": "Tekhnichna otsinka ryzyku",
        "compliance_section": "Regulyatorna vidpovidnist",
        "integrity_section": "Tsilisnist zvitu",
        "verified": "Systema SHI uspishno proyshla vsi perevirky dreyfingu, uperedhzenosti ta vryaduvannya.",
        "failed": "Systema SHI demonstruye znachni ryzyky, shcho potrebuyut nehaynogo rozsliduvannya.",
        "status_label": "Status systemy",
        "risk_tier_label": "Riven ryzyku",
        "generated_label": "Zgenerovano",
        "confidential": "KONFIDENTSIINO — Lyshe dlya avtoryzovanykh otrymuvachy",
    },
    "Hindi": {
        "title": "AI Jokhim aur Shasana Audit Riport",
        "subtitle": "Utpadana ML Vishvasniyata aur Drift Ashvasan",
        "executive_summary": "Karyakari Saransh",
        "technical_section": "Takniki Jokhim Mulyankan",
        "compliance_section": "Niyamak Anupalan",
        "integrity_section": "Riport ki Satyanishtha",
        "verified": "AI pranali ne sabhi drift, pakksha aur shasana sataapana janch puri kar li hai.",
        "failed": "AI pranali mein mahatvapurn jokhim sanketik hain jo tatkal janch ki mang karte hain.",
        "status_label": "Pranali Sthiti",
        "risk_tier_label": "Jokhim Star",
        "generated_label": "Utpanna",
        "confidential": "GUPT — Kewal Pradhikrit Prapta-kartaon Ke Liye",
    },
    "Urdu": {
        "title": "AI Khatra aur Hakumrani Audit Report",
        "subtitle": "Paidawar ML Bharosemandi aur Drifting Keyafiyat",
        "executive_summary": "Izami Khulasa",
        "technical_section": "Fanni Khatray Ki Jach",
        "compliance_section": "Qanooni Ittefaq",
        "integrity_section": "Report Ki Salaamat",
        "verified": "AI nizam ne tamam drift, taassub aur hukumrani ki jaanch mein kamyabi haasil ki.",
        "failed": "AI nizam mein numayan khatray ke nishaan hain jo fori tehqeeq ka taqaza karte hain.",
        "status_label": "Nizam Ki Halat",
        "risk_tier_label": "Khatray Ka Darjah",
        "generated_label": "Paida Kiya Gaya",
        "confidential": "KHUFIYA — Sirf Mujaz Mustafideen Ke Liye",
    },
    "Arabic": {
        "title": "Taqrir Muraja'at Mukhatarat wa Hukm al-Dhaka' al-Isnaa'i",
        "subtitle": "Mawthqiat ML al-Intaj wa Daman al-Inhiraf",
        "executive_summary": "Mulakhkhas Tanfidhee",
        "technical_section": "Taqyeem al-Mukhatarat al-Tiqniyya",
        "compliance_section": "al-Imtithal al-Tandhimi",
        "integrity_section": "Sahhhat al-Taqrir",
        "verified": "Ijaz al-nidham al-dhaka' al-isnaa'i jami' ikhtibaraat al-inhiraf wal-ta'assub wal-hukm.",
        "failed": "Yudhhir nidham al-dhaka' al-isnaa'i mu'ashirat mukhatara kabiira tattallab tahqeeqan fawriyyan.",
        "status_label": "Halat al-Nidham",
        "risk_tier_label": "Mustawa al-Mukhatar",
        "generated_label": "Sadara fi",
        "confidential": "SIRRI — Lilmustaqbilin al-Musarrah Lahum Faqat",
    },
}


# ---------------------------------------------------------------------------
# Style Factory
# ---------------------------------------------------------------------------

def _get_styles():
    base = getSampleStyleSheet()
    return {
        "h1": ParagraphStyle(
            "h1", fontName="Helvetica-Bold", fontSize=20,
            textColor=colors.white, alignment=TA_CENTER, spaceAfter=6,
        ),
        "h2": ParagraphStyle(
            "h2", fontName="Helvetica-Bold", fontSize=14,
            textColor=BRAND_DARK, spaceBefore=14, spaceAfter=6,
        ),
        "h3": ParagraphStyle(
            "h3", fontName="Helvetica-Bold", fontSize=11,
            textColor=BRAND_ACCENT, spaceBefore=8, spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "body", fontName="Helvetica", fontSize=10,
            textColor=TEXT_DARK, leading=15, spaceAfter=6,
        ),
        "small": ParagraphStyle(
            "small", fontName="Helvetica", fontSize=8,
            textColor=MID_GREY, alignment=TA_CENTER,
        ),
        "mono": ParagraphStyle(
            "mono", fontName="Courier", fontSize=8,
            textColor=TEXT_DARK, leading=12, spaceAfter=4,
        ),
    }


# ---------------------------------------------------------------------------
# Cover Page Builder
# ---------------------------------------------------------------------------

def _build_cover(lang: dict, report_data: dict) -> list:
    styles = _get_styles()
    elements = []

    # Header banner table
    status = report_data.get("system_status", "UNKNOWN")
    gen_time = report_data.get("verification_timestamp", datetime.now().isoformat())[:19]
    risk_tiers = [r.get("risk_tier", "N/A") for r in report_data.get("results", [])]
    highest_tier = "CRITICAL" if "CRITICAL" in risk_tiers else \
                   "HIGH" if "HIGH" in risk_tiers else \
                   "MEDIUM" if "MEDIUM" in risk_tiers else "LOW"
    tier_colour = RISK_COLOUR_MAP.get(highest_tier, RISK_HIGH)

    cover_data = [
        [Paragraph(lang["title"], styles["h1"])],
        [Paragraph(lang["subtitle"], ParagraphStyle(
            "sub", fontName="Helvetica", fontSize=11,
            textColor=colors.HexColor("#AED6F1"), alignment=TA_CENTER,
        ))],
    ]
    cover_table = Table(cover_data, colWidths=[17 * cm])
    cover_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), BRAND_DARK),
        ("TOPPADDING", (0, 0), (-1, -1), 18),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 18),
        ("LEFTPADDING", (0, 0), (-1, -1), 20),
        ("RIGHTPADDING", (0, 0), (-1, -1), 20),
        ("ROUNDEDCORNERS", [6]),
    ]))
    elements.append(cover_table)
    elements.append(Spacer(1, 0.5 * cm))

    # Info box
    info_data = [
        [lang["status_label"],      status],
        [lang["risk_tier_label"],   highest_tier],
        [lang["generated_label"],   gen_time],
        ["Version",                 report_data.get("version", "5.0.0-enterprise")],
        ["Git Commit",              report_data.get("git_commit", "N/A")[:16] + "..."],
    ]
    info_table = Table(info_data, colWidths=[6 * cm, 11 * cm])
    info_table.setStyle(TableStyle([
        ("FONTNAME",        (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME",        (1, 0), (1, -1), "Helvetica"),
        ("FONTSIZE",        (0, 0), (-1, -1), 10),
        ("TEXTCOLOR",       (0, 0), (0, -1), BRAND_ACCENT),
        ("TEXTCOLOR",       (1, 0), (1, -1), TEXT_DARK),
        ("ROWBACKGROUNDS",  (0, 0), (-1, -1), [LIGHT_GREY, colors.white]),
        ("GRID",            (0, 0), (-1, -1), 0.3, MID_GREY),
        ("TOPPADDING",      (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING",   (0, 0), (-1, -1), 6),
        ("LEFTPADDING",     (0, 0), (-1, -1), 10),
        # Colour highest risk tier value
        ("TEXTCOLOR",       (1, 1), (1, 1), tier_colour),
        ("FONTNAME",        (1, 1), (1, 1), "Helvetica-Bold"),
    ]))
    elements.append(info_table)
    elements.append(Spacer(1, 0.4 * cm))

    # Regulatory compliance badges row
    badges = ["EU AI Act", "GDPR", "ISO 27001", "SOC2", "NIST AI RMF", "UK AI Framework"]
    badge_data = [[Paragraph(b, ParagraphStyle(
        f"b{i}", fontName="Helvetica-Bold", fontSize=7.5,
        textColor=colors.white, alignment=TA_CENTER,
    )) for i, b in enumerate(badges)]]
    badge_table = Table(badge_data, colWidths=[2.83 * cm] * 6)
    badge_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), BRAND_ACCENT),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("ROUNDEDCORNERS", [4]),
    ]))
    elements.append(badge_table)
    elements.append(Spacer(1, 0.3 * cm))

    # Confidential footer
    elements.append(Paragraph(lang["confidential"], styles["small"]))
    elements.append(PageBreak())
    return elements


# ---------------------------------------------------------------------------
# Executive Summary
# ---------------------------------------------------------------------------

def _build_executive_summary(lang: dict, report_data: dict) -> list:
    styles = _get_styles()
    elements = [Paragraph(lang["executive_summary"], styles["h2"])]

    status = report_data.get("system_status", "UNKNOWN")
    summary_text = lang["verified"] if status == "VERIFIED" else lang["failed"]

    # Status highlight table
    status_colour = RISK_LOW if status == "VERIFIED" else RISK_CRIT
    st_data = [[Paragraph(
        f"{lang['status_label']}: {status} — {summary_text}",
        ParagraphStyle("sv", fontName="Helvetica-Bold", fontSize=10, textColor=colors.white),
    )]]
    st_table = Table(st_data, colWidths=[17 * cm])
    st_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), status_colour),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("LEFTPADDING", (0, 0), (-1, -1), 14),
        ("ROUNDEDCORNERS", [4]),
    ]))
    elements.append(st_table)
    elements.append(Spacer(1, 0.3 * cm))
    return elements


# ---------------------------------------------------------------------------
# Technical Risk Table
# ---------------------------------------------------------------------------

def _build_technical_section(lang: dict, report_data: dict) -> list:
    styles = _get_styles()
    elements = [Paragraph(lang["technical_section"], styles["h2"])]

    results = report_data.get("results", [])
    if not results:
        elements.append(Paragraph("No test results available.", styles["body"]))
        return elements

    headers = ["Case", "PSI", "KS p-value", "Risk Score", "Tier", "Status"]
    table_data = [headers]

    for r in results:
        tier = r.get("risk_tier", "N/A")
        status = r.get("status", "N/A")
        table_data.append([
            r.get("case", ""),
            str(r.get("psi", r.get("statistical_parity_difference", "N/A"))),
            str(r.get("ks_pvalue", "N/A")),
            str(r.get("risk_score", "N/A")),
            tier,
            status,
        ])

    col_widths = [3.5 * cm, 2.5 * cm, 2.5 * cm, 2.5 * cm, 2.5 * cm, 2 * cm]
    table = Table(table_data, colWidths=col_widths)

    style = [
        ("BACKGROUND",  (0, 0), (-1, 0), BRAND_DARK),
        ("TEXTCOLOR",   (0, 0), (-1, 0), colors.white),
        ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [LIGHT_GREY, colors.white]),
        ("GRID",        (0, 0), (-1, -1), 0.3, MID_GREY),
        ("ALIGN",       (1, 0), (-1, -1), "CENTER"),
        ("TOPPADDING",  (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ]

    # Colour-code risk tiers and statuses
    for i, r in enumerate(results, start=1):
        tier = r.get("risk_tier", "N/A").upper()
        tc = RISK_COLOUR_MAP.get(tier, RISK_MED)
        style.append(("TEXTCOLOR",  (4, i), (4, i), tc))
        style.append(("FONTNAME",   (4, i), (4, i), "Helvetica-Bold"))
        if r.get("status") == "FAIL":
            style.append(("TEXTCOLOR",  (5, i), (5, i), RISK_CRIT))
            style.append(("FONTNAME",   (5, i), (5, i), "Helvetica-Bold"))

    table.setStyle(TableStyle(style))
    elements.append(table)
    elements.append(Spacer(1, 0.3 * cm))
    return elements


# ---------------------------------------------------------------------------
# Compliance Section
# ---------------------------------------------------------------------------

def _build_compliance_section(lang: dict, report_data: dict) -> list:
    styles = _get_styles()
    elements = [Paragraph(lang["compliance_section"], styles["h2"])]

    reg = report_data.get("regulatory_alignment", {})
    eu = reg.get("eu_ai_act", {})

    if eu:
        elements.append(Paragraph("EU AI Act (2024)", styles["h3"]))
        for article, desc in eu.items():
            article_clean = article.replace("_", " ")
            elements.append(Paragraph(
                f"<b>{article_clean}:</b> {desc}",
                styles["body"],
            ))

    elements.append(Spacer(1, 0.2 * cm))
    elements.append(Paragraph(
        f"<b>UK AI Framework:</b> {reg.get('uk_ai_framework', 'N/A')}",
        styles["body"],
    ))
    elements.append(Paragraph(
        f"<b>GDPR:</b> {reg.get('gdpr', 'N/A')}",
        styles["body"],
    ))
    elements.append(Paragraph(
        f"<b>NIST AI RMF:</b> {reg.get('nist_ai_rmf', 'N/A')}",
        styles["body"],
    ))
    return elements


# ---------------------------------------------------------------------------
# Integrity Section
# ---------------------------------------------------------------------------

def _build_integrity_section(lang: dict, report_data: dict) -> list:
    styles = _get_styles()
    elements = [Paragraph(lang["integrity_section"], styles["h2"])]

    elements.append(Paragraph(
        f"<b>SHA-256 Integrity Hash:</b>",
        styles["body"],
    ))
    elements.append(Paragraph(
        report_data.get("integrity_hash_sha256", "N/A"),
        styles["mono"],
    ))
    elements.append(Paragraph(
        f"<b>Verification Timestamp:</b> {report_data.get('verification_timestamp', 'N/A')}",
        styles["body"],
    ))
    elements.append(Paragraph(
        f"<b>Python Version:</b> {report_data.get('python_version', 'N/A')} | "
        f"<b>Platform:</b> {report_data.get('platform', 'N/A')}",
        styles["body"],
    ))
    elements.append(Spacer(1, 0.5 * cm))
    elements.append(Paragraph(
        "This report was generated automatically by AegisML Enterprise. "
        "The integrity hash above provides tamper-evident verification of all report contents. "
        "Interpret metrics alongside domain expertise. "
        "Not a substitute for comprehensive model governance.",
        styles["small"],
    ))
    return elements


# ---------------------------------------------------------------------------
# Main PDF Generator
# ---------------------------------------------------------------------------

def generate_pdf(
    report_json_path: str,
    output_path: str = "verification/Audit_Report.pdf",
    language: str = "English",
) -> str:
    """
    Generate a professional multi-language audit PDF.

    Parameters
    ----------
    report_json_path  Path to verification_report.json
    output_path       Where to write the PDF
    language          Report language (default: English)

    Returns
    -------
    output_path
    """
    import os
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    if language not in LANGUAGE_CONTENT:
        language = "English"
    lang = LANGUAGE_CONTENT[language]

    with open(report_json_path) as f:
        report_data = json.load(f)

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
        leftMargin=1.5 * cm,
        rightMargin=1.5 * cm,
        title=lang["title"],
        author="AegisML Enterprise",
    )

    elements: list = []

    # Build sections
    elements.extend(_build_cover(lang, report_data))
    elements.extend(_build_executive_summary(lang, report_data))
    elements.append(Spacer(1, 0.3 * cm))
    elements.extend(_build_technical_section(lang, report_data))
    elements.append(PageBreak())
    elements.extend(_build_compliance_section(lang, report_data))
    elements.append(Spacer(1, 0.3 * cm))
    elements.extend(_build_integrity_section(lang, report_data))

    doc.build(elements)
    print(f"PDF generated ({language}): {output_path}")
    return output_path
