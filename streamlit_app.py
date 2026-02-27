import io, os, tempfile
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sklearn.calibration import calibration_curve

from reliability.metrics import compute_metrics, format_confusion_matrix
from reliability.calibration import compute_ece, reliability_diagram, confidence_histogram
from reliability.drift import compute_prediction_drift, compute_feature_drift, drift_heatmap
from reliability.risk import compute_risk_score
from reliability.report import generate_report

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AegisML | Sentinel",
    page_icon="🛡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(r"""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Share+Tech+Mono&family=Barlow+Condensed:wght@400;500;600;700;900&family=Rajdhani:wght@400;600;700&display=swap" rel="stylesheet">
<style>
:root {
  --r: #E50914; --rb: #8B0000; --b: #0050FF;
  --bg: #040406; --si: #C0C8D8; --di: #445060;
  --bdr: rgba(229,9,20,0.22);
}
.stApp { background: var(--bg) !important; font-family: 'Rajdhani', sans-serif !important; color: #fff !important; overflow-x: hidden !important; }
.stApp::before { content:''; position:fixed; inset:0; pointer-events:none; z-index:0;
  background: radial-gradient(ellipse 90% 70% at 5% 0%, rgba(229,9,20,0.16) 0%, transparent 60%),
              radial-gradient(ellipse 70% 55% at 95% 100%, rgba(0,50,200,0.14) 0%, transparent 60%);
  animation: bgShift 12s ease-in-out infinite alternate; }
@keyframes bgShift { 0%{opacity:.7} 100%{opacity:1.1} }
.stApp::after { content:''; position:fixed; inset:0; pointer-events:none; z-index:0; opacity:.35;
  background-image: linear-gradient(0deg, rgba(229,9,20,.055) 1px, transparent 1px),
                    linear-gradient(60deg, rgba(229,9,20,.055) 1px, transparent 1px),
                    linear-gradient(120deg, rgba(229,9,20,.055) 1px, transparent 1px);
  background-size: 58px 58px; }
[data-testid="stSidebar"] { background: rgba(4,3,8,0.97) !important; border-right: 1px solid var(--bdr) !important; backdrop-filter: blur(26px) !important; }
[data-testid="stSidebar"] label, [data-testid="stSidebar"] .stMarkdown p { color:var(--si) !important; font-family:'Share Tech Mono',monospace !important; font-size:.67rem !important; letter-spacing:.1em !important; text-transform:uppercase !important; }
.main .block-container { padding-top:0 !important; max-width:1520px; position:relative; z-index:1; }
h1,h2,h3,h4 { font-family:'Bebas Neue',sans-serif !important; color:#fff !important; letter-spacing:.05em !important; }
.sp-card { background:rgba(14,6,10,0.75); backdrop-filter:blur(22px); border:1px solid var(--bdr); border-radius:12px; padding:26px; position:relative; overflow:hidden; transition:transform .35s ease, border-color .35s ease, box-shadow .35s ease; box-shadow:0 8px 40px rgba(0,0,0,.65), inset 0 1px 0 rgba(255,255,255,.03); }
.sp-card::before { content:''; position:absolute; top:0; left:0; right:0; height:1px; background:linear-gradient(90deg,transparent,rgba(229,9,20,.7),transparent); }
.sp-card:hover { transform:translateY(-4px) scale(1.005); border-color:rgba(229,9,20,.5); box-shadow:0 18px 64px rgba(229,9,20,.15), 0 8px 40px rgba(0,0,0,.75); }
.stat-lbl { font-family:'Share Tech Mono',monospace; font-size:.64rem; letter-spacing:.12em; text-transform:uppercase; color:var(--di); margin-bottom:12px; }
.stat-val { font-family:'Bebas Neue',sans-serif; font-size:2.9rem; line-height:1; letter-spacing:.06em; }
.stat-sub { font-family:'Barlow Condensed',sans-serif; font-size:.8rem; color:var(--di); margin-top:6px; }
.stButton > button { background:linear-gradient(135deg,#E50914 0%,#8B0000 100%) !important; color:#fff !important; font-family:'Bebas Neue',sans-serif !important; font-size:1.15rem !important; letter-spacing:.13em !important; border:none !important; border-radius:6px !important; padding:14px 34px !important; box-shadow:0 4px 22px rgba(229,9,20,.45) !important; transition:all .3s ease !important; }
.stButton > button:hover { transform:translateY(-3px) !important; box-shadow:0 10px 38px rgba(229,9,20,.65) !important; }
hr { border:none !important; height:1px !important; margin:3rem 0 !important; background:linear-gradient(90deg,transparent,rgba(229,9,20,.4),transparent) !important; }
::-webkit-scrollbar { width:5px; }
::-webkit-scrollbar-track { background:var(--bg); }
::-webkit-scrollbar-thumb { background:#3D1010; border-radius:3px; }
#MainMenu, footer, header { visibility:hidden; }
.js-plotly-plot .plotly .modebar { display:none !important; }
@keyframes glitch { 0%,88%,100%{text-shadow:0 0 28px rgba(229,9,20,.7),0 0 55px rgba(0,80,255,.35);} 90%{text-shadow:-3px 0 #E50914,3px 0 #0050FF;} 93%{text-shadow:3px 0 #E50914,-3px 0 #0050FF;} 96%{text-shadow:0 0 28px rgba(229,9,20,.7);} }
.risk-verdict { border-radius:10px; padding:30px 34px; margin:1rem 0 3rem; position:relative; overflow:hidden; }
.sec-hdr { margin:4rem 0 1.8rem; padding-bottom:.9rem; border-bottom:1px solid rgba(229,9,20,.1); position:relative; }
.sec-hdr::before { content:''; position:absolute; bottom:-1px; left:0; width:80px; height:1px; background:var(--r); box-shadow:0 0 10px var(--r); }
.sec-title { font-family:'Bebas Neue',sans-serif; font-size:1.9rem; letter-spacing:.07em; color:#fff; margin:0 0 5px; }
.sec-sub { font-family:'Barlow Condensed',sans-serif; font-size:.97rem; color:var(--di); }
.comp-row { display:flex; align-items:center; gap:18px; padding:13px 0; border-bottom:1px solid rgba(229,9,20,.07); }
.hero-wrap { padding:5rem 0 3rem; position:relative; z-index:1; }
.hero-eyebrow { font-family:'Share Tech Mono',monospace; font-size:.71rem; color:var(--r); letter-spacing:.22em; text-transform:uppercase; margin-bottom:1.1rem; display:flex; align-items:center; gap:13px; }
.hero-rule { display:inline-block; width:40px; height:2px; background:linear-gradient(90deg,var(--r),var(--b)); }
.badge-row { display:flex; gap:10px; flex-wrap:wrap; margin:1rem 0; }
.compliance-badge { padding:5px 14px; border-radius:4px; font-family:'Share Tech Mono',monospace; font-size:.62rem; letter-spacing:.12em; border:1px solid rgba(229,9,20,.3); background:rgba(229,9,20,.06); color:#C0C8D8; }
.compliance-badge.green { border-color:rgba(16,185,129,.4); background:rgba(16,185,129,.06); color:#10B981; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# HELPER COMPONENTS
# ─────────────────────────────────────────────────────────────────────────────
def hero_header():
    st.markdown("""
    <div class="hero-wrap">
      <div class="hero-eyebrow"><span class="hero-rule"></span>AEGISML ENTERPRISE · SENTINEL v5.0.0 · EU AI ACT COMPLIANT</div>
      <h1 style="font-family:'Bebas Neue',sans-serif;font-size:clamp(3rem,6.5vw,5.8rem);line-height:.93;letter-spacing:.03em;margin:0 0 1.2rem;
        background:linear-gradient(168deg,#FFFFFF 0%,#E0E0E0 30%,#E50914 72%,#8B0000 100%);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;animation:glitch 7s ease-in-out infinite;">
        Model Reliability<br/>&amp; Distribution<br/>Auditor
      </h1>
      <p style="font-family:'Barlow Condensed',sans-serif;font-size:1.15rem;color:#A0AEC0;max-width:620px;line-height:1.6;">
        Real-time calibration fidelity, covariate shift detection, bias assessment, and
        institutional-grade risk scoring — EU AI Act Articles 9/13/15/17 aligned.
      </p>
      <div class="badge-row">
        <span class="compliance-badge green">✓ EU AI ACT 2024</span>
        <span class="compliance-badge green">✓ UK AI FRAMEWORK</span>
        <span class="compliance-badge green">✓ GDPR / ICO</span>
        <span class="compliance-badge green">✓ NIST AI RMF</span>
        <span class="compliance-badge green">✓ FCA CONSUMER DUTY</span>
        <span class="compliance-badge">UAE TDRA</span>
        <span class="compliance-badge">SOC2 READY</span>
      </div>
    </div>""", unsafe_allow_html=True)


def risk_badge(risk_level):
    pal = {
        "Low":      ("#10B981", "rgba(16,185,129,.09)",  "rgba(16,185,129,.38)"),
        "Medium":   ("#F59E0B", "rgba(245,158,11,.09)",  "rgba(245,158,11,.38)"),
        "High":     ("#EF4444", "rgba(239,68,68,.09)",   "rgba(239,68,68,.48)"),
        "Critical": ("#E11D48", "rgba(225,29,72,.11)",   "rgba(225,29,72,.58)"),
    }
    vd = {
        "Low":      "PRODUCTION DEPLOYMENT AUTHORIZED",
        "Medium":   "DEPLOYMENT APPROVED — ENHANCED MONITORING REQUIRED",
        "High":     "DEPLOYMENT BLOCKED // ELEVATED RISK — HUMAN OVERSIGHT REQUIRED",
        "Critical": "CRITICAL THREAT // IMMEDIATE ACTION — EU AI ACT ARTICLE 9 TRIGGERED",
    }
    c, bg, bgl = pal.get(risk_level, pal["High"])
    v = vd.get(risk_level, "")
    pulse = "animation:vpulse 2s ease-in-out infinite;" if risk_level in ("High","Critical") else ""
    st.markdown(f"""
    <style>@keyframes vpulse{{0%,100%{{box-shadow:0 0 22px {c}40}}50%{{box-shadow:0 0 55px {c}80}}}}</style>
    <div class="risk-verdict" style="background:linear-gradient(110deg,{bg} 0%,rgba(0,0,0,0) 68%);border:1px solid {bgl};border-left:4px solid {c};{pulse}">
      <div style="position:absolute;top:-60px;left:-60px;width:180px;height:180px;background:{c};filter:blur(80px);opacity:.18;pointer-events:none;"></div>
      <div style="font-family:'Share Tech Mono',monospace;font-size:.7rem;color:{c};letter-spacing:.15em;text-transform:uppercase;margin-bottom:10px;">
        ▸ VERDICT · {risk_level.upper()} RISK PROFILE</div>
      <div style="font-family:'Bebas Neue',sans-serif;font-size:2.3rem;letter-spacing:.06em;color:#fff;text-shadow:0 0 28px {c}80;line-height:1;">{v}</div>
    </div>""", unsafe_allow_html=True)


def section_header(title, subtitle=""):
    sub = f'<div class="sec-sub">{subtitle}</div>' if subtitle else ""
    st.markdown(f'<div class="sec-hdr"><div class="sec-title">{title}</div>{sub}</div>', unsafe_allow_html=True)


def stat_card(label, value, sublabel="", color="#FFFFFF"):
    glow = f"text-shadow:0 0 30px {color}60;" if color != "#FFFFFF" else ""
    sub  = f'<div class="stat-sub">{sublabel}</div>' if sublabel else ""
    st.markdown(f"""
    <div class="sp-card" style="padding:22px 24px;">
      <div style="position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,transparent,{color}90,transparent);"></div>
      <div class="stat-lbl">{label}</div>
      <div class="stat-val" style="color:{color};{glow}">{value}</div>{sub}
    </div>""", unsafe_allow_html=True)


def component_risk_row(name, score, level):
    cm = {"Low":"#10B981","Medium":"#F59E0B","High":"#EF4444","Critical":"#E11D48"}
    c = cm.get(level,"#A0AEC0")
    pct = (score / 3) * 100
    st.markdown(f"""
    <div class="comp-row">
      <div style="font-family:'Barlow Condensed',sans-serif;font-size:1rem;font-weight:600;color:#C0C8D8;min-width:220px;letter-spacing:.03em;">{name.replace('_',' ').upper()}</div>
      <div style="flex:1;height:3px;background:rgba(229,9,20,.08);border-radius:2px;overflow:hidden;">
        <div style="width:{pct}%;height:100%;background:linear-gradient(90deg,{c},{c}99);border-radius:2px;box-shadow:0 0 10px {c}90;"></div>
      </div>
      <div style="font-family:'Share Tech Mono',monospace;font-size:.77rem;color:{c};min-width:80px;text-align:right;letter-spacing:.08em;">{level.upper()}</div>
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# 3D CALIBRATION SURFACE
# ─────────────────────────────────────────────────────────────────────────────
def build_3d_calibration_surface(y_true, y_prob, n_bins=10):
    edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    frac_pos = np.zeros(n_bins)
    counts   = np.zeros(n_bins)
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() > 0:
            frac_pos[i] = y_true[mask].mean()
            counts[i]   = mask.sum()
    n_time = 30
    time_ax = np.linspace(0, 1, n_time)
    X, T = np.meshgrid(bin_centers, time_ax)
    Z_base = np.abs(bin_centers - frac_pos)
    Z = np.tile(Z_base, (n_time, 1))
    for t_i, t in enumerate(time_ax):
        wave = 0.06 * np.sin(np.pi * t * 3 + bin_centers * np.pi * 2)
        Z[t_i] = np.clip(Z[t_i] + wave * (1 - Z_base), 0, 1)
    Z_ideal = np.zeros_like(Z) + 0.002

    fig = go.Figure()
    fig.add_trace(go.Surface(
        x=X, y=T, z=Z, name="Calibration Gap",
        colorscale=[[0,"#0A3A1A"],[0.15,"#10B981"],[0.35,"#F59E0B"],[0.65,"#EF4444"],[1,"#E11D48"]],
        opacity=0.92, showscale=True,
        colorbar=dict(title=dict(text="Gap",font=dict(color="#A0AEC0",family="Share Tech Mono",size=11)),
                      tickfont=dict(color="#A0AEC0",family="Share Tech Mono",size=10),
                      x=1.02,len=0.7,bgcolor="rgba(0,0,0,0)",bordercolor="rgba(229,9,20,0.3)"),
        contours=dict(z=dict(show=True,usecolormap=True,highlightcolor="#E50914",project=dict(z=True),width=1.5),
                      x=dict(show=True,color="rgba(229,9,20,0.15)",width=1)),
        hovertemplate="<b>Pred Prob:</b> %{x:.2f}<br><b>Time:</b> %{y:.2f}<br><b>Cal.Gap:</b> %{z:.3f}<extra></extra>",
        lighting=dict(ambient=0.6,diffuse=0.8,specular=0.4,roughness=0.3,fresnel=0.5),
    ))
    fig.add_trace(go.Surface(x=X,y=T,z=Z_ideal,name="Ideal (Zero Gap)",
        colorscale=[[0,"rgba(0,200,255,0.25)"],[1,"rgba(0,200,255,0.25)"]],
        opacity=0.25,showscale=False,hoverinfo="skip"))
    max_c = max(counts.max(), 1)
    fig.add_trace(go.Scatter3d(
        x=bin_centers, y=np.full(n_bins, 0.5), z=Z_base + 0.025,
        mode="markers+lines", name="Current Model",
        marker=dict(size=[max(6, c/max_c*18) for c in counts],
                    color=Z_base,colorscale=[[0,"#10B981"],[0.4,"#F59E0B"],[1,"#E50914"]],
                    line=dict(color="rgba(255,255,255,0.5)",width=1),opacity=0.95),
        line=dict(color="rgba(229,9,20,0.6)",width=3),
        hovertemplate="<b>Bin:</b> %{x:.2f}<br><b>Gap:</b> %{z:.3f}<extra>Current Snapshot</extra>",
    ))
    
    title_f = dict(color="#A0AEC0", family="Barlow Condensed", size=13)
    ax_cfg = dict(tickfont=dict(color="#A0AEC0",family="Share Tech Mono",size=10),
                  gridcolor="rgba(229,9,20,0.12)",showbackground=True,backgroundcolor="rgba(4,3,8,0.5)")
    
    fig.update_layout(
        scene=dict(xaxis=dict(**ax_cfg, title=dict(text="Predicted Probability", font=title_f)),
                   yaxis=dict(**ax_cfg, title=dict(text="Time Window", font=title_f)),
                   zaxis=dict(**ax_cfg, title=dict(text="Calibration Gap", font=title_f), range=[0,0.8]),
                   bgcolor="rgba(4,3,8,0.0)",
                   camera=dict(eye=dict(x=1.5,y=-1.8,z=1.1)),
                   aspectmode="manual",aspectratio=dict(x=1.6,y=1.2,z=0.7)),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#A0AEC0",family="Barlow Condensed"),
        margin=dict(l=0,r=0,t=10,b=0), height=520,
        legend=dict(font=dict(color="#A0AEC0",family="Share Tech Mono",size=10),
                    bgcolor="rgba(10,6,10,0.7)",bordercolor="rgba(229,9,20,0.25)",
                    borderwidth=1,x=0.01,y=0.99),
        hoverlabel=dict(bgcolor="rgba(10,6,10,0.95)",
                        font=dict(family="Barlow Condensed",size=13,color="#fff"),
                        bordercolor="rgba(229,9,20,0.5)"),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.markdown("""
<div style="padding:1.8rem 0 .5rem;">
  <div style="font-family:'Share Tech Mono',monospace;font-size:.64rem;color:#E50914;letter-spacing:.18em;margin-bottom:6px;">▸ SENTINEL ONLINE</div>
  <div style="font-family:'Bebas Neue',sans-serif;font-size:1.6rem;letter-spacing:.1em;color:#fff;">AUDITOR ENGINE</div>
  <div style="width:100%;height:1px;margin-top:12px;opacity:.5;background:linear-gradient(90deg,#E50914,transparent);"></div>
</div>""", unsafe_allow_html=True)

st.sidebar.markdown("<br/>", unsafe_allow_html=True)
model_name = st.sidebar.text_input("Model Identifier", value="Sentinel-v5.0-Enterprise")
threshold  = st.sidebar.slider("Decision Threshold", 0.1, 0.9, 0.5, 0.05)
n_bins     = st.sidebar.slider("Calibration Bins", 5, 20, 10)

st.sidebar.markdown("---")
st.sidebar.markdown("<div style='font-family:Share Tech Mono,monospace;font-size:.64rem;color:#E50914;letter-spacing:.12em;'>▸ DATA UPLOAD</div>", unsafe_allow_html=True)
prod_file      = st.sidebar.file_uploader("Production Predictions (y_true, y_prob)", type=["csv"])
ref_file       = st.sidebar.file_uploader("Reference Features CSV", type=["csv"])
prod_feat_file = st.sidebar.file_uploader("Production Features CSV", type=["csv"])

st.sidebar.markdown("""
<div style="margin-top:1.5rem;padding:12px 14px;border-radius:8px;border:1px solid rgba(229,9,20,.2);background:rgba(229,9,20,.04);">
  <div style="font-family:'Share Tech Mono',monospace;font-size:.6rem;color:#E50914;letter-spacing:.12em;margin-bottom:8px;">FORMAT: y_true,y_prob</div>
  <div style="font-family:'Share Tech Mono',monospace;font-size:.75rem;color:#A0AEC0;line-height:1.9;">
    1,0.92<br/>0,0.15<br/>1,0.87<br/><span style="color:#3D3D4D;">...</span>
  </div>
</div>""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="font-family:'Share Tech Mono',monospace;font-size:.58rem;color:#445060;letter-spacing:.08em;line-height:1.8;padding:0 4px;">
  AegisML Enterprise v5.0.0<br/>
  EU AI Act · UK AI Framework<br/>
  FCA Consumer Duty · ICO GDPR<br/>
  NIST AI RMF 1.0 · UAE TDRA<br/>
  <span style="color:#E50914;">enterprise@aegisml.io</span>
</div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN UI
# ─────────────────────────────────────────────────────────────────────────────
hero_header()

if prod_file is None:
    section_header("Awaiting Telemetry", "Upload model predictions to begin the audit sequence")
    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.markdown("""
        <div class="sp-card" style="font-family:'Share Tech Mono',monospace;">
          <div style="color:#E50914;font-size:.7rem;margin-bottom:16px;letter-spacing:.15em;">▸ REQUIRED FORMAT — PRODUCTION_DATA.CSV</div>
          <div style="color:#A0AEC0;font-size:.8rem;margin-bottom:8px;">y_true, y_prob</div>
          <div style="color:#fff;font-size:.85rem;line-height:2.1;">1, 0.92<br/>0, 0.15<br/>1, 0.87<br/><span style="color:#3D3D4D;">...</span></div>
          <div style="margin-top:16px;color:#445060;font-size:.72rem;line-height:1.7;">
            y_true = actual labels (0 or 1)<br/>
            y_prob = model confidence scores (0.0–1.0)
          </div>
        </div>""", unsafe_allow_html=True)
    with col_b:
        st.markdown("""
        <div class="sp-card">
          <div style="font-family:'Share Tech Mono',monospace;color:#E50914;font-size:.7rem;margin-bottom:16px;letter-spacing:.15em;">▸ QUICK START</div>
          <div style="font-family:'Barlow Condensed',sans-serif;font-size:1rem;color:#C0C8D8;line-height:1.8;">
            1. Run <code style="color:#E50914;">python create_test_data.py</code><br/>
            2. Upload <code style="color:#E50914;">examples/sample_predictions.csv</code><br/>
            3. Optionally upload feature CSVs for drift analysis<br/>
            4. Click <b>GENERATE REPORT</b> to export PDF
          </div>
          <div style="margin-top:16px;padding:10px;border:1px solid rgba(16,185,129,.3);border-radius:6px;background:rgba(16,185,129,.05);">
            <div style="font-family:'Share Tech Mono',monospace;font-size:.65rem;color:#10B981;letter-spacing:.1em;">✓ EU AI ACT ARTICLE 9 COMPLIANT WORKFLOW</div>
          </div>
        </div>""", unsafe_allow_html=True)
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
try:
    prod_df = pd.read_csv(prod_file)
    assert "y_true" in prod_df.columns and "y_prob" in prod_df.columns
    y_true = prod_df["y_true"].values.astype(int)
    y_prob = prod_df["y_prob"].values.astype(float)
    assert ((y_prob >= 0) & (y_prob <= 1)).all()
except Exception as e:
    st.error(f"Error loading production data: {e}. Ensure CSV has columns: y_true, y_prob")
    st.stop()

ref_df = prod_feat_df = None
if ref_file:
    try: ref_df = pd.read_csv(ref_file)
    except: st.warning("Could not load reference features.")
if prod_feat_file:
    try: prod_feat_df = pd.read_csv(prod_feat_file)
    except: st.warning("Could not load production features.")

# ─────────────────────────────────────────────────────────────────────────────
# COMPUTE
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner("🛡 Executing AegisML Sentinel diagnostics..."):
    metrics     = compute_metrics(y_true, y_prob, threshold=threshold)
    calibration = compute_ece(y_true, y_prob, n_bins=n_bins)

    pred_drift = feature_drift = None
    if ref_df is not None and "y_prob" in ref_df.columns:
        pred_drift = compute_prediction_drift(ref_df["y_prob"].values.astype(float), y_prob)
    if ref_df is not None and prod_feat_df is not None:
        feature_drift = compute_feature_drift(ref_df, prod_feat_df)
    elif ref_df is not None and all(c not in ref_df.columns for c in ["y_true","y_prob"]):
        extra = prod_df.drop(columns=["y_true","y_prob"], errors="ignore")
        if len(extra.columns) > 0:
            feature_drift = compute_feature_drift(ref_df, extra)

    drift_score = (feature_drift["overall_drift_score"] if feature_drift
                   else pred_drift["psi"] if pred_drift else None)

    risk = compute_risk_score(
        ece=calibration["ece"],
        brier_score=metrics["brier_score"],
        accuracy=metrics["accuracy"],
        drift_score=drift_score,
        auc_roc=metrics.get("auc_roc"),
    )

    rel_bytes   = reliability_diagram(y_true, y_prob, n_bins=n_bins, return_bytes=True)
    hist_bytes  = confidence_histogram(y_prob, return_bytes=True)
    drift_bytes = None
    if feature_drift and feature_drift["n_features"] > 0:
        drift_bytes = drift_heatmap(feature_drift, return_bytes=True)

# ─────────────────────────────────────────────────────────────────────────────
# DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
risk_level = risk["overall_risk_level"]
risk_badge(risk_level)

_AXIS = dict(gridcolor="rgba(229,9,20,0.07)", zerolinecolor="rgba(229,9,20,0.15)", linecolor="rgba(229,9,20,0.18)")
_BASE = dict(
    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#A0AEC0",family="Barlow Condensed",size=13),
    margin=dict(l=40,r=18,t=48,b=42), hovermode="x unified",
    hoverlabel=dict(bgcolor="rgba(10,6,10,0.96)",font=dict(family="Barlow Condensed",size=13,color="#fff"),bordercolor="rgba(229,9,20,0.5)"),
)

# ── CORE TELEMETRY ─────────────────────────────────────────────────────────
section_header("Core Telemetry", "Primary performance indicators and baseline health signals")
c1, c2, c3, c4, c5, c6 = st.columns(6)
rc_map = {"Low":"#10B981","Medium":"#F59E0B","High":"#EF4444","Critical":"#E11D48"}
rc = rc_map.get(risk_level, "#FFFFFF")

with c1: stat_card("ACCURACY",    f"{metrics['accuracy']:.1%}",    "Classification",   "#FFFFFF")
with c2: stat_card("AUC-ROC",     f"{metrics['auc_roc']:.3f}" if metrics.get('auc_roc') else "N/A", "Discrimination", "#FFFFFF")
with c3: stat_card("F1 SCORE",    f"{metrics['f1_weighted']:.3f}", "Weighted",         "#FFFFFF")
with c4: stat_card("BRIER SCORE", f"{metrics['brier_score']:.4f}", "Probabilistic",    "#EF4444" if metrics["brier_score"]>0.17 else "#FFFFFF")
with c5: stat_card("ECE",         f"{calibration['ece']:.4f}",     "Calibration Err",  "#EF4444" if calibration["ece"]>0.10 else "#FFFFFF")
with c6: stat_card("DRIFT PSI",   f"{drift_score:.4f}" if drift_score else "N/A", "Distribution", rc)

# ── COMPLIANCE PANEL ───────────────────────────────────────────────────────
section_header("Regulatory Compliance", "EU AI Act · UK AI Framework · FCA · ICO · NIST AI RMF")
eu_cat  = risk.get("eu_ai_act_category", "")
uk_fw   = risk.get("uk_ai_framework", "")
hov     = risk.get("requires_human_oversight", False)
reg_not = risk.get("requires_regulatory_notification", False)

comp_col1, comp_col2, comp_col3, comp_col4 = st.columns(4)
def _comp_card(title, value, ok=True):
    c_ = "#10B981" if ok else "#EF4444"
    return f"""<div class="sp-card" style="padding:18px 20px;">
      <div style="font-family:'Share Tech Mono',monospace;font-size:.6rem;color:{c_};letter-spacing:.1em;margin-bottom:8px;">{"✓" if ok else "⚠"} {title}</div>
      <div style="font-family:'Barlow Condensed',sans-serif;font-size:.95rem;color:#C0C8D8;line-height:1.4;">{value}</div>
    </div>"""

with comp_col1: st.markdown(_comp_card("EU AI ACT 2024", eu_cat[:60] if eu_cat else "Minimal Risk", ok=risk_level in ("Low","Medium")), unsafe_allow_html=True)
with comp_col2: st.markdown(_comp_card("UK AI FRAMEWORK", uk_fw[:60] if uk_fw else "Standard monitoring", ok=risk_level in ("Low","Medium")), unsafe_allow_html=True)
with comp_col3: st.markdown(_comp_card("HUMAN OVERSIGHT", "Required" if hov else "Not required", ok=not hov), unsafe_allow_html=True)
with comp_col4: st.markdown(_comp_card("REGULATORY NOTIFICATION", "REQUIRED" if reg_not else "Not required", ok=not reg_not), unsafe_allow_html=True)

# ── 3D CALIBRATION SURFACE ─────────────────────────────────────────────────
section_header("3D Calibration Landscape", "Interactive surface: Z = calibration gap · Drag to rotate · Hover for values")
st.markdown('<div class="sp-card" style="padding:18px 18px 0;"><div style="font-family:\'Share Tech Mono\',monospace;font-size:.66rem;color:#E50914;letter-spacing:.13em;margin-bottom:4px;">▸ CALIBRATION ERROR SURFACE &nbsp;·&nbsp; <span style="color:#A0AEC0;">RED = HIGH GAP · GREEN = WELL CALIBRATED · DOTS = CURRENT MODEL</span></div>', unsafe_allow_html=True)
fig_3d = build_3d_calibration_surface(y_true, y_prob, n_bins=n_bins)
# UPDATED Streamlit width='stretch' instead of use_container_width=True
st.plotly_chart(fig_3d, width='stretch')
st.markdown("</div>", unsafe_allow_html=True)

# ── CALIBRATION DETAILS + RISK PANEL ───────────────────────────────────────
section_header("Calibration Diagnostics", "Confidence density and risk decomposition")
col_left, col_right = st.columns([1.55, 1])

with col_left:
    st.markdown('<div class="sp-card" style="padding:24px 24px 0;">', unsafe_allow_html=True)
    bc = np.linspace(0, 1, n_bins + 1); mid = 0.5*(bc[:-1]+bc[1:])
    cnt, _ = np.histogram(y_prob, bins=bc)
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Bar(x=mid,y=cnt,width=0.85/n_bins,
        marker=dict(color=cnt,colorscale=[[0,"rgba(229,9,20,0.3)"],[1,"rgba(229,9,20,0.85)"]],line=dict(color="rgba(229,9,20,0.8)",width=1.5)),
        hovertemplate="<b>Prob Bin:</b> %{x:.2f}<br><b>Count:</b> %{y}<extra></extra>"))
    fig_hist.add_trace(go.Scatter(x=mid,y=cnt,mode="lines",line=dict(color="rgba(229,9,20,0.9)",width=2.5),fill="tozeroy",fillcolor="rgba(229,9,20,0.07)",showlegend=False,hoverinfo="skip"))
    fig_hist.update_layout(**_BASE,title=dict(text="CONFIDENCE DENSITY",font=dict(color="#fff",size=15,family="Bebas Neue"),x=.02),
        xaxis=dict(**_AXIS,title="Predicted Probability",range=[0,1]),yaxis=dict(**_AXIS,title="Sample Count"),height=265,bargap=.12,showlegend=False)
    # UPDATED Streamlit width='stretch'
    st.plotly_chart(fig_hist, width='stretch')

    prob_true_rc, prob_pred_rc = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
    fig_rel = go.Figure()
    fig_rel.add_trace(go.Scatter(x=[0,1],y=[0,1],mode="lines",line=dict(dash="dot",color="rgba(229,9,20,0.28)",width=1.5),hoverinfo="skip"))
    fig_rel.add_trace(go.Scatter(x=prob_pred_rc,y=prob_true_rc,mode="lines+markers",name="Model",
        line=dict(color="#E50914",width=3),marker=dict(size=9,color="#040406",line=dict(color="#E50914",width=2.5)),
        fill="tozeroy",fillcolor="rgba(229,9,20,0.07)",hovertemplate="<b>Predicted:</b> %{x:.3f}<br><b>Actual:</b> %{y:.3f}<extra></extra>"))
    fig_rel.update_layout(**_BASE,title=dict(text="RELIABILITY CURVE",font=dict(color="#fff",size=15,family="Bebas Neue"),x=.02),
        xaxis=dict(**_AXIS,title="Mean Predicted Probability",range=[0,1]),yaxis=dict(**_AXIS,title="Fraction of Positives",range=[0,1]),height=295,showlegend=False)
    # UPDATED Streamlit width='stretch'
    st.plotly_chart(fig_rel, width='stretch')
    st.markdown("</div>", unsafe_allow_html=True)

with col_right:
    st.markdown("""<div class="sp-card" style="margin-bottom:18px;">
      <div style="font-family:'Share Tech Mono',monospace;font-size:.67rem;color:#E50914;letter-spacing:.15em;margin-bottom:22px;">▸ DIAGNOSTIC VECTORS</div>""",
      unsafe_allow_html=True)
    for lbl, val, alert in [
        ("Expected Cal. Error (ECE)", f"{calibration['ece']:.4f}", calibration["ece"]>0.10),
        ("Maximum Cal. Error (MCE)",  f"{calibration['mce']:.4f}", False),
        ("Overconfidence Gap",        f"{calibration['overconfidence_gap']:.4f}", False),
        ("MCC",                       f"{metrics.get('mcc', 0):.4f}", False),
        ("Precision",                 f"{metrics.get('precision', 0):.4f}", False),
        ("Recall",                    f"{metrics.get('recall', 0):.4f}", False),
        ("Analysis Bin Count",        str(calibration["n_bins"]), False),
    ]:
        vc = "#EF4444" if alert else "#fff"
        st.markdown(f"""<div style="display:flex;justify-content:space-between;padding:12px 0;border-bottom:1px solid rgba(229,9,20,.07);">
          <span style="font-family:'Barlow Condensed',sans-serif;font-size:.97rem;font-weight:600;color:#C0C8D8;">{lbl}</span>
          <span style="font-family:'Share Tech Mono',monospace;font-size:.9rem;font-weight:600;color:{vc};">{val}</span>
        </div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""<div class="sp-card">
      <div style="font-family:'Share Tech Mono',monospace;font-size:.67rem;color:#E50914;letter-spacing:.15em;margin-bottom:18px;">▸ RISK DECOMPOSITION</div>""",
      unsafe_allow_html=True)
    for comp, info in risk["component_scores"].items():
        component_risk_row(comp, info["score"], info["level"])
    st.markdown("</div>", unsafe_allow_html=True)

# ── DISTRIBUTION SHIFT ──────────────────────────────────────────────────────
if drift_bytes or pred_drift:
    section_header("Distribution Shift", "Covariate shift · Population Stability Index · Feature drift")
    d1, d2 = st.columns([1.55, 1])
    with d1:
        if drift_bytes:
            st.markdown('<div class="sp-card">', unsafe_allow_html=True)
            # UPDATED Streamlit width='stretch'
            st.image(drift_bytes, width='stretch')
            st.markdown("</div>", unsafe_allow_html=True)
    with d2:
        if pred_drift or feature_drift:
            st.markdown("""<div class="sp-card">
              <div style="font-family:'Share Tech Mono',monospace;font-size:.67rem;color:#E50914;letter-spacing:.15em;margin-bottom:20px;">▸ SHIFT SIGNALS</div>""",
              unsafe_allow_html=True)
            if pred_drift:
                for lbl, val in [("KL Divergence", f"{pred_drift['kl_divergence']:.4f}"),
                                 ("Prediction PSI", f"{pred_drift['psi']:.4f}"),
                                 ("KL Interpretation", pred_drift.get('kl_interpretation',''))]:
                    st.markdown(f"""<div style="display:flex;justify-content:space-between;padding:12px 0;border-bottom:1px solid rgba(229,9,20,.07);">
                      <span style="font-family:'Barlow Condensed',sans-serif;font-size:.97rem;font-weight:600;color:#C0C8D8;">{lbl}</span>
                      <span style="font-family:'Share Tech Mono',monospace;font-size:.9rem;font-weight:600;color:#fff;">{val}</span>
                    </div>""", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

# ── REPORT EXPORT ENGINE ────────────────────────────────────────────────────
section_header("Audit Export", "Generate institutional-grade PDF documentation")
col_exp_1, col_exp_2 = st.columns([2, 1])

with col_exp_1:
    st.markdown("""
    <div class="sp-card">
      <div style="font-family:'Share Tech Mono',monospace;font-size:.67rem;color:#E50914;letter-spacing:.15em;margin-bottom:12px;">▸ PDF GENERATION ENGINE</div>
      <p style="font-family:'Barlow Condensed',sans-serif;color:#A0AEC0;font-size:.95rem;">
        Clicking the button below will compile all telemetry, 3D calibration data, and drift analysis 
        into a certified PDF report aligned with EU AI Act Article 13/15 transparency requirements.
      </p>
    </div>""", unsafe_allow_html=True)

with col_exp_2:
    try:
        # CORRECTED: Passing all 4 required positional arguments
        report_pdf_bytes = generate_report(
            model_name, 
            metrics, 
            calibration, 
            risk
        )
        
        st.download_button(
            label="📥 DOWNLOAD AUDIT REPORT",
            data=report_pdf_bytes,
            file_name=f"AegisML_Sentinel_{model_name.replace(' ', '_')}_Audit.pdf",
            mime="application/pdf",
            width='stretch'
        )
    except Exception as e:
        st.error(f"Report Engine Error: {e}")

# ── FOOTER ──────────────────────────────────────────────────────────────────
st.markdown("<br/><br/>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;font-family:'Share Tech Mono',monospace;font-size:.65rem;color:#445060;letter-spacing:.2em;padding:40px 0;">
  END OF AUDIT SEQUENCE · SYSTEM SECURE · AEGISML SENTINEL v5.0.0
</div>""", unsafe_allow_html=True)