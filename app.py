"""
FutureFormers: AI-Powered Route Integrity QA System
=====================================================
Production-style Quality Assurance system for ride-hailing routing infrastructure.
Detects route continuity & connectivity errors using Feature Extraction,
Rule-Based Validation (Option B), Isolation Forest ML, and combined Decision Logic.

Team: FutureFormers | IIT Mandi Hackathon 3.0 | Problem Statement 2
Version: 4.0.0
"""

import streamlit as st
import pandas as pd
import numpy as np
import folium
import json
import io
from streamlit_folium import st_folium
from shapely import wkt
from shapely.geometry import Point, LineString
import re
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# =============================================================================
# CONFIGURATION
# =============================================================================

APP_CONFIG = {
    "title": "Route Integrity QA System",
    "version": "4.0.0",
    "icon": "🛣️",
    "precision": 6,
    "team": "FutureFormers",
}

# =============================================================================
# DEMO DATA — Problem 2 streets_xgen.wkt
# =============================================================================
DEMO_WKT_DATA = """LINESTRING(7071.421606 8585.627528, 7074.672945 8588.813669, 7074.902551 8589.026835, 7075.084535 8589.196346, 7079.265638 8592.884220, 7081.745726 8594.906230, 7087.160750 8598.968277, 7091.114457 8601.621165, 7097.122261 8605.189984, 7104.911244 8609.486740, 7105.304126 8609.703307, 7105.767874 8609.959559, 7144.536643 8631.196101, 7154.035087 8636.400567, 7154.349676 8636.571269, 7154.424397 8636.611861, 7163.899654 8641.749543, 7194.100989 8658.211975, 7204.788850 8664.037795, 7217.453480 8671.075087, 7225.937575 8676.277228, 7229.984882 8679.124346, 7234.367017 8682.488447, 7238.456504 8685.821480, 7243.451150 8690.254299, 7247.565354 8694.303307, 7251.962457 8699.059843, 7255.123087 8702.735811, 7259.170961 8707.866520, 7262.826576 8713.000630, 7262.874709 8713.070759, 7263.406488 8713.854425, 7268.192504 8721.195591, 7269.545140 8723.350488)
LINESTRING(7228.943036 8691.704220, 7230.263868 8692.301820)
LINESTRING(7230.130016 8690.909669, 7236.635528 8693.947843, 7238.203087 8694.558992, 7246.566879 8698.129852)
LINESTRING(7233.052082 8685.026022, 7233.114331 8684.900731)
LINESTRING(7235.343383 8681.162627, 7235.951754 8680.340183)
LINESTRING(7233.114331 8684.900731, 7234.209751 8682.695150, 7234.367017 8682.488447, 7234.542142 8682.245745, 7235.343383 8681.162627)
LINESTRING(7230.130016 8690.909669, 7233.052082 8685.026022)
LINESTRING(7255.510243 8699.099074, 7250.881096 8693.964737, 7245.560806 8688.592006, 7239.691219 8683.390375, 7235.951754 8680.340183)
LINESTRING(7219.235906 8686.145197, 7230.130016 8690.909669)
LINESTRING(7171.129701 8650.784296, 7172.734677 8650.679301, 7173.495099 8650.929033, 7174.214079 8651.165216, 7192.126998 8660.929323, 7197.898450 8664.075326)
LINESTRING(7150.309569 8645.723660, 7151.809323 8646.372283, 7152.455452 8646.687836)
LINESTRING(7178.488951 8636.167899, 7179.095849 8636.294381)
LINESTRING(7235.951754 8680.340183, 7235.062809 8679.661965, 7231.893165 8677.230009, 7226.680365 8673.607106, 7220.970935 8670.106261, 7215.024189 8666.676907, 7210.789172 8664.323641, 7203.506230 8660.303943, 7191.076082 8653.528460, 7187.602110 8651.634803, 7183.533657 8649.417090, 7179.077877 8646.988365, 7173.680598 8644.046343)
LINESTRING(7173.680598 8644.046343, 7178.431691 8637.105203, 7179.218646 8635.065449)
LINESTRING(7152.922545 8639.584441, 7153.182312 8638.969380)
LINESTRING(7152.922545 8639.584441, 7153.400580 8639.845512, 7155.481039 8640.973587, 7159.220787 8643.001380, 7165.913783 8646.640781, 7168.378110 8647.984063, 7170.192680 8648.973184, 7170.323414 8649.225921, 7170.529550 8649.624302, 7171.129701 8650.784296)
LINESTRING(7149.314835 8641.775055, 7151.010860 8642.887597, 7150.309569 8645.723660)
LINESTRING(7159.331735 8627.200894, 7160.009443 8627.429027, 7160.622293 8627.964718)
LINESTRING(7160.622293 8627.964718, 7161.134457 8628.255496, 7161.850828 8628.826280, 7163.425814 8629.276309, 7164.344580 8629.632567, 7165.696139 8630.103005, 7169.024466 8631.261468, 7171.403754 8632.280239, 7173.416693 8633.310123, 7174.659005 8634.026778, 7175.660031 8634.604195, 7177.199414 8635.492233, 7178.007798 8635.958532, 7178.488951 8636.167899)
LINESTRING(7160.535723 8626.610494, 7160.009443 8627.429027, 7159.280315 8628.578646, 7155.343729 8634.079106)
LINESTRING(7155.846879 8634.352252, 7154.349676 8636.571269, 7153.182312 8638.969380)
LINESTRING(7155.343729 8634.079106, 7156.744044 8634.838564, 7157.181770 8635.075880, 7158.321865 8635.694060, 7164.473783 8639.029814, 7171.316220 8642.757487, 7173.680598 8644.046343)
LINESTRING(7155.846879 8634.352252, 7155.343729 8634.079106)
LINESTRING(7160.535723 8626.610494, 7163.093197 8627.735509, 7169.459698 8630.137304, 7172.826746 8631.579005, 7178.738627 8634.989310, 7179.218646 8635.065449)
LINESTRING(7152.922545 8639.584441, 7146.890306 8636.283666, 7136.488517 8630.585235, 7134.319162 8629.396894, 7124.383899 8624.969065, 7123.323969 8624.777953, 7121.539899 8624.753065, 7118.794885 8624.141065, 7118.036787 8624.066457, 7117.057928 8624.132050, 7116.004913 8624.447093)
LINESTRING(7116.004913 8624.447093, 7112.647899 8624.312050, 7111.216913 8624.249065, 7107.446608 8624.105802, 7101.497594 8623.934816, 7098.702406 8624.024447)
LINESTRING(7097.387528 8625.209953, 7096.202759 8628.525808, 7096.091868 8628.836088, 7094.616094 8632.063616, 7094.575502 8632.152454, 7094.528787 8632.254614, 7094.032838 8633.339263, 7097.047370 8635.481575)
LINESTRING(7098.702406 8624.024447, 7099.318488 8623.211528)
LINESTRING(7097.387528 8625.209953, 7098.702406 8624.024447)
LINESTRING(7089.693165 8635.350614, 7093.515969 8630.246551, 7097.387528 8625.209953)
LINESTRING(7104.226620 8612.107370, 7104.311093 8611.937461)
LINESTRING(7099.318488 8623.211528, 7104.226620 8612.107370)
LINESTRING(7097.887446 8624.084598, 7083.675950 8619.934734, 7065.840416 8611.314803)
LINESTRING(7065.840416 8611.314803, 7065.689216 8610.036491, 7065.330009 8608.768101)
LINESTRING(7065.849090 8609.703591, 7066.134992 8610.118299, 7083.707584 8618.672806, 7085.072580 8619.137802, 7099.318488 8623.211528)
LINESTRING(7065.849090 8609.703591, 7065.689216 8610.036491, 7065.592441 8610.237921)
LINESTRING(7115.152479 8607.998154, 7118.572479 8608.358154, 7119.388517 8608.574154, 7124.440479 8611.886154, 7128.544535 8614.742117, 7129.348498 8615.426117, 7136.590847 8621.963206, 7137.610186 8622.521575, 7138.110047 8622.795345, 7155.354331 8626.387465, 7159.331735 8627.200894)
LINESTRING(7155.343729 8634.079106, 7154.315490 8633.515861, 7153.494236 8633.065890, 7144.602803 8628.194211, 7134.657279 8622.746306, 7128.239131 8619.230608, 7122.178772 8615.910898, 7116.403011 8612.747093, 7112.053701 8610.364630, 7108.643225 8608.496485, 7106.827238 8606.270154)
LINESTRING(7106.625298 8606.725342, 7106.827238 8606.270154)
LINESTRING(7106.625298 8606.725342, 7105.304126 8609.703307, 7104.311093 8611.937461)
LINESTRING(7106.827238 8606.270154, 7109.809455 8599.548132, 7110.782362 8597.356157)
LINESTRING(7079.423924 8595.277739, 7078.770085 8595.916724, 7076.599654 8596.376220)
LINESTRING(7076.599654 8596.376220, 7073.598444 8593.672479)
LINESTRING(7073.598444 8593.672479, 7072.557335 8594.218658, 7071.476598 8594.855376)
LINESTRING(7102.205405 8610.012227, 7095.419206 8606.216466, 7089.858992 8602.878217, 7084.575269 8599.029789, 7079.423924 8595.277739)
LINESTRING(7074.076422 8591.860120, 7073.526784 8591.375282)
LINESTRING(7072.623950 8594.066154, 7072.557335 8594.218658, 7065.849090 8609.703591)
LINESTRING(7073.526784 8591.375282, 7074.141449 8590.748598, 7074.097682 8590.508277)
LINESTRING(7104.226620 8612.107370, 7097.580850 8608.486054, 7094.466028 8606.711339, 7090.380907 8604.284655, 7087.550343 8602.438507, 7083.294236 8599.398803, 7081.232315 8597.852050, 7078.770085 8595.916724, 7074.076422 8591.860120)
LINESTRING(7072.623950 8594.066154, 7073.270646 8592.708359, 7074.076422 8591.860120)
LINESTRING(7076.039187 8586.425254, 7076.445109 8585.496170)
LINESTRING(7073.644989 8591.195792, 7074.097682 8590.508277, 7074.902551 8589.026835, 7076.039187 8586.425254)
LINESTRING(7076.445109 8585.496170, 7078.460542 8580.883238)
LINESTRING(7078.460542 8580.883238, 7088.449323 8586.079937, 7107.773102 8595.942917, 7107.972548 8596.047402, 7110.782362 8597.356157)"""


# =============================================================================
# CUSTOM CSS — GLASSMORPHISM + DM SANS
# =============================================================================

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,100..1000;1,9..40,100..1000&display=swap');

    :root {
        --primary: #6366f1; --primary-dark: #4f46e5; --primary-light: #818cf8;
        --secondary: #06b6d4; --success: #10b981; --warning: #f59e0b; --danger: #ef4444;
        --dark: #0f172a; --gray-900: #1e293b; --gray-800: #334155; --gray-600: #475569;
        --gray-400: #94a3b8; --gray-200: #e2e8f0; --gray-100: #f1f5f9; --white: #ffffff;
        --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1);
        --shadow-xl: 0 20px 25px -5px rgb(0 0 0 / 0.1), 0 8px 10px -6px rgb(0 0 0 / 0.1);
    }

    *, html, body, [class*="css"] {
        font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        background-attachment: fixed;
    }

    .main .block-container { padding: 2rem 2rem 3rem 2rem; max-width: 1400px; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(15,23,42,0.97) 0%, rgba(30,41,59,0.97) 100%);
        backdrop-filter: blur(20px); border-right: 1px solid rgba(255,255,255,0.1);
    }
    [data-testid="stSidebar"] * { color: var(--gray-100) !important; }
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 { color: var(--white) !important; font-weight: 600 !important; }
    [data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.15) !important; margin: 1.5rem 0; }

    /* Hero */
    .hero-container {
        background: rgba(255,255,255,0.92); backdrop-filter: blur(24px);
        border: 1px solid rgba(255,255,255,0.5); border-radius: 32px;
        padding: 2.5rem 2rem; margin-bottom: 2rem; text-align: center;
        box-shadow: var(--shadow-xl); position: relative; overflow: hidden;
    }
    .hero-container::before {
        content: ''; position: absolute; top: 0; left: 0; right: 0; height: 4px;
        background: linear-gradient(90deg, var(--primary), var(--secondary), var(--primary-light));
        background-size: 200% 100%; animation: grad 3s ease infinite;
    }
    @keyframes grad { 0%,100%{background-position:0% 50%} 50%{background-position:100% 50%} }
    .hero-icon { font-size: 3.5rem; margin-bottom: 0.25rem; animation: float 3s ease-in-out infinite; }
    @keyframes float { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-10px)} }
    .hero-title {
        font-size: 2.75rem; font-weight: 800;
        background: linear-gradient(135deg, var(--primary), var(--primary-dark), #7c3aed);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text; margin: 0 0 0.4rem 0; letter-spacing: -0.02em;
    }
    .hero-subtitle { font-size: 1.15rem; color: var(--gray-600); font-weight: 500; margin: 0; }
    .hero-badge {
        display: inline-flex; align-items: center; gap: 0.5rem;
        padding: 0.4rem 1rem; background: linear-gradient(135deg, var(--primary), var(--primary-dark));
        color: white !important; border-radius: 100px; font-size: 0.8rem; font-weight: 600;
        margin-top: 0.8rem; box-shadow: 0 4px 14px 0 rgba(99,102,241,0.4);
    }

    /* Metric Grid */
    .metric-grid { display: grid; grid-template-columns: repeat(5,1fr); gap: 1rem; margin-bottom: 1.5rem; }
    @media(max-width:1024px){ .metric-grid{grid-template-columns:repeat(3,1fr)} }
    @media(max-width:640px){ .metric-grid{grid-template-columns:repeat(2,1fr)} }
    .metric-card {
        background: rgba(255,255,255,0.92); backdrop-filter: blur(16px);
        border: 1px solid rgba(255,255,255,0.5); border-radius: 20px;
        padding: 1.25rem; text-align: center; box-shadow: var(--shadow-lg);
        transition: all 0.3s cubic-bezier(0.4,0,0.2,1); position: relative; overflow: hidden;
    }
    .metric-card::before { content:''; position:absolute; top:0; left:0; right:0; height:3px; border-radius:20px 20px 0 0; }
    .metric-card.purple::before { background: linear-gradient(90deg,#8b5cf6,#a78bfa); }
    .metric-card.blue::before { background: linear-gradient(90deg,#3b82f6,#60a5fa); }
    .metric-card.red::before { background: linear-gradient(90deg,#ef4444,#f87171); }
    .metric-card.orange::before { background: linear-gradient(90deg,#f59e0b,#fbbf24); }
    .metric-card.green::before { background: linear-gradient(90deg,#10b981,#34d399); }
    .metric-card:hover { transform: translateY(-5px) scale(1.02); box-shadow: 0 20px 40px -12px rgba(0,0,0,0.2); }
    .metric-icon { font-size: 1.75rem; margin-bottom: 0.3rem; }
    .metric-value { font-size: 2.2rem; font-weight: 800; color: var(--gray-900); line-height: 1; margin-bottom: 0.15rem; }
    .metric-label { font-size: 0.8rem; font-weight: 600; color: var(--gray-600); text-transform: uppercase; letter-spacing: 0.05em; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.4rem; background: rgba(255,255,255,0.6); backdrop-filter: blur(12px);
        border-radius: 16px; padding: 0.4rem; border: 1px solid rgba(255,255,255,0.3);
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent; border-radius: 12px; padding: 0.65rem 1.25rem;
        font-weight: 600; color: var(--gray-600); transition: all 0.2s ease;
    }
    .stTabs [data-baseweb="tab"]:hover { background: rgba(99,102,241,0.1); color: var(--primary); }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, var(--primary), var(--primary-dark)) !important;
        color: white !important; box-shadow: 0 4px 14px 0 rgba(99,102,241,0.4);
    }
    .stTabs [data-baseweb="tab-panel"] {
        background: rgba(255,255,255,0.87); backdrop-filter: blur(16px);
        border-radius: 24px; padding: 1.75rem; margin-top: 0.75rem;
        border: 1px solid rgba(255,255,255,0.4); box-shadow: var(--shadow-lg);
    }

    /* Section Header */
    .section-header { display:flex; align-items:center; gap:0.75rem; margin-bottom:1rem; }
    .section-header h3 { font-size:1.4rem; font-weight:700; color:var(--gray-900); margin:0; }
    .section-header .icon { font-size:1.5rem; }

    /* Map Container */
    .map-container { border-radius:20px; overflow:hidden; box-shadow:var(--shadow-lg); border:3px solid rgba(255,255,255,0.5); }

    /* Tables */
    .stDataFrame { border-radius:16px; overflow:hidden; box-shadow:0 4px 6px -1px rgb(0 0 0/0.1); }

    /* Buttons */
    .stButton button {
        background: linear-gradient(135deg, var(--primary), var(--primary-dark));
        color: white; border: none; border-radius: 12px; padding: 0.7rem 1.4rem;
        font-weight: 600; font-size: 0.9rem; transition: all 0.3s cubic-bezier(0.4,0,0.2,1);
        box-shadow: 0 4px 14px 0 rgba(99,102,241,0.4);
    }
    .stButton button:hover { transform: translateY(-2px); box-shadow: 0 6px 20px 0 rgba(99,102,241,0.5); }
    .stDownloadButton button {
        background: linear-gradient(135deg, #10b981, #059669);
        box-shadow: 0 4px 14px 0 rgba(16,185,129,0.4);
    }

    /* File Uploader */
    [data-testid="stFileUploader"] {
        background: rgba(255,255,255,0.1); border-radius: 12px; padding: 0.5rem;
        border: 2px dashed rgba(255,255,255,0.3);
    }

    /* Alerts */
    .stAlert { border-radius:16px; border:none; backdrop-filter:blur(8px); }
    .stAlert > div { border-radius:16px; }

    /* Welcome */
    .welcome-card {
        background: rgba(255,255,255,0.92); backdrop-filter: blur(20px);
        border: 1px solid rgba(255,255,255,0.5); border-radius: 28px;
        padding: 3rem; text-align: center; box-shadow: var(--shadow-xl);
    }
    .welcome-icon { font-size:5rem; margin-bottom:1rem; animation: pulse 2s ease-in-out infinite; }
    @keyframes pulse { 0%,100%{transform:scale(1)} 50%{transform:scale(1.05)} }
    .welcome-title { font-size:2rem; font-weight:700; color:var(--gray-900); margin-bottom:1rem; }
    .welcome-text { font-size:1.05rem; color:var(--gray-600); line-height:1.7; max-width:600px; margin:0 auto 2rem auto; }
    .step-list { display:flex; flex-direction:column; gap:0.85rem; max-width:480px; margin:0 auto; text-align:left; }
    .step-item {
        display:flex; align-items:center; gap:1rem; padding:0.9rem 1.1rem;
        background: linear-gradient(135deg, rgba(99,102,241,0.08), rgba(139,92,246,0.08));
        border-radius:14px; transition:all 0.2s ease;
    }
    .step-item:hover { transform:translateX(8px); background:linear-gradient(135deg, rgba(99,102,241,0.15), rgba(139,92,246,0.15)); }
    .step-number {
        width:30px; height:30px; background:linear-gradient(135deg, var(--primary), var(--primary-dark));
        color:white; border-radius:10px; display:flex; align-items:center; justify-content:center;
        font-weight:700; font-size:0.8rem; flex-shrink:0;
    }
    .step-text { font-size:0.95rem; font-weight:500; color:var(--gray-800); }

    /* Stats grid */
    .stats-grid { display:grid; grid-template-columns:repeat(2,1fr); gap:1.25rem; }
    .stat-card {
        background:linear-gradient(135deg, rgba(99,102,241,0.05), rgba(139,92,246,0.05));
        border-radius:16px; padding:1.25rem; border:1px solid rgba(99,102,241,0.1);
    }
    .stat-card h4 { font-size:0.95rem; font-weight:700; color:var(--primary); margin:0 0 0.75rem 0; display:flex; align-items:center; gap:0.5rem; }
    .stat-item { display:flex; justify-content:space-between; align-items:center; padding:0.4rem 0; border-bottom:1px solid rgba(99,102,241,0.1); }
    .stat-item:last-child { border-bottom:none; }
    .stat-label { font-size:0.85rem; color:var(--gray-600); }
    .stat-value { font-size:0.95rem; font-weight:700; color:var(--gray-900); }

    /* Quality Score */
    .quality-score { background:rgba(255,255,255,0.9); border-radius:20px; padding:1.75rem; text-align:center; margin-top:1.25rem; border:1px solid rgba(255,255,255,0.3); }
    .score-circle {
        width:130px; height:130px; border-radius:50%; display:flex; flex-direction:column;
        align-items:center; justify-content:center; margin:0 auto 0.75rem auto;
        font-weight:800; font-size:2.25rem; color:white; box-shadow:var(--shadow-lg);
    }
    .score-circle.excellent { background:linear-gradient(135deg,#10b981,#059669); }
    .score-circle.good { background:linear-gradient(135deg,#f59e0b,#d97706); }
    .score-circle.poor { background:linear-gradient(135deg,#ef4444,#dc2626); }
    .score-label { font-size:0.8rem; font-weight:600; text-transform:uppercase; letter-spacing:0.05em; margin-top:0.4rem; color:var(--gray-600); }

    /* Pipeline card */
    .pipeline-card {
        background: rgba(255,255,255,0.9); border-radius: 20px; padding: 1.5rem;
        border: 1px solid rgba(99,102,241,0.15); margin-bottom: 1rem;
    }
    .pipeline-step {
        display: flex; align-items: flex-start; gap: 1rem; padding: 0.75rem 0;
        border-bottom: 1px solid rgba(99,102,241,0.08);
    }
    .pipeline-step:last-child { border-bottom: none; }
    .pipeline-badge {
        width: 28px; height: 28px; border-radius: 8px; display: flex; align-items: center;
        justify-content: center; font-weight: 700; font-size: 0.75rem; flex-shrink: 0; color: white;
    }
    .pipeline-badge.extract { background: linear-gradient(135deg, #8b5cf6, #7c3aed); }
    .pipeline-badge.rules { background: linear-gradient(135deg, #f59e0b, #d97706); }
    .pipeline-badge.ml { background: linear-gradient(135deg, #06b6d4, #0891b2); }
    .pipeline-badge.decision { background: linear-gradient(135deg, #10b981, #059669); }

    /* Scrollbar & misc */
    #MainMenu {visibility:hidden;} footer {visibility:hidden;}
    html { scroll-behavior: smooth; }
    ::-webkit-scrollbar { width:8px; height:8px; }
    ::-webkit-scrollbar-track { background:rgba(0,0,0,0.1); border-radius:10px; }
    ::-webkit-scrollbar-thumb { background:linear-gradient(180deg, var(--primary), var(--primary-dark)); border-radius:10px; }
</style>
"""


# =============================================================================
# CORE ENGINE — Feature Extraction
# =============================================================================

class FeatureExtractor:
    """Extract geometric features per LINESTRING for QA analysis."""

    def __init__(self, lines: List[LineString], precision: int = 6):
        self.lines = lines
        self.precision = precision
        self._endpoint_map: Dict[Tuple, List[int]] = defaultdict(list)
        self._build_endpoint_map()

    def _round(self, pt: Tuple) -> Tuple:
        return (round(pt[0], self.precision), round(pt[1], self.precision))

    def _build_endpoint_map(self):
        for idx, line in enumerate(self.lines):
            coords = list(line.coords)
            self._endpoint_map[self._round(coords[0])].append(idx)
            self._endpoint_map[self._round(coords[-1])].append(idx)

    def extract_all(self) -> pd.DataFrame:
        """Return a DataFrame with one row per LINESTRING and all features."""
        rows = []
        all_endpoints = []
        for line in self.lines:
            c = list(line.coords)
            all_endpoints.append(self._round(c[0]))
            all_endpoints.append(self._round(c[-1]))

        for idx, line in enumerate(self.lines):
            coords = list(line.coords)
            length = line.length
            n_vertices = len(coords)
            vertex_density = n_vertices / length if length > 0 else 0
            start = self._round(coords[0])
            end = self._round(coords[-1])

            # Connectivity: how many OTHER segments share each endpoint
            start_degree = len(self._endpoint_map[start])
            end_degree = len(self._endpoint_map[end])

            # Min distance from each endpoint to nearest OTHER line
            start_pt = Point(start)
            end_pt = Point(end)
            min_dist_start = float('inf')
            min_dist_end = float('inf')
            for j, other in enumerate(self.lines):
                if j == idx:
                    continue
                d_s = start_pt.distance(other)
                d_e = end_pt.distance(other)
                if d_s < min_dist_start:
                    min_dist_start = d_s
                if d_e < min_dist_end:
                    min_dist_end = d_e
            connectivity_score = min(min_dist_start, min_dist_end)

            rows.append({
                'geometry_id': idx + 1,
                'length': round(length, 4),
                'n_vertices': n_vertices,
                'vertex_density': round(vertex_density, 6),
                'start_x': start[0], 'start_y': start[1],
                'end_x': end[0], 'end_y': end[1],
                'start_degree': start_degree,
                'end_degree': end_degree,
                'connectivity_score': round(connectivity_score, 4),
            })

        return pd.DataFrame(rows)


# =============================================================================
# CORE ENGINE — Rule-Based Validator (Option B)
# =============================================================================

class RuleBasedValidator:
    """
    Data-driven rule-based validation.  No hardcoded thresholds — all derived
    from percentile analysis of the dataset itself.
    """

    def validate(self, features: pd.DataFrame) -> List[Dict]:
        """Run all rule checks and return list of issue dicts."""
        issues: List[Dict] = []

        # --- Rule 1: Orphaned / Isolated Segments (low connectivity) --------
        # A segment whose BOTH endpoints connect to nothing else (degree == 1 each)
        # AND connectivity_score is high means it is spatially orphaned.
        conn_threshold = np.percentile(features['connectivity_score'], 75)
        for _, row in features.iterrows():
            if row['start_degree'] <= 1 and row['end_degree'] <= 1 and row['connectivity_score'] > conn_threshold:
                issues.append({
                    'geometry_id': int(row['geometry_id']),
                    'error_type': 'ISOLATED_SEGMENT',
                    'description': 'Both endpoints are disconnected from the rest of the network',
                    'connectivity_score': float(row['connectivity_score']),
                    'start': (row['start_x'], row['start_y']),
                    'end': (row['end_x'], row['end_y']),
                    'confidence': min(1.0, row['connectivity_score'] / (conn_threshold * 2)),
                    'source': 'rule',
                })

        # --- Rule 2: Extremely Short Segments (percentile-based) -------------
        p5 = np.percentile(features['length'], 5)
        median_len = np.median(features['length'])
        short_threshold = max(p5, median_len * 0.02)  # whichever is more generous
        for _, row in features.iterrows():
            if row['length'] <= short_threshold:
                issues.append({
                    'geometry_id': int(row['geometry_id']),
                    'error_type': 'UNREALISTICALLY_SHORT',
                    'description': f"Segment length {row['length']:.4f} is below the 5th-percentile threshold ({short_threshold:.4f})",
                    'length': float(row['length']),
                    'start': (row['start_x'], row['start_y']),
                    'end': (row['end_x'], row['end_y']),
                    'confidence': min(1.0, 1 - row['length'] / short_threshold) if short_threshold > 0 else 1.0,
                    'source': 'rule',
                })

        # --- Rule 3: Endpoint Gaps — Broken Continuity (adaptive) -----------
        # For each dangling endpoint (degree == 1), measure gap to nearest road.
        # Flag if gap > 0 but < adaptive threshold (dataset-derived).
        gap_values = features.loc[
            (features['start_degree'] == 1) | (features['end_degree'] == 1),
            'connectivity_score'
        ]
        if len(gap_values) > 0:
            gap_threshold = np.percentile(gap_values[gap_values > 0], 50) if len(gap_values[gap_values > 0]) > 0 else 5.0
        else:
            gap_threshold = 5.0

        for _, row in features.iterrows():
            # Check start endpoint
            if row['start_degree'] == 1 and 0 < row['connectivity_score'] < gap_threshold:
                issues.append({
                    'geometry_id': int(row['geometry_id']),
                    'error_type': 'ENDPOINT_GAP',
                    'description': f"Dangling start endpoint is {row['connectivity_score']:.4f} units from nearest road (threshold: {gap_threshold:.4f})",
                    'gap_distance': float(row['connectivity_score']),
                    'endpoint': (row['start_x'], row['start_y']),
                    'start': (row['start_x'], row['start_y']),
                    'end': (row['end_x'], row['end_y']),
                    'confidence': min(1.0, 1 - row['connectivity_score'] / gap_threshold),
                    'source': 'rule',
                })
            if row['end_degree'] == 1 and 0 < row['connectivity_score'] < gap_threshold:
                # Avoid duplicate if both endpoints dangling with same score
                existing = [i for i in issues if i['geometry_id'] == int(row['geometry_id']) and i['error_type'] == 'ENDPOINT_GAP']
                if not existing:
                    issues.append({
                        'geometry_id': int(row['geometry_id']),
                        'error_type': 'ENDPOINT_GAP',
                        'description': f"Dangling end endpoint is {row['connectivity_score']:.4f} units from nearest road (threshold: {gap_threshold:.4f})",
                        'gap_distance': float(row['connectivity_score']),
                        'endpoint': (row['end_x'], row['end_y']),
                        'start': (row['start_x'], row['start_y']),
                        'end': (row['end_x'], row['end_y']),
                        'confidence': min(1.0, 1 - row['connectivity_score'] / gap_threshold),
                        'source': 'rule',
                    })

        return issues


# =============================================================================
# CORE ENGINE — ML Anomaly Detection (Isolation Forest)
# =============================================================================

class AnomalyDetector:
    """Unsupervised anomaly detection using Isolation Forest on extracted features."""

    def __init__(self, contamination: float = 0.15):
        self.contamination = contamination
        self.model = None
        self.scaler = StandardScaler()

    def detect(self, features: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict]]:
        """
        Train Isolation Forest on dataset and return anomaly flags + issues.
        """
        feat_cols = ['length', 'n_vertices', 'vertex_density', 'connectivity_score',
                     'start_degree', 'end_degree']
        X = features[feat_cols].values

        X_scaled = self.scaler.fit_transform(X)

        self.model = IsolationForest(
            contamination=self.contamination,
            n_estimators=100,
            random_state=42,
        )
        preds = self.model.fit_predict(X_scaled)
        scores = self.model.decision_function(X_scaled)

        features = features.copy()
        features['ml_anomaly'] = (preds == -1).astype(int)
        features['ml_score'] = np.round(-scores, 4)  # higher = more anomalous

        issues: List[Dict] = []
        for _, row in features[features['ml_anomaly'] == 1].iterrows():
            issues.append({
                'geometry_id': int(row['geometry_id']),
                'error_type': 'ML_ANOMALY',
                'description': f"Isolation Forest flagged as geometric outlier (score: {row['ml_score']:.4f})",
                'ml_score': float(row['ml_score']),
                'start': (row['start_x'], row['start_y']),
                'end': (row['end_x'], row['end_y']),
                'confidence': min(1.0, float(row['ml_score']) / 0.5) if row['ml_score'] > 0 else 0.3,
                'source': 'ml',
            })

        return features, issues


# =============================================================================
# CORE ENGINE — Decision Logic
# =============================================================================

class DecisionEngine:
    """Combines rule-based and ML flags into a single QA report with confidence."""

    @staticmethod
    def combine(rule_issues: List[Dict], ml_issues: List[Dict]) -> List[Dict]:
        """Merge and deduplicate issues, boosting confidence for double-flagged segments."""
        by_id: Dict[int, List[Dict]] = defaultdict(list)
        for issue in rule_issues + ml_issues:
            by_id[issue['geometry_id']].append(issue)

        final: List[Dict] = []
        seen_keys = set()

        for gid, items in by_id.items():
            sources = set(i['source'] for i in items)
            both = 'rule' in sources and 'ml' in sources

            for item in items:
                key = (item['geometry_id'], item['error_type'])
                if key in seen_keys:
                    continue
                seen_keys.add(key)

                entry = dict(item)
                if both:
                    entry['confidence'] = min(1.0, entry.get('confidence', 0.5) * 1.3)
                    entry['confirmed_by'] = 'rule+ml'
                else:
                    entry['confirmed_by'] = entry['source']

                # Severity label
                c = entry['confidence']
                entry['severity'] = 'HIGH' if c >= 0.7 else ('MEDIUM' if c >= 0.4 else 'LOW')

                final.append(entry)

        final.sort(key=lambda x: -x['confidence'])
        return final


# =============================================================================
# CORE ENGINE — QA Report Builder
# =============================================================================

def build_error_report(issues: List[Dict]) -> Dict:
    """Build the error_report.json structure per hackathon spec."""
    report_items = []
    for issue in issues:
        report_items.append({
            'geometry_id': issue['geometry_id'],
            'error_type': issue['error_type'],
            'severity': issue.get('severity', 'MEDIUM'),
            'confidence': round(issue.get('confidence', 0.5), 4),
            'description': issue.get('description', ''),
            'coordinates': {
                'start': list(issue.get('start', (0, 0))),
                'end': list(issue.get('end', (0, 0))),
            },
            'confirmed_by': issue.get('confirmed_by', issue.get('source', 'rule')),
        })
    return {
        'report_version': APP_CONFIG['version'],
        'team': APP_CONFIG['team'],
        'total_issues': len(report_items),
        'issues': report_items,
    }


# =============================================================================
# WKT PARSER
# =============================================================================

def parse_wkt(wkt_text: str) -> List[LineString]:
    """Parse WKT text to list of valid LineStrings."""
    cleaned = re.sub(r'\s+', ' ', wkt_text)
    matches = re.findall(r'LINESTRING\s*\([^)]+\)', cleaned, re.IGNORECASE)
    lines = []
    for m in matches:
        try:
            geom = wkt.loads(m)
            if isinstance(geom, LineString) and geom.is_valid and not geom.is_empty:
                lines.append(geom)
        except Exception:
            continue
    return lines


# =============================================================================
# MAP VISUALIZATION
# =============================================================================

def create_map(lines: List[LineString], issues: List[Dict]) -> folium.Map:
    """Create a beautiful Folium map showing the network and flagged issues."""
    if not lines:
        return folium.Map(location=[0, 0], zoom_start=2, tiles='cartodbpositron')

    # Compute bounds
    all_x, all_y = [], []
    for line in lines:
        for c in line.coords:
            all_x.append(c[0]); all_y.append(c[1])
    cx, cy = np.mean(all_x), np.mean(all_y)
    scale = 0.0001

    def norm(x, y):
        return [(y - cy) * scale, (x - cx) * scale]

    m = folium.Map(location=[0, 0], zoom_start=15, tiles='cartodbdark_matter')
    folium.TileLayer('cartodbpositron', name='Light').add_to(m)

    # Flagged geometry IDs
    flagged_ids = set(i['geometry_id'] for i in issues)
    severity_map = {}
    for i in issues:
        gid = i['geometry_id']
        if gid not in severity_map or i['severity'] == 'HIGH':
            severity_map[gid] = i['severity']

    # Draw roads
    road_group = folium.FeatureGroup(name='🛣️ Road Network')
    error_road_group = folium.FeatureGroup(name='⚠️ Flagged Segments')
    for idx, line in enumerate(lines):
        gid = idx + 1
        coords = [norm(c[0], c[1]) for c in line.coords]
        if gid in flagged_ids:
            sev = severity_map.get(gid, 'MEDIUM')
            color = '#ef4444' if sev == 'HIGH' else '#f59e0b' if sev == 'MEDIUM' else '#60a5fa'
            folium.PolyLine(coords, color=color, weight=4, opacity=0.95,
                tooltip=f"Segment #{gid} — {sev}").add_to(error_road_group)
        else:
            folium.PolyLine(coords, color='#60a5fa', weight=2.5, opacity=0.7,
                tooltip=f"Segment #{gid}").add_to(road_group)
    road_group.add_to(m)
    error_road_group.add_to(m)

    # Error markers
    if issues:
        marker_group = folium.FeatureGroup(name='📍 Error Locations')
        for i, issue in enumerate(issues):
            loc = norm(issue['start'][0], issue['start'][1])
            sev = issue.get('severity', 'MEDIUM')
            color = '#ef4444' if sev == 'HIGH' else '#f59e0b' if sev == 'MEDIUM' else '#94a3b8'
            icon_sym = '🔴' if sev == 'HIGH' else '🟠' if sev == 'MEDIUM' else '🔵'

            # Glow
            folium.CircleMarker(location=loc, radius=16, color=color, fill=True,
                fillColor=color, fillOpacity=0.2, weight=0).add_to(marker_group)
            # Marker
            folium.CircleMarker(location=loc, radius=8, color='white', weight=2,
                fill=True, fillColor=color, fillOpacity=0.95,
                popup=folium.Popup(
                    f"""<div style="font-family:'DM Sans',sans-serif;min-width:200px;padding:4px;">
                    <h4 style="color:{color};margin:0 0 8px;font-weight:700;">{icon_sym} {issue['error_type']}</h4>
                    <div style="background:#f8fafc;border-radius:8px;padding:8px;">
                        <p style="margin:4px 0;color:#334155;font-size:0.85rem;"><b>Segment:</b> #{issue['geometry_id']}</p>
                        <p style="margin:4px 0;color:#334155;font-size:0.85rem;"><b>Confidence:</b> {issue.get('confidence',0):.0%}</p>
                        <p style="margin:4px 0;color:#334155;font-size:0.85rem;"><b>Severity:</b>
                            <span style="background:{color};color:white;padding:1px 8px;border-radius:12px;font-size:0.75rem;">{sev}</span></p>
                        <p style="margin:4px 0;color:#334155;font-size:0.85rem;"><b>Source:</b> {issue.get('confirmed_by','')}</p>
                    </div></div>""", max_width=240),
                tooltip=f"#{issue['geometry_id']} {issue['error_type']} ({sev})"
            ).add_to(marker_group)
        marker_group.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    all_coords = [norm(c[0], c[1]) for line in lines for c in line.coords]
    m.fit_bounds(all_coords)
    return m


# =============================================================================
# UI COMPONENTS
# =============================================================================

def render_hero():
    st.markdown("""
        <div class="hero-container">
            <div class="hero-icon">🛣️</div>
            <h1 class="hero-title">Route Integrity QA</h1>
            <p class="hero-subtitle">AI-powered quality assurance for ride-hailing routing infrastructure</p>
            <div class="hero-badge"><span>✨</span><span>v4.0 • FutureFormers • IIT Mandi Hackathon</span></div>
        </div>
    """, unsafe_allow_html=True)


def render_metrics(stats: Dict, issues: List[Dict]):
    total = stats['total_segments']
    n_issues = len(issues)
    high = sum(1 for i in issues if i.get('severity') == 'HIGH')
    med = sum(1 for i in issues if i.get('severity') == 'MEDIUM')
    ml_count = sum(1 for i in issues if i.get('source') == 'ml' or i.get('confirmed_by') == 'rule+ml')
    st.markdown(f"""
        <div class="metric-grid">
            <div class="metric-card purple"><div class="metric-icon">🛣️</div><div class="metric-value">{total}</div><div class="metric-label">Segments</div></div>
            <div class="metric-card blue"><div class="metric-icon">🔍</div><div class="metric-value">{n_issues}</div><div class="metric-label">Issues Found</div></div>
            <div class="metric-card red"><div class="metric-icon">🚨</div><div class="metric-value">{high}</div><div class="metric-label">High Severity</div></div>
            <div class="metric-card orange"><div class="metric-icon">⚡</div><div class="metric-value">{med}</div><div class="metric-label">Medium Severity</div></div>
            <div class="metric-card green"><div class="metric-icon">🤖</div><div class="metric-value">{ml_count}</div><div class="metric-label">ML Flagged</div></div>
        </div>
    """, unsafe_allow_html=True)


def render_pipeline_info():
    st.markdown("""
        <div class="pipeline-card">
            <h4 style="margin:0 0 0.75rem;color:#1e293b;font-weight:700;">🔬 Detection Pipeline</h4>
            <div class="pipeline-step">
                <div class="pipeline-badge extract">A</div>
                <div><b style="color:#1e293b;">Feature Extraction</b><br><span style="color:#64748b;font-size:0.85rem;">Length, vertices, vertex density, connectivity score per segment</span></div>
            </div>
            <div class="pipeline-step">
                <div class="pipeline-badge rules">B</div>
                <div><b style="color:#1e293b;">Rule-Based Validation</b><br><span style="color:#64748b;font-size:0.85rem;">Orphaned segments, percentile-based short links, adaptive gap thresholds</span></div>
            </div>
            <div class="pipeline-step">
                <div class="pipeline-badge ml">C</div>
                <div><b style="color:#1e293b;">ML Anomaly Detection</b><br><span style="color:#64748b;font-size:0.85rem;">Isolation Forest trained on dataset — identifies geometric outliers</span></div>
            </div>
            <div class="pipeline-step">
                <div class="pipeline-badge decision">D</div>
                <div><b style="color:#1e293b;">Decision Logic</b><br><span style="color:#64748b;font-size:0.85rem;">Combine rule + ML flags, boost confidence for double-flagged, assign severity</span></div>
            </div>
        </div>
    """, unsafe_allow_html=True)


def render_stats(stats: Dict, issues: List[Dict]):
    st.markdown("""
        <div class="stats-grid">
            <div class="stat-card">
                <h4>📏 Network Metrics</h4>
                <div class="stat-item"><span class="stat-label">Total Length</span><span class="stat-value">{:,.2f}</span></div>
                <div class="stat-item"><span class="stat-label">Avg Segment</span><span class="stat-value">{:.2f}</span></div>
                <div class="stat-item"><span class="stat-label">Shortest</span><span class="stat-value">{:.4f}</span></div>
                <div class="stat-item"><span class="stat-label">Longest</span><span class="stat-value">{:.2f}</span></div>
            </div>
            <div class="stat-card">
                <h4>🔗 Topology</h4>
                <div class="stat-item"><span class="stat-label">Total Nodes</span><span class="stat-value">{}</span></div>
                <div class="stat-item"><span class="stat-label">Connected</span><span class="stat-value">{}</span></div>
                <div class="stat-item"><span class="stat-label">Dangling</span><span class="stat-value">{}</span></div>
                <div class="stat-item"><span class="stat-label">Segments</span><span class="stat-value">{}</span></div>
            </div>
        </div>
    """.format(
        stats['total_length'], stats['avg_length'], stats['min_length'], stats['max_length'],
        stats['total_endpoints'], stats['connected_nodes'], stats['dangling_nodes'], stats['total_segments']
    ), unsafe_allow_html=True)

    # Quality Score
    if stats['total_segments'] > 0:
        error_rate = len(issues) / stats['total_segments'] * 100
        quality = max(0, 100 - error_rate * 8)
        sc = "excellent" if quality >= 85 else "good" if quality >= 60 else "poor"
        sl = "Excellent" if quality >= 85 else "Good" if quality >= 60 else "Needs Attention"
        se = "🌟" if quality >= 85 else "👍" if quality >= 60 else "🔧"
        st.markdown(f"""
            <div class="quality-score">
                <h3 style="margin:0 0 0.75rem;color:#1e293b;font-weight:700;">Network Quality Score</h3>
                <div class="score-circle {sc}"><span>{quality:.0f}%</span></div>
                <div class="score-label">{se} {sl}</div>
            </div>
        """, unsafe_allow_html=True)


def render_issue_table(issues: List[Dict]):
    if not issues:
        st.markdown("""
            <div style="text-align:center;padding:2.5rem;background:linear-gradient(135deg,rgba(16,185,129,0.1),rgba(5,150,105,0.1));border-radius:16px;">
                <div style="font-size:3.5rem;margin-bottom:0.75rem;">✅</div>
                <h3 style="color:#059669;margin:0 0 0.4rem;font-weight:700;">All Clear!</h3>
                <p style="color:#64748b;margin:0;">No route integrity issues detected.</p>
            </div>
        """, unsafe_allow_html=True)
        return

    rows = []
    for i in issues:
        rows.append({
            'Seg #': i['geometry_id'],
            'Error Type': i['error_type'],
            'Severity': i.get('severity', 'MEDIUM'),
            'Confidence': f"{i.get('confidence', 0):.0%}",
            'Source': i.get('confirmed_by', i.get('source', '')),
            'Description': i.get('description', '')[:80],
        })
    df = pd.DataFrame(rows)
    df.index = df.index + 1
    df.index.name = '#'

    def style_sev(val):
        if val == 'HIGH':
            return 'background:#fef2f2;color:#991b1b;font-weight:600;'
        elif val == 'MEDIUM':
            return 'background:#fffbeb;color:#92400e;font-weight:600;'
        return 'background:#f0f9ff;color:#1e40af;font-weight:600;'

    styled = df.style.map(style_sev, subset=['Severity'])
    st.dataframe(styled, use_container_width=True, height=420)


def render_feature_table(features: pd.DataFrame):
    """Show the extracted feature matrix."""
    display_cols = ['geometry_id', 'length', 'n_vertices', 'vertex_density',
                    'connectivity_score', 'start_degree', 'end_degree']
    extra_cols = [c for c in ['ml_anomaly', 'ml_score'] if c in features.columns]
    df = features[display_cols + extra_cols].copy()
    df.columns = ['Seg #', 'Length', 'Vertices', 'Vertex Density',
                   'Connectivity', 'Start Deg', 'End Deg'] + \
                  (['ML Flag', 'ML Score'] if extra_cols else [])

    def style_anomaly(val):
        if val == 1:
            return 'background:#fef2f2;color:#991b1b;font-weight:700;'
        return ''

    styled = df.style.format({
        'Length': '{:.4f}', 'Vertex Density': '{:.6f}',
        'Connectivity': '{:.4f}',
    })
    if 'ML Flag' in df.columns:
        styled = styled.map(style_anomaly, subset=['ML Flag'])
    st.dataframe(styled, use_container_width=True, height=420)


def compute_stats(lines: List[LineString], features: pd.DataFrame) -> Dict:
    """Compute summary stats from lines & features."""
    lengths = [l.length for l in lines]
    # Build endpoint map for topology stats
    ep_count: Dict[Tuple, int] = defaultdict(int)
    prec = APP_CONFIG['precision']
    for line in lines:
        c = list(line.coords)
        s = (round(c[0][0], prec), round(c[0][1], prec))
        e = (round(c[-1][0], prec), round(c[-1][1], prec))
        ep_count[s] += 1
        ep_count[e] += 1

    return {
        'total_segments': len(lines),
        'total_length': round(sum(lengths), 2),
        'avg_length': round(np.mean(lengths), 2),
        'min_length': round(min(lengths), 4),
        'max_length': round(max(lengths), 2),
        'total_endpoints': len(ep_count),
        'connected_nodes': sum(1 for c in ep_count.values() if c > 1),
        'dangling_nodes': sum(1 for c in ep_count.values() if c == 1),
    }


def render_welcome():
    st.markdown("""
        <div class="welcome-card">
            <div class="welcome-icon">🛣️</div>
            <h2 class="welcome-title">Route Integrity QA System</h2>
            <p class="welcome-text">
                AI-powered quality assurance for ride-hailing routing infrastructure.
                Detect broken routes, isolated segments, and connectivity gaps using
                rule-based validation and machine learning — with zero manual labeling.
            </p>
            <div class="step-list">
                <div class="step-item"><div class="step-number">1</div><span class="step-text">Upload a .wkt file or load the demo dataset</span></div>
                <div class="step-item"><div class="step-number">2</div><span class="step-text">One-click analysis — no parameter tuning needed</span></div>
                <div class="step-item"><div class="step-number">3</div><span class="step-text">Explore flagged issues on the interactive map</span></div>
                <div class="step-item"><div class="step-number">4</div><span class="step-text">Download error_report.json for your records</span></div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    with st.expander("📖 Supported Format"):
        st.code("LINESTRING(x1 y1, x2 y2, x3 y3, ...)\nLINESTRING(x1 y1, x2 y2, ...)", language="text")
        st.markdown("Each line should be a valid WKT LINESTRING. Coordinates in any projected CRS.")


def render_sidebar() -> Tuple[Optional[str], float]:
    with st.sidebar:
        st.markdown("""
            <div style="text-align:center;padding:1rem 0;">
                <div style="font-size:2.5rem;margin-bottom:0.4rem;">🛣️</div>
                <h2 style="margin:0;font-weight:800;font-size:1.4rem;">Route QA System</h2>
                <p style="margin:0.2rem 0 0;font-size:0.8rem;opacity:0.7;">v4.0 • FutureFormers</p>
            </div>
        """, unsafe_allow_html=True)

        st.divider()
        st.markdown("### 📁 Data Input")
        uploaded = st.file_uploader("Upload .wkt / .txt", type=['wkt', 'txt'],
            help="LINESTRING geometries")
        use_demo = st.button("🎯 Load Demo Data", type="primary", use_container_width=True)

        st.divider()
        st.markdown("### ⚙️ ML Parameters")
        contamination = st.slider("Anomaly Sensitivity", 0.05, 0.30, 0.15, 0.01,
            help="Isolation Forest contamination — higher = flag more segments")

        st.divider()
        st.markdown("### 💡 Detection Pipeline")
        st.markdown("""
        <div style="font-size:0.82rem;line-height:1.6;">
            <p><b>A.</b> Feature Extraction</p>
            <p><b>B.</b> Rule-Based Validation</p>
            <p><b>C.</b> Isolation Forest ML</p>
            <p><b>D.</b> Combined Decision Logic</p>
        </div>
        """, unsafe_allow_html=True)

        st.divider()
        st.markdown("""
            <div style="text-align:center;font-size:0.72rem;opacity:0.6;">
                FutureFormers • IIT Mandi Hackathon 3.0<br>Problem Statement 2<br>© 2026
            </div>
        """, unsafe_allow_html=True)

    wkt_data = None
    if use_demo:
        wkt_data = DEMO_WKT_DATA
        st.session_state['data_source'] = 'demo'
    elif uploaded is not None:
        wkt_data = uploaded.read().decode('utf-8')
        st.session_state['data_source'] = 'upload'
    elif st.session_state.get('data_source') == 'demo':
        wkt_data = DEMO_WKT_DATA

    return wkt_data, contamination


# =============================================================================
# MAIN
# =============================================================================

def main():
    st.set_page_config(
        page_title=APP_CONFIG['title'],
        page_icon=APP_CONFIG['icon'],
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    if 'data_source' not in st.session_state:
        st.session_state['data_source'] = None

    wkt_data, contamination = render_sidebar()
    render_hero()

    if wkt_data:
        with st.spinner("🔬 Running full QA pipeline..."):
            # 1. Parse
            lines = parse_wkt(wkt_data)
            if not lines:
                st.error("No valid LINESTRING geometries found in file.")
                return

            # 2. Feature Extraction
            extractor = FeatureExtractor(lines, APP_CONFIG['precision'])
            features = extractor.extract_all()

            # 3. Rule-Based Validation
            validator = RuleBasedValidator()
            rule_issues = validator.validate(features)

            # 4. ML Anomaly Detection
            detector = AnomalyDetector(contamination=contamination)
            features, ml_issues = detector.detect(features)

            # 5. Decision Logic
            all_issues = DecisionEngine.combine(rule_issues, ml_issues)

            # 6. Stats
            stats = compute_stats(lines, features)

            # 7. Build report
            report = build_error_report(all_issues)

        # --- Metrics Dashboard ---
        render_metrics(stats, all_issues)

        # --- Tabs ---
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🗺️ Map View",
            "📋 QA Report",
            "🧬 Feature Matrix",
            "📊 Statistics",
            "🔬 Pipeline"
        ])

        with tab1:
            st.markdown("""
                <div class="section-header"><span class="icon">🗺️</span><h3>Interactive Network Map</h3></div>
                <p style="color:#64748b;margin-bottom:0.75rem;font-size:0.9rem;">
                    Red/Orange segments are flagged • Click markers for details • Toggle layers top-right
                </p>
            """, unsafe_allow_html=True)
            st.markdown('<div class="map-container">', unsafe_allow_html=True)
            map_obj = create_map(lines, all_issues)
            st_folium(map_obj, height=550, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with tab2:
            st.markdown("""<div class="section-header"><span class="icon">📋</span><h3>Detected Issues</h3></div>""", unsafe_allow_html=True)
            render_issue_table(all_issues)
            if all_issues:
                st.markdown("<br>", unsafe_allow_html=True)
                col1, col2 = st.columns(2)
                with col1:
                    csv_data = pd.DataFrame([{
                        'Segment': i['geometry_id'], 'Type': i['error_type'],
                        'Severity': i.get('severity',''), 'Confidence': i.get('confidence',0),
                        'Source': i.get('confirmed_by',''), 'Description': i.get('description','')
                    } for i in all_issues]).to_csv(index=False)
                    st.download_button("📥 Download CSV", csv_data, "qa_error_report.csv", "text/csv", use_container_width=True)
                with col2:
                    json_str = json.dumps(report, indent=2)
                    st.download_button("📥 Download error_report.json", json_str, "error_report.json", "application/json", use_container_width=True)

        with tab3:
            st.markdown("""<div class="section-header"><span class="icon">🧬</span><h3>Feature Matrix</h3></div>
                <p style="color:#64748b;margin-bottom:0.75rem;font-size:0.9rem;">
                    Extracted features per segment — ML Flag = 1 means Isolation Forest anomaly
                </p>
            """, unsafe_allow_html=True)
            render_feature_table(features)

        with tab4:
            st.markdown("""<div class="section-header"><span class="icon">📊</span><h3>Network Statistics</h3></div>""", unsafe_allow_html=True)
            render_stats(stats, all_issues)

        with tab5:
            st.markdown("""<div class="section-header"><span class="icon">🔬</span><h3>Detection Pipeline</h3></div>
                <p style="color:#64748b;margin-bottom:0.75rem;font-size:0.9rem;">
                    Four-stage pipeline: no hardcoded thresholds, no manual labeling, fully data-driven.
                </p>
            """, unsafe_allow_html=True)
            render_pipeline_info()

            st.markdown("---")
            st.markdown("##### 📄 Full error_report.json Preview")
            st.json(report)

    else:
        render_welcome()


if __name__ == "__main__":
    main()
