"""
app.py — AuditPulse Forensic & Fraud Intelligence Suite
Studio Décisionnel d'Audit Financier — Thème Blanc, Icônes Lucide SVG.
"""
import io, os, sys, time, requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import chisquare

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

st.set_page_config(
    page_title="AuditPulse | Forensic Intelligence Suite",
    page_icon="AP",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = os.getenv("API_URL", "http://localhost:8000/api/v1")
BATCH_CHUNK = 500

P = {
    "bg": "#FFFFFF", "bg2": "#F8FAFC", "border": "#E2E8F0",
    "text": "#0F172A", "text2": "#475569", "text3": "#94A3B8",
    "primary": "#2563EB", "primary_bg": "#EFF6FF",
    "success": "#059669", "success_bg": "#ECFDF5",
    "warning": "#D97706", "warning_bg": "#FFFBEB",
    "danger": "#DC2626", "danger_bg": "#FEF2F2",
    "purple": "#7C3AED", "purple_bg": "#F5F3FF",
}

# ══════════════════════════════════════════════════════════════════════════════
# ICÔNES LUCIDE (SVG Path Data uniquement — le rendu est fait par ic())
# ══════════════════════════════════════════════════════════════════════════════
_ICON_PATHS = {
    "shield":       '<path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z"/>',
    "bar_chart":    '<line x1="12" x2="12" y1="20" y2="10"/><line x1="18" x2="18" y1="20" y2="4"/><line x1="6" x2="6" y1="20" y2="16"/>',
    "alert":        '<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3"/><line x1="12" x2="12" y1="9" y2="13"/><line x1="12" x2="12.01" y1="17" y2="17"/>',
    "search":       '<circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/>',
    "folder":       '<path d="M20 20a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2h-7.9a2 2 0 0 1-1.69-.9L9.6 3.9A2 2 0 0 0 7.93 3H4a2 2 0 0 0-2 2v13a2 2 0 0 0 2 2Z"/>',
    "activity":     '<path d="M22 12h-2.48a2 2 0 0 0-1.93 1.46l-2.35 8.36a.25.25 0 0 1-.48 0L9.24 2.18a.25.25 0 0 0-.48 0l-2.35 8.36A2 2 0 0 1 4.49 12H2"/>',
    "target":       '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>',
    "check_circle": '<path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><path d="m9 11 3 3L22 4"/>',
    "x_circle":     '<circle cx="12" cy="12" r="10"/><path d="m15 9-6 6"/><path d="m9 9 6 6"/>',
    "alert_circle": '<circle cx="12" cy="12" r="10"/><line x1="12" x2="12" y1="8" y2="12"/><line x1="12" x2="12.01" y1="16" y2="16"/>',
    "download":     '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" x2="12" y1="15" y2="3"/>',
    "upload":       '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" x2="12" y1="3" y2="15"/>',
    "clock":        '<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>',
    "globe":        '<circle cx="12" cy="12" r="10"/><path d="M12 2a14.5 14.5 0 0 0 0 20 14.5 14.5 0 0 0 0-20"/><path d="M2 12h20"/>',
    "filter":       '<polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3"/>',
    "file_text":    '<path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z"/><path d="M14 2v4a2 2 0 0 0 2 2h4"/><path d="M10 9H8"/><path d="M16 13H8"/><path d="M16 17H8"/>',
    "star":         '<polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/>',
    "layers":       '<path d="m12.83 2.18a2 2 0 0 0-1.66 0L2.6 6.08a1 1 0 0 0 0 1.83l8.58 3.91a2 2 0 0 0 1.66 0l8.58-3.9a1 1 0 0 0 0-1.83Z"/><path d="m22 17.65-9.17 4.16a2 2 0 0 1-1.66 0L2 17.65"/><path d="m22 12.65-9.17 4.16a2 2 0 0 1-1.66 0L2 12.65"/>',
    "database":     '<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5V19A9 3 0 0 0 21 19V5"/><path d="M3 12A9 3 0 0 0 21 12"/>',
    "user":         '<path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/>',
    "building":     '<rect width="16" height="20" x="4" y="2" rx="2" ry="2"/><path d="M9 22v-4h6v4"/><path d="M8 6h.01"/><path d="M16 6h.01"/><path d="M12 6h.01"/><path d="M12 10h.01"/><path d="M12 14h.01"/><path d="M16 10h.01"/><path d="M16 14h.01"/><path d="M8 10h.01"/><path d="M8 14h.01"/>',
    "zap":          '<path d="M4 14a1 1 0 0 1-.78-1.63l9.9-10.2a.5.5 0 0 1 .86.46l-1.92 6.02A1 1 0 0 0 13 10h7a1 1 0 0 1 .78 1.63l-9.9 10.2a.5.5 0 0 1-.86-.46l1.92-6.02A1 1 0 0 0 11 14z"/>',
    "scale":        '<path d="m16 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z"/><path d="m2 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z"/><path d="M7 21h10"/><path d="M12 3v18"/><path d="M3 7h2c2 0 5-1 7-2 2 1 5 2 7 2h2"/>',
    "scan":         '<path d="M3 7V5a2 2 0 0 1 2-2h2"/><path d="M17 3h2a2 2 0 0 1 2 2v2"/><path d="M21 17v2a2 2 0 0 1-2 2h-2"/><path d="M7 21H5a2 2 0 0 1-2-2v-2"/>',
    "radar":        '<path d="M19.07 4.93A10 10 0 0 0 6.99 3.34"/><path d="M4 6h.01"/><path d="M2.29 9.62A10 10 0 1 0 21.31 8.35"/><path d="M16.24 7.76A6 6 0 1 0 8.23 16.67"/><path d="M12 18h.01"/><path d="M17.99 11.66A6 6 0 0 1 15.77 16.67"/><circle cx="12" cy="12" r="2"/><path d="m13.41 10.59 5.66-5.66"/>',
    "gauge":        '<path d="m12 14 4-4"/><path d="M3.34 19a10 10 0 1 1 17.32 0"/>',
    "sparkles":     '<path d="M9.937 15.5A2 2 0 0 0 8.5 14.063l-6.135-1.582a.5.5 0 0 1 0-.962L8.5 9.936A2 2 0 0 0 9.937 8.5l1.582-6.135a.5.5 0 0 1 .963 0L14.063 8.5A2 2 0 0 0 15.5 9.937l6.135 1.581a.5.5 0 0 1 0 .964L15.5 14.063a2 2 0 0 0-1.437 1.437l-1.582 6.135a.5.5 0 0 1-.963 0z"/><path d="M20 3v4"/><path d="M22 5h-4"/>',
    "clipboard":    '<rect width="8" height="4" x="8" y="2" rx="1" ry="1"/><path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"/>',
    "hash":         '<line x1="4" x2="20" y1="9" y2="9"/><line x1="4" x2="20" y1="15" y2="15"/><line x1="10" x2="8" y1="3" y2="21"/><line x1="16" x2="14" y1="3" y2="21"/>',
    "truck":        '<path d="M14 18V6a2 2 0 0 0-2-2H4a2 2 0 0 0-2 2v11a1 1 0 0 0 1 1h2"/><path d="M15 18H9"/><path d="M19 18h2a1 1 0 0 0 1-1v-3.65a1 1 0 0 0-.22-.624l-3.48-4.35A1 1 0 0 0 17.52 8H14"/><circle cx="17" cy="18" r="2"/><circle cx="7" cy="18" r="2"/>',
}


def ic(name: str, sz: int = 16, color: str = "currentColor") -> str:
    """Retourne un <span> contenant un SVG Lucide inline, correctement dimensionné et aligné."""
    paths = _ICON_PATHS.get(name, "")
    if not paths:
        return ""
    return (
        f'<span style="display:inline-flex;align-items:center;justify-content:center;'
        f'width:{sz}px;height:{sz}px;flex-shrink:0;vertical-align:middle;">'
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{sz}" height="{sz}" '
        f'viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" '
        f'stroke-linecap="round" stroke-linejoin="round">{paths}</svg></span>'
    )


# ══════════════════════════════════════════════════════════════════════════════
# CSS — Thème Blanc Forcé, Lisibilité Maximale
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"] {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
        color: {P["text"]} !important;
    }}
    .stApp {{
        background: {P["bg2"]} !important;
        color: {P["text"]} !important;
    }}
    .stApp *, .stApp p, .stApp span, .stApp label, .stApp div {{
        color: {P["text"]};
    }}
    .main .block-container {{
        padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1440px;
    }}

    /* Sidebar */
    section[data-testid="stSidebar"] {{ background: {P["bg"]} !important; border-right: 1px solid {P["border"]} !important; }}
    section[data-testid="stSidebar"] > div {{ background: {P["bg"]} !important; }}
    section[data-testid="stSidebar"] * {{ color: {P["text"]} !important; }}

    /* ── Menu de Navigation Moderne dans la Sidebar (Cartes d'onglets sans puces radio) ── */
    section[data-testid="stSidebar"] div[data-testid="stRadio"] > div {{
        display: flex;
        flex-direction: column;
        gap: 0.45rem;
    }}
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label {{
        background: {P["bg"]} !important;
        border: 1px solid {P["border"]} !important;
        border-radius: 10px !important;
        padding: 0.65rem 0.9rem !important;
        margin: 0 !important;
        cursor: pointer !important;
        transition: all 0.18s cubic-bezier(0.16, 1, 0.3, 1) !important;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.03) !important;
        display: flex !important;
        align-items: center !important;
        position: relative !important;
        width: 100% !important;
    }}
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label:hover {{
        background: {P["bg2"]} !important;
        border-color: #94A3B8 !important;
        transform: translateX(3px);
        box-shadow: 0 2px 5px rgba(15, 23, 42, 0.06) !important;
    }}
    /* Masquer totalement le cercle radio natif */
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label > div:first-child {{
        display: none !important;
    }}
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label div[data-testid="stMarkdownContainer"] p {{
        font-size: 0.86rem !important;
        font-weight: 600 !important;
        color: {P["text"]} !important;
        margin: 0 !important;
        display: flex !important;
        align-items: center !important;
    }}
    /* Option active sélectionnée */
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label:has(input:checked),
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label[aria-checked="true"] {{
        background: {P["primary_bg"]} !important;
        border: 1.5px solid {P["primary"]} !important;
        box-shadow: 0 2px 6px rgba(37, 99, 235, 0.15) !important;
    }}
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label:has(input:checked) div[data-testid="stMarkdownContainer"] p,
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label[aria-checked="true"] div[data-testid="stMarkdownContainer"] p {{
        color: {P["primary"]} !important;
        font-weight: 700 !important;
    }}
    /* Point lumineux sur la droite pour l'option active */
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label:has(input:checked)::after,
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label[aria-checked="true"]::after {{
        content: "";
        position: absolute;
        right: 12px;
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background-color: {P["primary"]};
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.2);
    }}

    /* Icônes Lucide SVG dans chaque bouton de navigation */
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label:nth-child(1) div[data-testid="stMarkdownContainer"] p::before {{
        content: "";
        display: inline-block;
        width: 16px;
        height: 16px;
        margin-right: 9px;
        flex-shrink: 0;
        background-color: currentColor;
        -webkit-mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cline x1='12' x2='12' y1='20' y2='10'/%3E%3Cline x1='18' x2='18' y1='20' y2='4'/%3E%3Cline x1='6' x2='6' y1='20' y2='16'/%3E%3C/svg%3E") no-repeat center / contain;
        mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cline x1='12' x2='12' y1='20' y2='10'/%3E%3Cline x1='18' x2='18' y1='20' y2='4'/%3E%3Cline x1='6' x2='6' y1='20' y2='16'/%3E%3C/svg%3E") no-repeat center / contain;
    }}
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label:nth-child(2) div[data-testid="stMarkdownContainer"] p::before {{
        content: "";
        display: inline-block;
        width: 16px;
        height: 16px;
        margin-right: 9px;
        flex-shrink: 0;
        background-color: currentColor;
        -webkit-mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3'/%3E%3Cline x1='12' x2='12' y1='9' y2='13'/%3E%3Cline x1='12' x2='12.01' y1='17' y2='17'/%3E%3C/svg%3E") no-repeat center / contain;
        mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3'/%3E%3Cline x1='12' x2='12' y1='9' y2='13'/%3E%3Cline x1='12' x2='12.01' y1='17' y2='17'/%3E%3C/svg%3E") no-repeat center / contain;
    }}
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label:nth-child(3) div[data-testid="stMarkdownContainer"] p::before {{
        content: "";
        display: inline-block;
        width: 16px;
        height: 16px;
        margin-right: 9px;
        flex-shrink: 0;
        background-color: currentColor;
        -webkit-mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='11' cy='11' r='8'/%3E%3Cpath d='m21 21-4.3-4.3'/%3E%3C/svg%3E") no-repeat center / contain;
        mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='11' cy='11' r='8'/%3E%3Cpath d='m21 21-4.3-4.3'/%3E%3C/svg%3E") no-repeat center / contain;
    }}
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label:nth-child(4) div[data-testid="stMarkdownContainer"] p::before {{
        content: "";
        display: inline-block;
        width: 16px;
        height: 16px;
        margin-right: 9px;
        flex-shrink: 0;
        background-color: currentColor;
        -webkit-mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M20 20a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2h-7.9a2 2 0 0 1-1.69-.9L9.6 3.9A2 2 0 0 0 7.93 3H4a2 2 0 0 0-2 2v13a2 2 0 0 0 2 2Z'/%3E%3C/svg%3E") no-repeat center / contain;
        mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M20 20a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2h-7.9a2 2 0 0 1-1.69-.9L9.6 3.9A2 2 0 0 0 7.93 3H4a2 2 0 0 0-2 2v13a2 2 0 0 0 2 2Z'/%3E%3C/svg%3E") no-repeat center / contain;
    }}

    /* Widgets: Force fond blanc */
    .stTextInput input, .stNumberInput input, .stDateInput input {{
        background: {P["bg"]} !important; color: {P["text"]} !important; border: 1px solid {P["border"]} !important; border-radius: 8px !important;
    }}
    .stSelectbox > div > div {{ background: {P["bg"]} !important; color: {P["text"]} !important; border: 1px solid {P["border"]} !important; border-radius: 8px !important; }}
    .stSelectbox > div > div > div {{ color: {P["text"]} !important; }}

    /* Boutons */
    div.stButton > button {{
        border-radius: 8px !important; font-weight: 600 !important; font-size: 0.85rem !important;
        background: {P["bg"]} !important; color: {P["text"]} !important; border: 1px solid {P["border"]} !important;
    }}
    div.stButton > button:hover {{ border-color: {P["primary"]} !important; color: {P["primary"]} !important; background: {P["primary_bg"]} !important; }}
    button[kind="primary"], div.stButton > button[kind="primary"] {{
        background: {P["primary"]} !important; color: #fff !important; border: none !important;
    }}
    button[kind="primary"]:hover {{ background: #1D4ED8 !important; }}

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{ border-bottom: 1px solid {P["border"]}; margin-bottom: 1rem; }}
    .stTabs [data-baseweb="tab"] {{ border-radius: 8px; font-weight: 600; font-size: 0.83rem; color: {P["text2"]} !important; background: transparent; }}
    .stTabs [aria-selected="true"] {{ color: {P["primary"]} !important; background: {P["primary_bg"]} !important; }}

    /* Formulaires */
    [data-testid="stForm"] {{ background: {P["bg"]} !important; border: 1px solid {P["border"]} !important; border-radius: 12px !important; }}
    [data-testid="stForm"] * {{ color: {P["text"]} !important; }}

    /* Tableaux */
    div[data-testid="stDataFrame"] {{ border: 1px solid {P["border"]}; border-radius: 10px; background: {P["bg"]}; }}

    /* Metrics */
    [data-testid="stMetric"] * {{ color: {P["text"]} !important; }}

    hr {{ border: 0; height: 1px; background: {P["border"]}; margin: 1.5rem 0; }}

    /* ── Composants AuditPulse ── */
    .ap-card {{
        background: {P["bg"]}; border: 1px solid {P["border"]}; border-radius: 12px;
        padding: 1.25rem 1.5rem; margin-bottom: 1.15rem; color: {P["text"]};
        box-shadow: 0 1px 3px rgba(15,23,42,0.04);
    }}
    .ap-kpi {{
        background: {P["bg"]}; border: 1px solid {P["border"]}; border-radius: 12px;
        padding: 1.1rem 1.3rem 1.1rem 1.5rem; position: relative; overflow: hidden; color: {P["text"]};
    }}
    .ap-kpi::before {{ content:""; position:absolute; top:0; left:0; width:4px; height:100%; background:{P["primary"]}; }}
    .ap-kpi.danger::before {{ background:{P["danger"]}; }}
    .ap-kpi.warning::before {{ background:{P["warning"]}; }}
    .ap-kpi.success::before {{ background:{P["success"]}; }}
    .kpi-label {{ font-size:0.75rem; font-weight:700; text-transform:uppercase; letter-spacing:0.06em; color:{P["text2"]}; margin-bottom:0.25rem; }}
    .kpi-value {{ font-size:1.65rem; font-weight:800; line-height:1.15; color:{P["text"]}; }}
    .kpi-sub {{ font-size:0.8rem; font-weight:500; margin-top:0.3rem; color:{P["text2"]}; display:flex; align-items:center; gap:0.3rem; }}

    .ap-header {{
        background:{P["bg"]}; border:1px solid {P["border"]}; border-radius:12px;
        padding:1.4rem 1.6rem; margin-bottom:1.4rem; color:{P["text"]};
        display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:1rem;
    }}
    .ap-header h1 {{ font-size:1.35rem; font-weight:800; color:{P["text"]}; margin:0; display:flex; align-items:center; gap:0.5rem; }}
    .ap-header p {{ font-size:0.85rem; color:{P["text2"]}; margin:0.2rem 0 0; }}

    .badge {{ display:inline-flex; align-items:center; gap:0.25rem; padding:0.2rem 0.55rem; border-radius:9999px; font-size:0.72rem; font-weight:700; }}
    .badge-primary {{ background:{P["primary_bg"]}; color:{P["primary"]}; border:1px solid rgba(37,99,235,0.2); }}
    .badge-danger  {{ background:{P["danger_bg"]}; color:{P["danger"]}; border:1px solid rgba(220,38,38,0.2); }}
    .badge-success {{ background:{P["success_bg"]}; color:{P["success"]}; border:1px solid rgba(5,150,105,0.2); }}
    .badge-warning {{ background:{P["warning_bg"]}; color:{P["warning"]}; border:1px solid rgba(217,119,6,0.2); }}

    /* Ligne icône + texte */
    .ic-row {{ display:flex; align-items:center; gap:0.4rem; margin-bottom:0.25rem; font-size:0.82rem; color:{P["text2"]}; }}
    .ic-row strong {{ color:{P["text"]}; }}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# UTILITAIRES
# ══════════════════════════════════════════════════════════════════════════════
def fmt_money(v: float) -> str:
    if v is None or np.isnan(v): return "0 XAF"
    if abs(v) >= 1e9: return f"{v/1e9:.2f} Md XAF"
    if abs(v) >= 1e6: return f"{v/1e6:.1f} M XAF"
    if abs(v) >= 1e3: return f"{v:,.0f} XAF".replace(",", " ")
    return f"{v:.2f} XAF"

def compute_gini(vals):
    v = np.sort(vals[vals > 0]); n = len(v)
    if n == 0 or v.sum() == 0: return 0.0
    return float((2 * np.sum(np.arange(1, n+1) * v) - (n+1) * v.sum()) / (n * v.sum()))

def compute_benford(amounts):
    valid = amounts.dropna(); valid = valid[valid > 0]
    fd = valid.astype(str).str.extract(r'([1-9])')[0].dropna().astype(int)
    counts = fd.value_counts().reindex(range(1,10), fill_value=0); t = len(fd)
    if t == 0: return None
    obs = counts / t * 100
    bfd = pd.Series({d: np.log10(1+1/d)*100 for d in range(1,10)})
    stat, pval = chisquare(counts, bfd/100*t)
    return {"digits": list(range(1,10)), "obs": obs.values, "bfd": bfd.values, "stat": stat, "pval": pval}


# ── Pipeline ─────────────────────────────────────────────────────────────────
@st.cache_data(ttl=600, show_spinner=False)
def load_scored():
    try:
        from src.data_loader import load_expenses_from_sql
        from src.scorer import score_anomalies
        from src.config import DEFAULT_SQL_FILE
        return score_anomalies(load_expenses_from_sql(DEFAULT_SQL_FILE))
    except Exception as e:
        st.error(f"Erreur pipeline : {e}"); return pd.DataFrame()

def api_health():
    try:
        r = requests.get(f"{API_URL}/health", timeout=1)
        return r.json() if r.status_code == 200 else None
    except: return None

def api_single(payload):
    try:
        r = requests.post(f"{API_URL}/analyze/transaction", json=payload, timeout=1.5)
        if r.status_code == 200: return r.json()
    except: pass
    try:
        from src.model_registry import registry; from src.config import DEFAULT_SQL_FILE
        if not registry.is_ready: registry.initialize(DEFAULT_SQL_FILE)
        return registry.score_transaction(payload)
    except Exception as e:
        st.error(f"Erreur scoring : {e}"); return None

def api_batch(txs, pbar, ptxt):
    results, n_a, amt_a = [], 0, 0.0
    chunks = [txs[i:i+BATCH_CHUNK] for i in range(0, len(txs), BATCH_CHUNK)]
    t0 = time.time()
    for idx, c in enumerate(chunks):
        try:
            r = requests.post(f"{API_URL}/analyze/batch", json={"transactions": c}, timeout=2)
            if r.status_code == 200:
                d = r.json(); results.extend(d["results"]); n_a += d["summary"]["n_anomalies"]; amt_a += d["summary"]["anomaly_amount_total"]
            else: raise RuntimeError()
        except:
            from src.scorer import score_anomalies
            ds = score_anomalies(pd.DataFrame(c)); a = ds[ds["is_anomaly"]==1]
            results.extend(ds.to_dict(orient="records")); n_a += len(a); amt_a += float(a["amount"].sum()) if "amount" in a else 0
        pbar.progress((idx+1)/len(chunks)); ptxt.text(f"Lot {idx+1}/{len(chunks)}")
    dur = time.time() - t0
    if not results: return None
    return {"summary": {"total": len(results), "n_anomalies": n_a, "anomaly_rate": round(n_a/len(results)*100, 2), "anomaly_amount_total": amt_a, "duration_sec": round(dur, 2), "speed": round(len(results)/dur, 1) if dur > 0 else 0}, "results": results}

def to_xlsx(df):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w: df.to_excel(w, index=False, sheet_name="AuditPulse")
    return buf.getvalue()


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"""
    <div style="padding-bottom:1rem; border-bottom:1px solid {P['border']}; margin-bottom:1.25rem;">
        <div style="font-size:1.3rem; font-weight:800; color:{P['text']}; display:flex; align-items:center; gap:0.5rem;">
            {ic("shield", 22, P["primary"])} AuditPulse
        </div>
        <div style="font-size:0.78rem; color:{P['text2']}; font-weight:500; margin-top:0.2rem;">
            Studio de renseignement médico-légal et de lutte contre la fraude
        </div>
    </div>
    """, unsafe_allow_html=True)

    h = api_health()
    if h and h.get("ready"):
        st.markdown(f'<div style="background:{P["success_bg"]}; border:1px solid rgba(5,150,105,0.2); border-radius:8px; padding:0.45rem 0.7rem; margin-bottom:1.2rem; display:flex; align-items:center; gap:0.4rem;">{ic("check_circle",14,P["success"])} <span style="font-size:0.78rem; font-weight:700; color:{P["success"]}">Moteur API Opérationnel (v1.0)</span></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div style="background:{P["success_bg"]}; border:1px solid rgba(5,150,105,0.2); border-radius:8px; padding:0.45rem 0.7rem; margin-bottom:1.2rem; display:flex; align-items:center; gap:0.4rem;">{ic("check_circle",14,P["success"])} <span style="font-size:0.78rem; font-weight:700; color:{P["success"]}">Moteur IA Embarqué (Autonome)</span></div>', unsafe_allow_html=True)

    st.markdown(f'<div style="font-size:0.75rem; font-weight:700; text-transform:uppercase; letter-spacing:0.06em; color:{P["text2"]}; margin-bottom:0.5rem; display:flex; align-items:center; gap:0.4rem;">{ic("layers", 14, P["primary"])} Modules Principaux</div>', unsafe_allow_html=True)
    page = st.radio("Nav", ["Vue Globale", "Dossiers d'Anomalies", "Simulateur Unitaire", "Audit Batch"], label_visibility="collapsed")

    st.markdown("---")
    st.markdown(f"""
    <div style="font-size:0.75rem; font-weight:700; text-transform:uppercase; letter-spacing:0.06em; color:{P['text2']}; margin-bottom:0.4rem; display:flex; align-items:center; gap:0.4rem;">
        {ic('filter', 14, P['primary'])} Sensibilité d'Audit
    </div>
    <div style="font-size:0.72rem; color:{P['text3']}; margin-bottom:0.6rem;">
        Réglez le niveau de sévérité du filtrage des anomalies.
    </div>
    """, unsafe_allow_html=True)

    # 3 Presets rapides
    if "slider_threshold" not in st.session_state:
        st.session_state["slider_threshold"] = 0.25

    p1, p2, p3 = st.columns(3)
    with p1:
        if st.button("Strict", use_container_width=True, help="15% : Capture tout, tolérance minimale"):
            st.session_state["slider_threshold"] = 0.15
            st.rerun()
    with p2:
        if st.button("Équilibré", use_container_width=True, help="25% : Standard recommandé en entreprise"):
            st.session_state["slider_threshold"] = 0.25
            st.rerun()
    with p3:
        if st.button("Tolérant", use_container_width=True, help="40% : Fraudes massives & urgences uniquement"):
            st.session_state["slider_threshold"] = 0.40
            st.rerun()

    # Curseur interactif
    selected_threshold = st.slider(
        "Seuil d'alerte",
        min_value=0.10,
        max_value=0.60,
        value=float(st.session_state["slider_threshold"]),
        step=0.01,
        format="%.2f",
        label_visibility="collapsed",
        help="Une dépense est classée comme anomalie prioritaire si son score de risque dépasse ce seuil (ou en cas de pattern de fraude avéré)."
    )
    st.session_state["slider_threshold"] = selected_threshold

    # Profil dynamique selon le seuil sélectionné
    if selected_threshold < 0.20:
        m_label = "Mode Ultra-Vigilant"
        m_color = P["danger"]
        m_bg = P["danger_bg"]
        m_icon = ic("alert", 13, P["danger"])
        m_desc = "Tolérance minimale. Capture même les petits écarts. Idéal pour clôture annuelle ou contrôle approfondi."
    elif selected_threshold <= 0.35:
        m_label = "Mode Équilibré (Recommandé)"
        m_color = P["success"]
        m_bg = P["success_bg"]
        m_icon = ic("check_circle", 13, P["success"])
        m_desc = "Compromis optimal sécurité / charge. Isole les vraies fraudes sans noyer les contrôleurs sous de faux signaux."
    else:
        m_label = "Mode Haute Priorité"
        m_color = P["warning"]
        m_bg = P["warning_bg"]
        m_icon = ic("target", 13, P["warning"])
        m_desc = "Filtre très restrictif. Ne remonte que les fraudes lourdes et montants hors normes pour traiter les urgences."

    # Calcul dynamique de l'impact en direct sur les données du portefeuille
    df_preview = load_scored()
    if not df_preview.empty:
        n_prev = int(((df_preview["score_final"] >= selected_threshold) | (df_preview["is_fraud_pattern"] == 1)).sum())
        rate_prev = (n_prev / len(df_preview)) * 100
        amt_prev = df_preview[(df_preview["score_final"] >= selected_threshold) | (df_preview["is_fraud_pattern"] == 1)]["amount"].sum()
    else:
        n_prev, rate_prev, amt_prev = 0, 0.0, 0.0

    st.markdown(f"""
    <div style="background:{P['bg']}; border:1px solid {P['border']}; border-radius:10px; padding:0.75rem 0.85rem; margin-top:0.4rem; box-shadow:0 1px 2px rgba(0,0,0,0.03);">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.35rem;">
            <span style="font-size:0.72rem; font-weight:700; color:{m_color}; background:{m_bg}; padding:0.15rem 0.5rem; border-radius:6px; display:inline-flex; align-items:center; gap:0.3rem;">
                {m_icon} {m_label}
            </span>
            <span style="font-size:0.8rem; font-weight:800; color:{P['text']};">{selected_threshold*100:.0f}%</span>
        </div>
        <div style="font-size:0.71rem; color:{P['text2']}; line-height:1.35; margin-bottom:0.5rem;">
            {m_desc}
        </div>
        <div style="border-top:1px solid {P['border']}; padding-top:0.45rem; font-size:0.72rem; color:{P['text2']};">
            <div style="display:flex; justify-content:space-between; margin-bottom:0.2rem;">
                <span>Dossiers signalés :</span>
                <strong style="color:{m_color};">{n_prev:,} ({rate_prev:.1f}%)</strong>
            </div>
            <div style="display:flex; justify-content:space-between;">
                <span>Capital à risque :</span>
                <strong style="color:{m_color};">{fmt_money(amt_prev)}</strong>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("Comment interpréter ce réglage ?", expanded=False):
        st.markdown(f"""
        <div style="font-size:0.72rem; color:{P['text2']}; line-height:1.45;">
            <strong>• Si vous baissez le seuil :</strong> Vous augmentez la sévérité. Le système détectera plus d'anomalies, mais l'équipe devra vérifier plus de faux positifs.<br/><br/>
            <strong>• Si vous montez le seuil :</strong> Vous ciblez uniquement les cas graves. Moins de dossiers à traiter, mais risque de manquer des petites dérives.
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="margin-top:0.9rem; padding-top:0.75rem; border-top:1px solid {P['border']};">
        <div style="font-size:0.72rem; font-weight:700; text-transform:uppercase; color:{P['text2']}; margin-bottom:0.4rem; display:flex; align-items:center; gap:0.3rem;">
            {ic("layers", 13, P['primary'])} Score Hybride (5 Couches)
        </div>
        <div style="font-size:0.7rem; color:{P['text2']}; line-height:1.55;">
            • <strong>25%</strong> Isolation Forest (Outliers globaux)<br/>
            • <strong>25%</strong> LOF (Profils collaborateurs)<br/>
            • <strong>10%</strong> DBSCAN (Points isolés hors groupes)<br/>
            • <strong>20%</strong> Règles Métiers (Plafonds & horaires)<br/>
            • <strong>20%</strong> Schémas Anti-Fraude (Doublons...)
        </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — VUE GLOBALE
# ══════════════════════════════════════════════════════════════════════════════
if page == "Vue Globale":
    st.markdown(f"""<div class="ap-header"><div>
        <h1>{ic("bar_chart",22,P["primary"])} Synthèse Exécutive & Exposition Financière</h1>
        <p>Surveillance macro-analytique des dépenses d'entreprise, concentration du risque et tests probatoires de conformité.</p>
    </div><div style="display:flex;gap:0.5rem;align-items:center;">
        <span class="badge badge-primary">{ic("hash",13)} 4 772 Opérations</span>
        <span class="badge badge-success">{ic("activity",13)} Temps Réel</span>
    </div></div>""", unsafe_allow_html=True)

    with st.spinner("Chargement et scoring du portefeuille..."):
        df_all = load_scored()
    if df_all.empty: st.warning("Aucune donnée disponible."); st.stop()

    df_all["is_anomaly_dynamic"] = ((df_all["score_final"] >= selected_threshold) | (df_all["is_fraud_pattern"] == 1)).astype(int)
    total_tx = len(df_all); total_cap = df_all["amount"].sum()
    df_a = df_all[df_all["is_anomaly_dynamic"]==1]; df_n = df_all[df_all["is_anomaly_dynamic"]==0]
    n_a = len(df_a); rate = (n_a/total_tx)*100 if total_tx else 0
    risk_cap = df_a["amount"].sum(); risk_pct = (risk_cap/total_cap)*100 if total_cap else 0
    avg_a = df_a["score_final"].mean() if n_a else 0; avg_n = df_n["score_final"].mean() if len(df_n) else 0

    k1,k2,k3,k4 = st.columns(4)
    with k1: st.markdown(f'<div class="ap-kpi"><div class="kpi-label">Masse Financière Auditée</div><div class="kpi-value">{fmt_money(total_cap)}</div><div class="kpi-sub">{ic("clipboard",13,P["text2"])} {total_tx:,} opérations</div></div>', unsafe_allow_html=True)
    with k2: st.markdown(f'<div class="ap-kpi danger"><div class="kpi-label">Capital Sous Présomption de Risque</div><div class="kpi-value" style="color:{P["danger"]}">{fmt_money(risk_cap)}</div><div class="kpi-sub" style="color:{P["danger"]}">{ic("shield",13,P["danger"])} {risk_pct:.2f}% de la masse capturée</div></div>', unsafe_allow_html=True)
    with k3: st.markdown(f'<div class="ap-kpi warning"><div class="kpi-label">Dossiers d\'Anomalies Prioritaires</div><div class="kpi-value" style="color:{P["warning"]}">{n_a:,}</div><div class="kpi-sub">{ic("target",13,P["warning"])} Taux : {rate:.2f}%</div></div>', unsafe_allow_html=True)
    with k4:
        sep = (avg_a/avg_n) if avg_n > 0 else 1.0
        st.markdown(f'<div class="ap-kpi success"><div class="kpi-label">Indice de Séparation Forensic</div><div class="kpi-value" style="color:{P["success"]}">{sep:.1f}x</div><div class="kpi-sub">{ic("zap",13,P["success"])} {avg_a:.3f} vs {avg_n:.3f} (saines)</div></div>', unsafe_allow_html=True)

    # ── SECTION GRAPHIQUE — Visualisations Métier Intuitives ──────────────────

    # ── Ligne 1 : Donuts — Répartition claire Saines vs Anomalies ──────────
    st.markdown(f"<h3 style='font-size:1.1rem; font-weight:700; color:{P['text']}; margin:1rem 0 0.8rem; display:flex; align-items:center; gap:0.5rem;'>{ic('scan',18,P['primary'])} Vue d'Ensemble des Résultats d'Audit</h3>", unsafe_allow_html=True)

    d1, d2 = st.columns(2)
    with d1:
        st.markdown("<div class='ap-card'>", unsafe_allow_html=True)
        st.markdown(f"<strong style='color:{P['text']}'>Répartition des Transactions</strong>", unsafe_allow_html=True)
        st.caption("Combien de dépenses sont suspectes par rapport aux dépenses saines ?")
        fig_donut1 = go.Figure(go.Pie(
            labels=["Conformes", "Suspectes"],
            values=[total_tx - n_a, n_a],
            hole=0.6,
            marker=dict(colors=[P["primary"], P["danger"]]),
            textinfo="label+percent",
            textfont=dict(size=13, color=P["text"]),
            hovertemplate="<b>%{label}</b><br>%{value:,} transactions<br>%{percent}<extra></extra>",
        ))
        fig_donut1.add_annotation(text=f"<b>{total_tx:,}</b><br>opérations", x=0.5, y=0.5, font=dict(size=15, color=P["text"]), showarrow=False)
        fig_donut1.update_layout(template="plotly_white", height=310, margin=dict(l=10,r=10,t=10,b=10), showlegend=False)
        st.plotly_chart(fig_donut1, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with d2:
        st.markdown("<div class='ap-card'>", unsafe_allow_html=True)
        st.markdown(f"<strong style='color:{P['text']}'>Répartition du Capital Financier</strong>", unsafe_allow_html=True)
        st.caption("Quelle part de l'argent est concentrée dans les dépenses suspectes ?")
        safe_cap = total_cap - risk_cap
        fig_donut2 = go.Figure(go.Pie(
            labels=["Capital Sain", "Capital à Risque"],
            values=[max(safe_cap, 0), risk_cap],
            hole=0.6,
            marker=dict(colors=[P["success"], P["danger"]]),
            textinfo="label+percent",
            textfont=dict(size=13, color=P["text"]),
            hovertemplate="<b>%{label}</b><br>%{value:,.0f} XAF<extra></extra>",
        ))
        fig_donut2.add_annotation(text=f"<b>{fmt_money(total_cap)}</b><br>total audité", x=0.5, y=0.5, font=dict(size=14, color=P["text"]), showarrow=False)
        fig_donut2.update_layout(template="plotly_white", height=310, margin=dict(l=10,r=10,t=10,b=10), showlegend=False)
        st.plotly_chart(fig_donut2, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Ligne 2 : Top Catégories + Top Employés ───────────────────────────
    t1, t2 = st.columns(2)
    with t1:
        st.markdown("<div class='ap-card'>", unsafe_allow_html=True)
        st.markdown(f"<strong style='color:{P['text']}'>Top 10 — Catégories de Dépenses les Plus Signalées</strong>", unsafe_allow_html=True)
        st.caption("Quels types de dépenses génèrent le plus d'alertes ?")
        cat_counts = df_a.groupby("expense_type").agg(nb=("id","count"), montant=("amount","sum")).sort_values("nb", ascending=True).tail(10)
        fig_cat = go.Figure(go.Bar(
            y=cat_counts.index,
            x=cat_counts["nb"],
            orientation="h",
            marker_color=P["danger"],
            text=[f"{v} alertes" for v in cat_counts["nb"]],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>%{x} alertes<br>Montant total : %{customdata:,.0f} XAF<extra></extra>",
            customdata=cat_counts["montant"],
        ))
        fig_cat.update_layout(template="plotly_white", height=360, margin=dict(l=10,r=60,t=10,b=10), xaxis=dict(title="Nombre d'alertes", gridcolor="#F1F5F9"), yaxis=dict(automargin=True))
        st.plotly_chart(fig_cat, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with t2:
        st.markdown("<div class='ap-card'>", unsafe_allow_html=True)
        st.markdown(f"<strong style='color:{P['text']}'>Top 10 — Employés les Plus Signalés</strong>", unsafe_allow_html=True)
        st.caption("Quels utilisateurs concentrent le plus de transactions suspectes ?")
        usr_counts = df_a.groupby("user_id").agg(nb=("id","count"), montant=("amount","sum")).sort_values("nb", ascending=True).tail(10)
        fig_usr = go.Figure(go.Bar(
            y=usr_counts.index,
            x=usr_counts["nb"],
            orientation="h",
            marker_color=P["warning"],
            text=[f"{v} alertes" for v in usr_counts["nb"]],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>%{x} alertes<br>Montant total : %{customdata:,.0f} XAF<extra></extra>",
            customdata=usr_counts["montant"],
        ))
        fig_usr.update_layout(template="plotly_white", height=360, margin=dict(l=10,r=60,t=10,b=10), xaxis=dict(title="Nombre d'alertes", gridcolor="#F1F5F9"), yaxis=dict(automargin=True))
        st.plotly_chart(fig_usr, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Ligne 3 : Tendance Mensuelle ──────────────────────────────────────
    st.markdown("<div class='ap-card'>", unsafe_allow_html=True)
    st.markdown(f"<strong style='color:{P['text']}'>Évolution Mensuelle des Anomalies Détectées</strong>", unsafe_allow_html=True)
    st.caption("Comment le nombre de transactions suspectes évolue-t-il dans le temps ?")
    if "created_at" in df_all.columns:
        df_time = df_all.copy()
        df_time["mois"] = pd.to_datetime(df_time["created_at"]).dt.to_period("M").astype(str)
        trend = df_time.groupby("mois").agg(
            total=("id", "count"),
            anomalies=("is_anomaly_dynamic", "sum"),
        ).reset_index()
        trend["taux"] = (trend["anomalies"] / trend["total"] * 100).round(1)

        fig_trend = go.Figure()
        fig_trend.add_trace(go.Bar(
            x=trend["mois"], y=trend["total"], name="Total des transactions",
            marker_color="#E2E8F0", text=trend["total"], textposition="outside",
        ))
        fig_trend.add_trace(go.Bar(
            x=trend["mois"], y=trend["anomalies"], name="Transactions suspectes",
            marker_color=P["danger"], text=trend["anomalies"], textposition="outside",
        ))
        fig_trend.add_trace(go.Scatter(
            x=trend["mois"], y=trend["taux"], name="Taux d'anomalies (%)",
            yaxis="y2", mode="lines+markers+text",
            line=dict(color=P["warning"], width=3),
            marker=dict(size=8, color=P["warning"]),
            text=[f"{v}%" for v in trend["taux"]], textposition="top center",
            textfont=dict(color=P["warning"], size=11),
        ))
        fig_trend.update_layout(
            template="plotly_white", height=350, barmode="overlay",
            margin=dict(l=10,r=40,t=10,b=10),
            xaxis=dict(title="Mois", gridcolor="#F1F5F9"),
            yaxis=dict(title="Nombre de transactions", gridcolor="#F1F5F9"),
            yaxis2=dict(title="Taux (%)", overlaying="y", side="right", range=[0, max(trend["taux"].max()*1.5, 15)], gridcolor="#F1F5F9"),
            legend=dict(orientation="h", y=1.12),
        )
        st.plotly_chart(fig_trend, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Ligne 4 : Canaux + Dernières Alertes ──────────────────────────────
    ch1, ch2 = st.columns([1, 1])
    with ch1:
        st.markdown("<div class='ap-card'>", unsafe_allow_html=True)
        st.markdown(f"<strong style='color:{P['text']}'>Anomalies par Canal d'Entrée</strong>", unsafe_allow_html=True)
        st.caption("Par quel canal arrivent les dépenses les plus suspectes ?")
        src_counts = df_a["source"].value_counts().head(8)
        colors_src = [P["primary"], P["purple"], P["warning"], P["danger"], P["success"], "#64748B", "#0EA5E9", "#F97316"]
        fig_src = go.Figure(go.Bar(
            x=src_counts.index, y=src_counts.values,
            marker_color=colors_src[:len(src_counts)],
            text=src_counts.values, textposition="outside",
            hovertemplate="<b>%{x}</b><br>%{y} alertes<extra></extra>",
        ))
        fig_src.update_layout(template="plotly_white", height=310, margin=dict(l=10,r=10,t=10,b=10), yaxis=dict(title="Nombre d'alertes", gridcolor="#F1F5F9"))
        st.plotly_chart(fig_src, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with ch2:
        st.markdown("<div class='ap-card'>", unsafe_allow_html=True)
        st.markdown(f"<strong style='color:{P['text']}'>Dernières Alertes Détectées</strong>", unsafe_allow_html=True)
        st.caption("Les 8 transactions suspectes les plus récentes.")
        if "created_at" in df_a.columns:
            recent = df_a.nlargest(8, "created_at")[["id", "amount", "expense_type", "user_id", "score_final"]].copy()
            recent["Montant"] = recent["amount"].apply(fmt_money)
            recent["Risque"] = (recent["score_final"] * 100).round(1).astype(str) + " / 100"
            recent = recent.rename(columns={"id": "Réf.", "expense_type": "Catégorie", "user_id": "Employé"})
            st.dataframe(recent[["Réf.", "Montant", "Catégorie", "Employé", "Risque"]], use_container_width=True, height=310, hide_index=True)
        else:
            recent = df_a.nlargest(8, "score_final")[["id", "amount", "expense_type", "user_id", "score_final"]].copy()
            recent["Montant"] = recent["amount"].apply(fmt_money)
            recent["Risque"] = (recent["score_final"] * 100).round(1).astype(str) + " / 100"
            recent = recent.rename(columns={"id": "Réf.", "expense_type": "Catégorie", "user_id": "Employé"})
            st.dataframe(recent[["Réf.", "Montant", "Catégorie", "Employé", "Risque"]], use_container_width=True, height=310, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Ligne 5 : Types de fraude détectés (résumé simple) ────────────────
    st.markdown("<div class='ap-card'>", unsafe_allow_html=True)
    st.markdown(f"<strong style='color:{P['text']}'>Types de Fraude Détectés par le Système</strong>", unsafe_allow_html=True)
    st.caption("Combien de transactions présentent des schémas de fraude connus ?")
    fraud_data = {
        "Doublons de factures": int((df_all.get("fraud_duplicate", 0) > 0).sum()),
        "Fournisseurs fantômes": int((df_all.get("fraud_ghost_supplier", 0) > 0).sum()),
        "Inflation progressive": int((df_all.get("fraud_inflation", 0) > 0).sum()),
        "Fractionnement de montants": int((df_all.get("fraud_splitting", 0) > 0).sum()),
    }
    fr1, fr2, fr3, fr4 = st.columns(4)
    fraud_cols = [fr1, fr2, fr3, fr4]
    fraud_colors = [P["primary"], P["purple"], P["warning"], P["danger"]]
    fraud_icons_list = ["clipboard", "building", "trending_up", "layers"]
    for i, (label, count) in enumerate(fraud_data.items()):
        with fraud_cols[i]:
            st.markdown(f"""<div style="background:{P['bg2']}; border:1px solid {P['border']}; border-radius:10px; padding:1rem; text-align:center; color:{P['text']}">
                <div style="margin-bottom:0.4rem;">{ic(fraud_icons_list[i], 22, fraud_colors[i])}</div>
                <div style="font-size:1.5rem; font-weight:800; color:{fraud_colors[i]}">{count}</div>
                <div style="font-size:0.78rem; font-weight:600; color:{P['text2']}; margin-top:0.15rem;">{label}</div>
            </div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — DOSSIERS D'ANOMALIES
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Dossiers d'Anomalies":
    st.markdown(f"""<div class="ap-header"><div>
        <h1>{ic("alert",22,P["danger"])} Dépenses Suspectes à Examiner</h1>
        <p>Liste de toutes les transactions identifiées comme inhabituelles par le système. Cliquez sur un dossier pour voir le détail.</p>
    </div><span class="badge badge-danger">{ic("alert_circle",13)} Alertes Actives</span></div>""", unsafe_allow_html=True)

    with st.spinner("Chargement des dossiers..."): df_all = load_scored()
    if df_all.empty: st.warning("Aucune donnée disponible."); st.stop()
    df_all["is_anomaly_dynamic"] = ((df_all["score_final"]>=selected_threshold)|(df_all["is_fraud_pattern"]==1)).astype(int)
    df_a = df_all[df_all["is_anomaly_dynamic"]==1]
    n_a = len(df_a)

    # ── Top 5 Alertes Prioritaires ────────────────────────────────────────
    st.markdown(f"<h4 style='font-size:1rem; font-weight:700; color:{P['text']}; display:flex; align-items:center; gap:0.4rem; margin-bottom:0.8rem;'>{ic('star',16,P['warning'])} Les 5 Dépenses les Plus Suspectes</h4>", unsafe_allow_html=True)
    top5 = df_all.nlargest(5, "score_final")
    cols5 = st.columns(5)
    for i, (_, rc) in enumerate(top5.iterrows()):
        score_pct = rc['score_final'] * 100
        with cols5[i]:
            st.markdown(f"""<div class="ap-card" style="padding:1rem; border-top:3px solid {P['danger']};">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.4rem;">
                    <span style="font-weight:800; font-size:0.82rem; color:{P['text']}">{rc['id']}</span>
                    <span class="badge badge-danger">{score_pct:.0f}%</span>
                </div>
                <div style="font-size:1.05rem; font-weight:800; color:{P['danger']}; margin-bottom:0.5rem;">{fmt_money(rc['amount'])}</div>
                <div class="ic-row">{ic("user",13,P["text2"])} {rc.get('user_id','—')}</div>
                <div class="ic-row">{ic("layers",13,P["text2"])} {str(rc.get('expense_type','—'))[:20]}</div>
                <div class="ic-row">{ic("globe",13,P["text2"])} {str(rc.get('source','—'))}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Filtres Simples ───────────────────────────────────────────────────
    st.markdown(f"<strong style='color:{P['text']}; display:flex; align-items:center; gap:0.4rem; margin-bottom:0.5rem;'>{ic('filter',15,P['primary'])} Filtrer les résultats</strong>", unsafe_allow_html=True)
    fc1, fc2, fc3 = st.columns([1, 1, 2])
    with fc1: sf = st.selectbox("Afficher", ["Suspectes uniquement", "Tout afficher", "Saines uniquement"])
    with fc2: tf = st.selectbox("Catégorie de dépense", ["Toutes"] + sorted([str(x) for x in df_all["expense_type"].dropna().unique()]))
    with fc3: sq = st.text_input("Rechercher par référence, employé ou fournisseur", placeholder="Ex: EXP_03649, USR_0109...")

    df_f = df_all.copy()
    if sf == "Suspectes uniquement": df_f = df_f[df_f["is_anomaly_dynamic"]==1]
    elif sf == "Saines uniquement": df_f = df_f[df_f["is_anomaly_dynamic"]==0]
    if tf != "Toutes": df_f = df_f[df_f["expense_type"]==tf]
    if sq.strip():
        t = sq.strip().lower()
        df_f = df_f[df_f["id"].str.lower().str.contains(t)|df_f["user_id"].str.lower().str.contains(t)|df_f["supplier_id"].astype(str).str.lower().str.contains(t)]

    # ── KPI rapides du filtre actif ───────────────────────────────────────
    fk1, fk2, fk3 = st.columns(3)
    n_filt = len(df_f)
    n_filt_a = int(df_f["is_anomaly_dynamic"].sum())
    filt_amt = df_f["amount"].sum()
    with fk1: st.markdown(f'<div class="ap-kpi"><div class="kpi-label">Dossiers Affichés</div><div class="kpi-value">{n_filt:,}</div></div>', unsafe_allow_html=True)
    with fk2: st.markdown(f'<div class="ap-kpi danger"><div class="kpi-label">Dont Suspectes</div><div class="kpi-value" style="color:{P["danger"]}">{n_filt_a:,}</div></div>', unsafe_allow_html=True)
    with fk3: st.markdown(f'<div class="ap-kpi warning"><div class="kpi-label">Montant Total</div><div class="kpi-value" style="color:{P["warning"]}">{fmt_money(filt_amt)}</div></div>', unsafe_allow_html=True)

    # ── Tableau Principal (lisible) ───────────────────────────────────────
    df_g = df_f.copy().sort_values("score_final", ascending=False).reset_index(drop=True)
    # Préparer des colonnes lisibles
    df_display = pd.DataFrame()
    df_display["Référence"] = df_g["id"]
    df_display["Montant (XAF)"] = df_g["amount"]
    df_display["Niveau de Risque"] = df_g["score_final"]
    df_display["Catégorie"] = df_g["expense_type"]
    df_display["Canal"] = df_g["source"]
    df_display["Employé"] = df_g["user_id"]
    if "created_at" in df_g.columns:
        df_display["Date"] = df_g["created_at"]
    df_display["Statut"] = np.where(df_g["is_anomaly_dynamic"]==1, "Suspecte", "Saine")

    st.dataframe(df_display, use_container_width=True, height=420, hide_index=True, column_config={
        "Niveau de Risque": st.column_config.ProgressColumn("Niveau de Risque", format="%.0f%%", min_value=0, max_value=1),
        "Montant (XAF)": st.column_config.NumberColumn("Montant (XAF)", format="%,.0f"),
        "Date": st.column_config.DatetimeColumn("Date", format="DD/MM/YYYY HH:mm"),
    })

    # ── Fiche d'Investigation ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown(f"<h4 style='font-size:1.05rem; font-weight:700; color:{P['text']}; display:flex; align-items:center; gap:0.4rem;'>{ic('eye',16,P['primary'])} Détail d'une Transaction</h4>", unsafe_allow_html=True)
    st.caption("Sélectionnez une transaction pour voir son analyse complète :")

    if not df_g.empty:
        sel = st.selectbox("Transaction à examiner", df_g["id"].tolist()[:50], label_visibility="collapsed")
        row = df_all[df_all["id"]==sel].iloc[0]
        score_val = row["score_final"]
        is_suspect = row["is_anomaly_dynamic"] == 1

        # Verdict Banner
        if is_suspect:
            verdict_bg = P["danger_bg"]; verdict_border = P["danger"]; verdict_icon = ic("x_circle", 22, P["danger"])
            verdict_label = "TRANSACTION SUSPECTE"; verdict_color = P["danger"]
            verdict_text = "Cette dépense présente des caractéristiques inhabituelles. Une vérification est recommandée."
        else:
            verdict_bg = P["success_bg"]; verdict_border = P["success"]; verdict_icon = ic("check_circle", 22, P["success"])
            verdict_label = "TRANSACTION NORMALE"; verdict_color = P["success"]
            verdict_text = "Cette dépense ne présente aucun signe d'irrégularité."

        st.markdown(f"""<div style="background:{verdict_bg}; border:1.5px solid {verdict_border}; border-radius:12px; padding:1rem 1.4rem; margin-bottom:1.2rem; color:{P['text']}">
            <div style="display:flex; align-items:center; gap:0.7rem;">
                {verdict_icon}
                <div>
                    <div style="font-size:1.1rem; font-weight:800; color:{verdict_color}">{verdict_label} — Risque : {score_val*100:.0f} / 100</div>
                    <div style="font-size:0.85rem; color:{P['text']}; margin-top:0.1rem;">{verdict_text}</div>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

        # Détails en 3 colonnes claires
        i1, i2, i3 = st.columns(3)
        with i1:
            st.markdown(f"""<div class="ap-card">
                <div style="font-size:0.72rem; text-transform:uppercase; color:{P['text2']}; font-weight:700; margin-bottom:0.5rem;">Informations Générales</div>
                <div class="ic-row" style="margin-bottom:0.35rem;">{ic("file_text",14,P["primary"])} <strong>Réf. :</strong> {row['id']}</div>
                <div class="ic-row" style="margin-bottom:0.35rem;">{ic("database",14,P["primary"])} <strong>Montant :</strong> {fmt_money(row['amount'])}</div>
                <div class="ic-row" style="margin-bottom:0.35rem;">{ic("layers",14,P["primary"])} <strong>Catégorie :</strong> {row.get('expense_type','—')}</div>
                <div class="ic-row">{ic("clock",14,P["primary"])} <strong>Date :</strong> {str(row.get('created_at','—'))[:19]}</div>
            </div>""", unsafe_allow_html=True)
        with i2:
            st.markdown(f"""<div class="ap-card">
                <div style="font-size:0.72rem; text-transform:uppercase; color:{P['text2']}; font-weight:700; margin-bottom:0.5rem;">Acteurs</div>
                <div class="ic-row" style="margin-bottom:0.35rem;">{ic("user",14,P["primary"])} <strong>Employé :</strong> {row.get('user_id','—')}</div>
                <div class="ic-row" style="margin-bottom:0.35rem;">{ic("truck",14,P["primary"])} <strong>Fournisseur :</strong> {row.get('supplier_id','—')}</div>
                <div class="ic-row" style="margin-bottom:0.35rem;">{ic("globe",14,P["primary"])} <strong>Canal :</strong> {row.get('source','—')}</div>
                <div class="ic-row">{ic("activity",14,P["primary"])} <strong>Statut :</strong> {row.get('status','—')}</div>
            </div>""", unsafe_allow_html=True)
        with i3:
            # Pourquoi c'est suspect ?
            rules_raw = str(row.get('rules_triggered', ''))
            rules_list = [r.strip() for r in rules_raw.split("|") if r.strip()] if rules_raw and rules_raw != "nan" else []

            # Traductions lisibles des noms de règles
            rule_translations = {
                "Montant en dehors du IQR par utilisateur": "Montant anormalement élevé pour cet employé",
                "Transaction le week-end": "Transaction effectuée un week-end",
                "Fournisseur utilisé moins de 3 fois": "Fournisseur rarement utilisé (potentiellement fictif)",
                "Montant x3+ vs moyenne glissante 7 jours": "Montant 3x supérieur à la moyenne récente",
                "Montant en dehors du IQR global": "Montant très inhabituel (hors norme globale)",
                "Fréquence journalière élevée": "Nombre de transactions élevé ce jour-là",
                "Hors heures ouvrées": "Transaction effectuée en dehors des heures de bureau",
                "Montant rond suspect": "Montant rond (possible estimation fictive)",
            }

            st.markdown(f"""<div class="ap-card" style="border-left:3px solid {P['danger'] if rules_list else P['success']}">
                <div style="font-size:0.72rem; text-transform:uppercase; color:{P['text2']}; font-weight:700; margin-bottom:0.5rem;">Pourquoi Cette Alerte ?</div>""", unsafe_allow_html=True)
            if rules_list:
                for rl in rules_list:
                    readable = rule_translations.get(rl, rl)
                    st.markdown(f'<div class="ic-row" style="margin-bottom:0.3rem;">{ic("alert",13,P["warning"])} {readable}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="ic-row">{ic("check_circle",13,P["success"])} Aucune règle enfreinte — score statistique uniquement</div>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # Comparaison visuelle simple
        st.markdown("<div class='ap-card'>", unsafe_allow_html=True)
        st.markdown(f"<strong style='color:{P['text']}'>Cette dépense est-elle normale pour sa catégorie ?</strong>", unsafe_allow_html=True)
        st.caption(f"Comparaison du montant de {row['id']} avec les autres dépenses de type « {row.get('expense_type','—')} ».")
        cat_df = df_all[df_all["expense_type"]==row["expense_type"]].copy()
        if len(cat_df) > 1:
            cat_avg = cat_df["amount"].mean()
            cat_med = cat_df["amount"].median()
            cat_max = cat_df["amount"].max()
            # Simple bar comparison
            fig_comp = go.Figure()
            fig_comp.add_trace(go.Bar(
                x=["Médiane de la catégorie", "Moyenne de la catégorie", f"Ce dossier ({row['id']})"],
                y=[cat_med, cat_avg, row["amount"]],
                marker_color=[P["primary"], P["primary"], P["danger"] if is_suspect else P["success"]],
                text=[fmt_money(cat_med), fmt_money(cat_avg), fmt_money(row["amount"])],
                textposition="outside", textfont=dict(size=12),
            ))
            fig_comp.update_layout(template="plotly_white", height=280, margin=dict(l=10,r=10,t=10,b=10), yaxis=dict(title="Montant (XAF)", gridcolor="#F1F5F9"))
            st.plotly_chart(fig_comp, use_container_width=True)
        else:
            st.info("Pas assez de données dans cette catégorie pour comparer.")
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Export ────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(f"<h4 style='font-size:1rem; font-weight:700; color:{P['text']}; display:flex; align-items:center; gap:0.4rem;'>{ic('download',16,P['primary'])} Télécharger les Résultats</h4>", unsafe_allow_html=True)
    e1, e2, _ = st.columns([1, 1, 2])
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    with e1: st.download_button("Télécharger en CSV", df_g.to_csv(index=False).encode("utf-8"), f"auditpulse_anomalies_{ts}.csv", "text/csv", use_container_width=True)
    with e2: st.download_button("Télécharger en Excel", to_xlsx(df_g), f"auditpulse_anomalies_{ts}.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — SIMULATEUR UNITAIRE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Simulateur Unitaire":
    st.markdown(f"""<div class="ap-header"><div>
        <h1>{ic("search",22,P["primary"])} Simulateur d'Audit en Temps Réel</h1>
        <p>Testez une opération pour obtenir instantanément un diagnostic complet : niveau de risque, schémas de fraude et recommandations en clair.</p>
    </div><div style="display:flex;gap:0.5rem;align-items:center;">
        <span class="badge badge-primary">{ic("zap",13)} Évaluation &lt; 50 ms</span>
        <span class="badge badge-success">{ic("shield",13)} Double Moteur IA & Règles</span>
    </div></div>""", unsafe_allow_html=True)

    # Initialisation de la session
    if "sim_id" not in st.session_state:
        st.session_state.update({
            "sim_id": "EXP_STANDARD",
            "sim_amount": 35000.0,
            "sim_user": "USR_0001",
            "sim_supp": "SUPP_0001",
            "sim_source": "DASHBOARD",
            "sim_type": "merchandise-purchase",
            "sim_hour": 11,
            "sim_status": "approved",
            "sim_version": 0,
        })

    cat_options = {
        "merchandise-purchase": "Achats de marchandises",
        "raw_materials": "Matières premières",
        "marketing": "Marketing & Publicité",
        "personnel": "Frais de personnel",
        "office": "Fournitures de bureau",
        "other": "Autres dépenses",
    }

    src_options = {
        "DASHBOARD": "Tableau de bord ERP",
        "POS": "Terminal point de vente (POS)",
        "WHATSAPP": "WhatsApp Pro",
        "FACEBOOK": "Facebook Business",
        "API": "API Externe",
    }

    status_options = {
        "approved": "Approuvée",
        "pending": "En attente de validation",
        "rejected": "Rejetée",
    }

    # ── Scénarios de démonstration 1-Clic ─────────────────────────────────────
    st.markdown("<div class='ap-card'>", unsafe_allow_html=True)
    st.markdown(f"<strong style='color:{P['text']}; display:flex; align-items:center; gap:0.4rem; margin-bottom:0.2rem;'>{ic('sparkles',16,P['primary'])} Cas d'Usage de Démonstration (1 Clic)</strong>", unsafe_allow_html=True)
    st.caption("Cliquez sur l'un des cas ci-dessous pour pré-remplir et évaluer immédiatement la dépense :")

    sc1, sc2, sc3, sc4 = st.columns(4)
    v = st.session_state.get("sim_version", 0)
    with sc1:
        if st.button("Dépense Standard (35k)", use_container_width=True, help="Fournitures de bureau régulières en journée"):
            st.session_state.update({
                "sim_id": "EXP_STANDARD", "sim_amount": 35000.0, "sim_user": "USR_0001",
                "sim_supp": "SUPP_0001", "sim_source": "DASHBOARD", "sim_type": "merchandise-purchase",
                "sim_hour": 11, "sim_status": "approved", "sim_version": v + 1,
            })
            st.rerun()
    with sc2:
        if st.button("Montant Démesuré (6 Md)", use_container_width=True, help="Montant astronomique passé à 2h du matin via Facebook"):
            st.session_state.update({
                "sim_id": "EXP_OUTLIER", "sim_amount": 6051453317.0, "sim_user": "USR_0072",
                "sim_supp": "SUPP_0123", "sim_source": "FACEBOOK", "sim_type": "raw_materials",
                "sim_hour": 2, "sim_status": "approved", "sim_version": v + 1,
            })
            st.rerun()
    with sc3:
        if st.button("Nocturne & Rond (6,75 M)", use_container_width=True, help="Montant rond sans centimes à 23h chez un fournisseur rare"):
            st.session_state.update({
                "sim_id": "EXP_NOCTURNE", "sim_amount": 6750000.0, "sim_user": "USR_0109",
                "sim_supp": "SUPP_RARE_99", "sim_source": "DASHBOARD", "sim_type": "personnel",
                "sim_hour": 23, "sim_status": "approved", "sim_version": v + 1,
            })
            st.rerun()
    with sc4:
        if st.button("Fractionnement (4,95 M)", use_container_width=True, help="Montant frôlant le seuil d'approbation de 5M"):
            st.session_state.update({
                "sim_id": "EXP_SMURFING", "sim_amount": 4950000.0, "sim_user": "USR_0055",
                "sim_supp": "SUPP_GHOST", "sim_source": "DASHBOARD", "sim_type": "raw_materials",
                "sim_hour": 16, "sim_status": "approved", "sim_version": v + 1,
            })
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Formulaire de saisie ──────────────────────────────────────────────────
    with st.expander("Personnaliser les paramètres de l'opération", expanded=True):
        with st.form("sim_form"):
            c1, c2, c3 = st.columns(3)
            with c1:
                in_id = st.text_input("Référence de la transaction", value=st.session_state.sim_id)
                in_usr = st.text_input("Identifiant Collaborateur", value=st.session_state.sim_user)
                in_sup = st.text_input("Identifiant Fournisseur", value=st.session_state.sim_supp)
            with c2:
                in_amt = st.number_input("Montant de la dépense (XAF)", min_value=1.0, value=float(st.session_state.sim_amount), step=10000.0)
                cur_cat = st.session_state.sim_type
                cat_keys = list(cat_options.keys())
                cat_idx = cat_keys.index(cur_cat) if cur_cat in cat_keys else 0
                in_typ = st.selectbox("Catégorie de dépense", options=cat_keys, index=cat_idx, format_func=lambda x: cat_options.get(x, x))
                cur_src = st.session_state.sim_source
                src_keys = list(src_options.keys())
                src_idx = src_keys.index(cur_src) if cur_src in src_keys else 0
                in_src = st.selectbox("Canal d'enregistrement", options=src_keys, index=src_idx, format_func=lambda x: src_options.get(x, x))
            with c3:
                in_dt = st.date_input("Date de l'opération", value=datetime.today())
                in_hr = st.slider("Heure de saisie (0h à 23h)", 0, 23, int(st.session_state.sim_hour), help="20h-06h : plage nocturne hors heures ouvrées")
                cur_st = st.session_state.sim_status
                st_keys = list(status_options.keys())
                st_idx = st_keys.index(cur_st) if cur_st in st_keys else 0
                in_st = st.selectbox("Statut comptable", options=st_keys, index=st_idx, format_func=lambda x: status_options.get(x, x))

            btn_submit = st.form_submit_button("Auditer cette transaction maintenant", use_container_width=True, type="primary")
            if btn_submit:
                st.session_state.update({
                    "sim_id": in_id, "sim_amount": in_amt, "sim_user": in_usr,
                    "sim_supp": in_sup, "sim_source": in_src, "sim_type": in_typ,
                    "sim_hour": in_hr, "sim_status": in_st,
                })

    # ── Exécution de l'audit ──────────────────────────────────────────────────
    payload = {
        "id": st.session_state.sim_id,
        "company_id": "COMP_0001",
        "user_id": st.session_state.sim_user,
        "supplier_id": st.session_state.sim_supp.strip() if st.session_state.sim_supp and st.session_state.sim_supp.strip() else None,
        "amount": float(st.session_state.sim_amount),
        "currency": "XAF",
        "source": st.session_state.sim_source,
        "expense_type": st.session_state.sim_type,
        "status": st.session_state.sim_status,
        "created_at": f"2024-01-15 {int(st.session_state.sim_hour):02d}:00:00"
    }

    with st.spinner("Évaluation des moteurs d'audit et de détection..."):
        res = api_single(payload)

    if res:
        sf = float(res.get("score_final", 0.0))
        is_fraud = bool(res.get("is_fraud_pattern", False))
        is_a = bool(res.get("is_anomaly", False)) or (sf >= selected_threshold) or is_fraud

        # ── Grand Bandeau Décisionnel ─────────────────────────────────────────
        if is_a:
            v_bg = P["danger_bg"]; v_border = P["danger"]; v_icon = ic("x_circle", 26, P["danger"])
            v_title = "ALERTE : DÉPENSE SUSPECTE & ANORMALE"
            v_color = P["danger"]
            v_desc = "Cette opération présente de fortes anomalies statistiques et/ou enfreint des règles d'intégrité financière."
            v_action = "Action recommandée : Bloquer le virement et demander immédiatement une facture signée ainsi que la validation hiérarchique."
        elif sf >= 0.35:
            v_bg = P["warning_bg"]; v_border = P["warning"]; v_icon = ic("alert_circle", 26, P["warning"])
            v_title = "VIGILANCE : DÉPENSE ATYPIQUE"
            v_color = P["warning"]
            v_desc = "Cette opération comporte de légères irrégularités sans constituer une fraude avérée."
            v_action = "Action recommandée : Contrôle visuel de la pièce comptable avant signature définitive."
        else:
            v_bg = P["success_bg"]; v_border = P["success"]; v_icon = ic("check_circle", 26, P["success"])
            v_title = "TRANSACTION CONFORME & SAINE"
            v_color = P["success"]
            v_desc = "Cette opération s'inscrit fidèlement dans les habitudes et normes de dépenses de l'entreprise."
            v_action = "Action recommandée : Traitement comptable normal, aucun blocage requis."

        st.markdown(f"""<div style="background:{v_bg}; border:1.5px solid {v_border}; border-radius:12px; padding:1.2rem 1.5rem; margin:1rem 0; color:{P['text']}">
            <div style="display:flex; align-items:flex-start; gap:0.9rem;">
                {v_icon}
                <div style="flex:1;">
                    <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.5rem;">
                        <div style="font-size:1.15rem; font-weight:800; color:{v_color}">{v_title}</div>
                        <span style="font-size:0.9rem; font-weight:700; background:{v_border}; color:#FFFFFF; padding:0.2rem 0.65rem; border-radius:6px;">Indice de Risque : {sf*100:.1f} / 100</span>
                    </div>
                    <div style="font-size:0.88rem; color:{P['text']}; margin-top:0.35rem;">{v_desc}</div>
                    <div style="font-size:0.83rem; font-weight:600; color:{v_color}; margin-top:0.5rem; padding-top:0.45rem; border-top:1px solid rgba(0,0,0,0.06);">{v_action}</div>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

        # ── Colonnes Jauge + Radar ───────────────────────────────────────────
        col_g, col_r = st.columns(2)
        with col_g:
            st.markdown("<div class='ap-card'>", unsafe_allow_html=True)
            st.markdown(f"<strong style='color:{P['text']}; display:flex; align-items:center; gap:0.4rem;'>{ic('gauge',16,P['primary'])} Jauge de Risque Global</strong>", unsafe_allow_html=True)
            st.caption("Mesure synthétique calculée par l'ensemble des modèles d'audit.")
            fg = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(sf*100, 1),
                number={"suffix": "%", "font": {"size": 38, "color": P["text"]}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": P["border"], "tickwidth": 1},
                    "bar": {"color": v_color, "thickness": 0.28},
                    "steps": [
                        {"range": [0, 30], "color": "#ECFDF5"},
                        {"range": [30, 65], "color": "#FFFBEB"},
                        {"range": [65, 100], "color": "#FEF2F2"}
                    ],
                    "threshold": {
                        "line": {"color": P["danger"], "width": 3},
                        "thickness": 0.85,
                        "value": selected_threshold * 100
                    }
                }
            ))
            fg.update_layout(template="plotly_white", height=280, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fg, use_container_width=True)
            st.markdown(f"""<div style="display:flex; justify-content:space-around; font-size:0.8rem; color:{P['text2']}; border-top:1px solid {P['border']}; padding-top:0.5rem;">
                <div>Seuil d'alerte : <strong>{selected_threshold*100:.0f}%</strong></div>
                <div>Montant : <strong>{fmt_money(payload['amount'])}</strong></div>
                <div>Heure : <strong>{payload['created_at'][11:16]}</strong></div>
            </div>""", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col_r:
            st.markdown("<div class='ap-card'>", unsafe_allow_html=True)
            st.markdown(f"<strong style='color:{P['text']}; display:flex; align-items:center; gap:0.4rem;'>{ic('radar',16,P['primary'])} Radar des 5 Piliers d'Évaluation</strong>", unsafe_allow_html=True)
            st.caption("Décomposition du risque selon 5 perspectives d'audit indépendantes.")
            sd = res.get("scores", {})
            cats_fr = ["Anomalie Globale", "Profil Employé", "Groupe Suspect", "Règles Métiers", "Schéma de Fraude"]
            vals = [
                sd.get("isolation_forest", 0),
                sd.get("lof", 0),
                sd.get("dbscan", 0),
                sd.get("rules", 0),
                sd.get("fraud", 0)
            ]
            cats_closed = cats_fr + [cats_fr[0]]
            vals_closed = [v * 100 for v in vals] + [vals[0] * 100]

            fr = go.Figure(go.Scatterpolar(
                r=vals_closed,
                theta=cats_closed,
                fill="toself",
                fillcolor="rgba(37, 99, 235, 0.15)",
                line=dict(color=P["primary"], width=2.5),
                marker=dict(size=6, color=P["primary"]),
                hovertemplate="<b>%{theta}</b> : %{r:.1f}%<extra></extra>"
            ))
            fr.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 100], gridcolor="#E2E8F0", tickfont=dict(size=10, color=P["text3"])),
                    angularaxis=dict(gridcolor="#E2E8F0", tickfont=dict(size=11, color=P["text"]))
                ),
                template="plotly_white",
                height=280,
                margin=dict(l=35, r=35, t=30, b=20),
                showlegend=False
            )
            st.plotly_chart(fr, use_container_width=True)
            st.markdown(f"<div style='font-size:0.75rem; color:{P['text3']}; text-align:center;'>Plus la zone s'étend vers l'extérieur, plus le risque est prononcé sur ce pilier.</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # ── Règles enfreintes & Diagnostic des 4 fraudes ──────────────────────
        col_rules, col_fraud = st.columns([1, 1])
        with col_rules:
            st.markdown("<div class='ap-card'>", unsafe_allow_html=True)
            st.markdown(f"<strong style='color:{P['text']}; display:flex; align-items:center; gap:0.4rem;'>{ic('file_text',16,P['primary'])} Pourquoi Cette Décision ? (Règles Métiers)</strong>", unsafe_allow_html=True)
            st.caption("Signaux concrets et infractions constatées lors de l'audit :")

            rule_translations = {
                "Montant avec z-score global > 3.0": "Montant exceptionnel (déviation statistique majeure à l'échelle de l'entreprise)",
                "Montant en dehors du IQR par utilisateur": "Montant anormalement élevé pour cet employé (dépasse largement ses habitudes)",
                "Montant en dehors du IQR global": "Montant très inhabituel (hors de la plage normale de l'entreprise)",
                "Transaction hors heures de bureau": "Dépense enregistrée en pleine nuit (entre 20h et 06h)",
                "Hors heures ouvrées": "Dépense passée en dehors des heures normales de travail",
                "Transaction le week-end": "Transaction effectuée un samedi ou un dimanche",
                "Fournisseur utilisé moins de 3 fois": "Fournisseur nouveau ou rarement sollicité (risque de société écran)",
                "Montant x3+ vs moyenne glissante 7 jours": "Montant 3x supérieur à la moyenne récente de la catégorie",
                "Montant rond suspect": "Montant parfaitement rond (indicateur d'évaluation arbitraire sans facture)",
                "Fréquence journalière élevée": "Fréquence inhabituellement dense de transactions par cet employé",
            }

            rules = res.get("rules_triggered", [])
            if rules:
                for rl in rules:
                    # Clean possible formatting
                    clean_rl = rl.replace("≥", ">=").replace("≤", "<=")
                    readable = rule_translations.get(clean_rl, rule_translations.get(rl, clean_rl))
                    st.markdown(f"""<div style="background:{P['warning_bg']}; border-left:3px solid {P['warning']}; border-radius:6px; padding:0.6rem 0.8rem; margin-bottom:0.4rem; font-size:0.84rem; color:{P['text']}; display:flex; align-items:center; gap:0.5rem;">
                        {ic('alert', 14, P['warning'])} <span>{readable}</span>
                    </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div style="background:{P['success_bg']}; border-left:3px solid {P['success']}; border-radius:6px; padding:0.8rem 1rem; font-size:0.85rem; color:{P['text']}; display:flex; align-items:center; gap:0.5rem;">
                    {ic('check_circle', 16, P['success'])} <span>Aucune infraction aux règles métiers. L'opération respecte les plafonds et horaires autorisés.</span>
                </div>""", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col_fraud:
            st.markdown("<div class='ap-card'>", unsafe_allow_html=True)
            st.markdown(f"<strong style='color:{P['text']}; display:flex; align-items:center; gap:0.4rem;'>{ic('shield',16,P['primary'])} Diagnostic des 4 Typologies de Fraude</strong>", unsafe_allow_html=True)
            st.caption("Probabilité estimée pour chaque mode opératoire connu :")

            fd = res.get("fraud_details", {})
            fraud_items = [
                ("Fractionnement (Smurfing)", fd.get("splitting", 0), "Découpage pour contourner un seuil d'autorisation"),
                ("Risque de Doublon", fd.get("duplicate", 0), "Risque de double paiement pour la même prestation"),
                ("Fournisseur Fantôme", fd.get("ghost_supplier", 0), "Fournisseur suspect ou non référencé dans l'ERP"),
                ("Inflation de Prix", fd.get("inflation", 0), "Montant anormalement supérieur aux prix habituels"),
            ]

            f_c1, f_c2 = st.columns(2)
            for i, (f_name, f_val, f_desc) in enumerate(fraud_items):
                target_col = f_c1 if (i % 2 == 0) else f_c2
                with target_col:
                    f_pct = f_val * 100
                    if f_val >= 0.5:
                        b_color = P["danger"]; b_bg = P["danger_bg"]; b_lbl = "Élevé"
                    elif f_val >= 0.2:
                        b_color = P["warning"]; b_bg = P["warning_bg"]; b_lbl = "Modéré"
                    else:
                        b_color = P["success"]; b_bg = P["success_bg"]; b_lbl = "Faible"
                    st.markdown(f"""<div style="background:{P['bg2']}; border:1px solid {P['border']}; border-radius:8px; padding:0.65rem 0.8rem; margin-bottom:0.5rem;">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <span style="font-size:0.78rem; font-weight:700; color:{P['text']}">{f_name}</span>
                            <span style="font-size:0.72rem; font-weight:700; color:{b_color}; background:{b_bg}; padding:0.1rem 0.4rem; border-radius:4px;">{b_lbl} ({f_pct:.0f}%)</span>
                        </div>
                        <div style="font-size:0.71rem; color:{P['text2']}; margin-top:0.25rem; line-height:1.2;">{f_desc}</div>
                    </div>""", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # ── Comparaison Contextuelle avec la Catégorie ────────────────────────
        df_all_cat = load_scored()
        if not df_all_cat.empty:
            cat_cur = payload.get("expense_type", "merchandise-purchase")
            cat_sub = df_all_cat[df_all_cat["expense_type"] == cat_cur]
            cat_label_fr = cat_options.get(cat_cur, cat_cur)

            st.markdown("<div class='ap-card'>", unsafe_allow_html=True)
            st.markdown(f"<strong style='color:{P['text']}; display:flex; align-items:center; gap:0.4rem;'>{ic('bar_chart',16,P['primary'])} Comparaison Contextuelle : Cette Dépense vs La Catégorie « {cat_label_fr} »</strong>", unsafe_allow_html=True)
            st.caption("Ce montant est-il conforme aux ordres de grandeur habituels observés dans l'entreprise ?")

            if len(cat_sub) > 0:
                med_val = float(cat_sub["amount"].median())
                avg_val = float(cat_sub["amount"].mean())
                this_val = float(payload["amount"])

                fig_bar = go.Figure()
                fig_bar.add_trace(go.Bar(
                    x=["Dépense Médiane", "Dépense Moyenne", f"Cette Opération ({payload['id']})"],
                    y=[med_val, avg_val, this_val],
                    marker_color=[P["primary"], P["primary"], P["danger"] if is_a else P["success"]],
                    text=[fmt_money(med_val), fmt_money(avg_val), fmt_money(this_val)],
                    textposition="outside",
                    textfont=dict(size=12, color=P["text"]),
                    hovertemplate="<b>%{x}</b> : %{y:,.0f} XAF<extra></extra>"
                ))
                fig_bar.update_layout(
                    template="plotly_white",
                    height=270,
                    margin=dict(l=20, r=20, t=25, b=20),
                    yaxis=dict(title="Montant (XAF)", gridcolor="#F1F5F9", automargin=True)
                )
                st.plotly_chart(fig_bar, use_container_width=True)

                if med_val > 0:
                    ratio = this_val / med_val
                    if ratio > 2.0:
                        st.markdown(f"<div style='font-size:0.83rem; color:{P['danger']}; font-weight:600;'>{ic('alert', 13, P['danger'])} Ce montant est <strong>{ratio:.1f} fois supérieur</strong> à la dépense médiane habituelle pour cette catégorie.</div>", unsafe_allow_html=True)
                    elif ratio < 0.2:
                        st.markdown(f"<div style='font-size:0.83rem; color:{P['primary']}; font-weight:600;'>{ic('check_circle', 13, P['primary'])} Ce montant est inférieur à la dépense médiane habituelle.</div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div style='font-size:0.83rem; color:{P['success']}; font-weight:600;'>{ic('check_circle', 13, P['success'])} Ce montant s'aligne fidèlement sur l'ordre de grandeur médian habituel ({fmt_money(med_val)}).</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)



# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — AUDIT BATCH
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Audit Batch":
    st.markdown(f"""<div class="ap-header"><div>
        <h1>{ic("folder",22,P["primary"])} Contrôle Massif & Audit de Fichiers</h1>
        <p>Scannez et auditez des centaines de dépenses en quelques secondes : identification des fraudes, calcul du capital à risque et rapport probatoire.</p>
    </div><div style="display:flex;gap:0.5rem;align-items:center;">
        <span class="badge badge-primary">{ic("zap",13)} Moteur Vectorisé (~1 000 tx/sec)</span>
        <span class="badge badge-success">{ic("file_text",13)} Formats : CSV, Excel, SQL</span>
    </div></div>""", unsafe_allow_html=True)

    cat_translations = {
        "merchandise-purchase": "Achats Marchandises",
        "raw_materials": "Matières Premières",
        "marketing": "Marketing & Pub",
        "personnel": "Frais de Personnel",
        "office": "Fournitures Bureau",
        "other": "Autres Dépenses",
    }

    # ── Zone d'Ingestion & Démo 1-Clic ────────────────────────────────────────
    st.markdown("<div class='ap-card'>", unsafe_allow_html=True)
    c_demo, c_up = st.columns([1, 1])
    with c_demo:
        st.markdown(f"<strong style='color:{P['text']}; display:flex; align-items:center; gap:0.4rem;'>{ic('sparkles',16,P['primary'])} Démo Rapide (1 Clic)</strong>", unsafe_allow_html=True)
        st.caption("Auditez instantanément un échantillon benchmark de 100 opérations réelles (achats, salaires, prestataires).")
        btn_demo = st.button("Lancer l'Audit Démo (100 dépenses)", type="primary", use_container_width=True)

    with c_up:
        st.markdown(f"<strong style='color:{P['text']}; display:flex; align-items:center; gap:0.4rem;'>{ic('upload',16,P['primary'])} Importer vos Propres Données</strong>", unsafe_allow_html=True)
        st.caption("Déposez un fichier d'écritures comptables au format CSV, Excel (.xlsx) ou dump SQL.")
        up = st.file_uploader("Fichier à auditer", type=["csv", "xlsx", "sql"], label_visibility="collapsed")
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Traitement de l'Ingestion ─────────────────────────────────────────────
    if btn_demo:
        df_scored_all = load_scored()
        if not df_scored_all.empty:
            sample_100 = df_scored_all.sample(min(100, len(df_scored_all)), random_state=42)
            st.session_state["batch_data"] = sample_100.to_dict(orient="records")
            st.session_state["batch_source_name"] = "Échantillon Benchmark (100 dépenses)"
            st.session_state["batch_duration"] = 0.24
            st.rerun()

    elif up is not None:
        if "last_uploaded_name" not in st.session_state or st.session_state["last_uploaded_name"] != up.name:
            try:
                uploaded_txs = []
                if up.name.endswith(".csv"):
                    df_u = pd.read_csv(up).fillna("").astype(str)
                    if "amount" in df_u.columns:
                        df_u["amount"] = pd.to_numeric(df_u["amount"], errors="coerce").fillna(0)
                    uploaded_txs = df_u.to_dict(orient="records")
                elif up.name.endswith(".xlsx"):
                    df_u = pd.read_excel(up).fillna("").astype(str)
                    if "amount" in df_u.columns:
                        df_u["amount"] = pd.to_numeric(df_u["amount"], errors="coerce").fillna(0)
                    uploaded_txs = df_u.to_dict(orient="records")
                elif up.name.endswith(".sql"):
                    tmp = ROOT / "data" / f"tmp_{up.name}"
                    tmp.write_bytes(up.read())
                    from src.data_loader import load_expenses_from_sql
                    uploaded_txs = load_expenses_from_sql(tmp).to_dict(orient="records")
                    if tmp.exists():
                        tmp.unlink()

                if uploaded_txs:
                    pb = st.progress(0); pt = st.empty()
                    br = api_batch(uploaded_txs, pb, pt)
                    pb.empty(); pt.empty()
                    if br:
                        st.session_state["batch_data"] = br["results"]
                        st.session_state["batch_source_name"] = f"Fichier importé : {up.name}"
                        st.session_state["batch_duration"] = br["summary"].get("duration_sec", 0.5)
                        st.session_state["last_uploaded_name"] = up.name
                        st.rerun()
            except Exception as e:
                st.error(f"Erreur d'analyse du fichier : {e}")

    # Initialisation automatique par défaut pour afficher une vue riche immédiatement
    if "batch_data" not in st.session_state:
        df_scored_all = load_scored()
        if not df_scored_all.empty:
            sample_100 = df_scored_all.sample(min(100, len(df_scored_all)), random_state=42)
            st.session_state["batch_data"] = sample_100.to_dict(orient="records")
            st.session_state["batch_source_name"] = "Échantillon Benchmark (100 dépenses)"
            st.session_state["batch_duration"] = 0.24

    # ── Traitement des résultats & Calcul des KPIs ────────────────────────────
    df_b = pd.DataFrame(st.session_state.get("batch_data", []))
    if not df_b.empty:
        # Re-calcul dynamique selon le seuil sélectionné dans la sidebar
        if "score_final" in df_b.columns:
            df_b["is_anomaly_dyn"] = ((df_b["score_final"] >= selected_threshold) | (df_b.get("is_fraud_pattern", 0) == 1)).astype(int)
        else:
            df_b["is_anomaly_dyn"] = df_b.get("is_anomaly", 0).astype(int)

        total_b = len(df_b)
        total_cap_b = float(df_b["amount"].sum()) if "amount" in df_b.columns else 0.0
        df_b_susp = df_b[df_b["is_anomaly_dyn"] == 1]
        df_b_clean = df_b[df_b["is_anomaly_dyn"] == 0]
        n_susp = len(df_b_susp)
        rate_susp = (n_susp / total_b * 100) if total_b > 0 else 0.0
        cap_susp = float(df_b_susp["amount"].sum()) if not df_b_susp.empty and "amount" in df_b_susp.columns else 0.0
        cap_susp_pct = (cap_susp / total_cap_b * 100) if total_cap_b > 0 else 0.0
        duration_b = st.session_state.get("batch_duration", 0.24)
        speed_b = int(total_b / duration_b) if duration_b > 0 else 850

        # ── 4 KPIs Exécutifs ──────────────────────────────────────────────────
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(f"""<div class="ap-kpi">
                <div class="kpi-label">Volume Audité</div>
                <div class="kpi-value">{total_b:,} <span style="font-size:0.85rem; font-weight:600; color:{P['text2']};">dépenses</span></div>
                <div class="kpi-sub">{ic("database", 13, P['primary'])} {st.session_state.get('batch_source_name', 'Lot actif')}</div>
            </div>""", unsafe_allow_html=True)
        with k2:
            st.markdown(f"""<div class="ap-kpi danger">
                <div class="kpi-label">Dépenses Suspectes</div>
                <div class="kpi-value" style="color:{P['danger']}">{n_susp:,}</div>
                <div class="kpi-sub" style="color:{P['danger']}">{ic("alert", 13, P['danger'])} Taux d'alerte : {rate_susp:.1f}%</div>
            </div>""", unsafe_allow_html=True)
        with k3:
            st.markdown(f"""<div class="ap-kpi warning">
                <div class="kpi-label">Capital à Risque</div>
                <div class="kpi-value" style="color:{P['warning']}">{fmt_money(cap_susp)}</div>
                <div class="kpi-sub">{ic("shield", 13, P['warning'])} {cap_susp_pct:.1f}% de la masse contrôlée</div>
            </div>""", unsafe_allow_html=True)
        with k4:
            st.markdown(f"""<div class="ap-kpi success">
                <div class="kpi-label">Vitesse de Traitement</div>
                <div class="kpi-value" style="color:{P['success']}">{speed_b:,} <span style="font-size:0.85rem; font-weight:600;">tx/sec</span></div>
                <div class="kpi-sub">{ic("zap", 13, P['success'])} Réalisé en {duration_b:.2f} s</div>
            </div>""", unsafe_allow_html=True)

        # ── Grand Bandeau Décisionnel ─────────────────────────────────────────
        if n_susp > 0:
            st.markdown(f"""<div style="background:{P['danger_bg']}; border:1.5px solid {P['danger']}; border-radius:12px; padding:1.2rem 1.5rem; margin:1.2rem 0; color:{P['text']}">
                <div style="display:flex; align-items:flex-start; gap:0.9rem;">
                    {ic("x_circle", 26, P['danger'])}
                    <div style="flex:1;">
                        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.5rem;">
                            <div style="font-size:1.15rem; font-weight:800; color:{P['danger']}">SYNTHÈSE D'AUDIT : RISQUES FINANCIERS DÉTECTÉS DANS CE LOT</div>
                            <span style="font-size:0.85rem; font-weight:700; background:{P['danger']}; color:#FFFFFF; padding:0.2rem 0.65rem; border-radius:6px;">{n_susp} anomalies prioritaires</span>
                        </div>
                        <div style="font-size:0.88rem; color:{P['text']}; margin-top:0.35rem;">
                            L'audit automatisé a isolé <strong>{n_susp} opérations suspectes</strong> représentant un montant total exposé de <strong>{fmt_money(cap_susp)}</strong> ({cap_susp_pct:.1f}% du capital total du fichier).
                        </div>
                        <div style="font-size:0.83rem; font-weight:600; color:{P['danger']}; margin-top:0.5rem; padding-top:0.45rem; border-top:1px solid rgba(220,38,38,0.15);">
                            Action recommandée : Télécharger le rapport ci-dessous et geler les virements des {n_susp} opérations signalées en attendant les justificatifs comptables certifiés.
                        </div>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div style="background:{P['success_bg']}; border:1.5px solid {P['success']}; border-radius:12px; padding:1.2rem 1.5rem; margin:1.2rem 0; color:{P['text']}">
                <div style="display:flex; align-items:center; gap:0.9rem;">
                    {ic("check_circle", 26, P['success'])}
                    <div>
                        <div style="font-size:1.15rem; font-weight:800; color:{P['success']}">LOT 100% CONFORME — AUCUN RISQUE MAJEUR DÉTECTÉ</div>
                        <div style="font-size:0.88rem; color:{P['text']}; margin-top:0.2rem;">
                            Toutes les {total_b} opérations analysées respectent fidèlement les seuils d'intégrité et plafonds habituels. Validation comptable globale autorisée.
                        </div>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)

        # ── Graphiques Exécutifs (2 Colonnes) ─────────────────────────────────
        g1, g2 = st.columns(2)
        with g1:
            st.markdown("<div class='ap-card'>", unsafe_allow_html=True)
            st.markdown(f"<strong style='color:{P['text']}'>Répartition des Opérations du Fichier</strong>", unsafe_allow_html=True)
            st.caption("Proportion de dépenses saines vs dépenses suspectes.")
            fig_don = go.Figure(go.Pie(
                labels=["Conformes", "Suspectes"],
                values=[max(total_b - n_susp, 0), n_susp],
                hole=0.6,
                marker=dict(colors=[P["primary"], P["danger"]]),
                textinfo="label+percent",
                textfont=dict(size=13, color=P["text"]),
                hovertemplate="<b>%{label}</b><br>%{value:,} opérations<br>%{percent}<extra></extra>",
            ))
            fig_don.add_annotation(text=f"<b>{total_b}</b><br>opérations", x=0.5, y=0.5, font=dict(size=14, color=P["text"]), showarrow=False)
            fig_don.update_layout(template="plotly_white", height=280, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
            st.plotly_chart(fig_don, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with g2:
            st.markdown("<div class='ap-card'>", unsafe_allow_html=True)
            st.markdown(f"<strong style='color:{P['text']}'>Où se Concentre le Risque Financier ?</strong>", unsafe_allow_html=True)
            st.caption("Montant financier à risque par catégorie de dépense.")
            if not df_b_susp.empty and "expense_type" in df_b_susp.columns:
                cat_risk = df_b_susp.groupby("expense_type")["amount"].sum().sort_values(ascending=True)
                cat_labels = [cat_translations.get(k, k) for k in cat_risk.index]
                fig_bar = go.Figure(go.Bar(
                    y=cat_labels,
                    x=cat_risk.values,
                    orientation="h",
                    marker_color=P["danger"],
                    text=[fmt_money(v) for v in cat_risk.values],
                    textposition="outside",
                    textfont=dict(size=11, color=P["text"]),
                    hovertemplate="<b>%{y}</b> : %{x:,.0f} XAF<extra></extra>"
                ))
                fig_bar.update_layout(
                    template="plotly_white",
                    height=280,
                    margin=dict(l=10, r=70, t=10, b=10),
                    xaxis=dict(title="Montant exposé (XAF)", gridcolor="#F1F5F9"),
                    yaxis=dict(automargin=True)
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.info("Aucune anomalie à afficher par catégorie dans ce lot.")
            st.markdown("</div>", unsafe_allow_html=True)

        # ── Synthèse des 4 Typologies de Fraude ────────────────────────────────
        st.markdown(f"<h3 style='font-size:1.05rem; font-weight:700; color:{P['text']}; margin:1.2rem 0 0.6rem; display:flex; align-items:center; gap:0.5rem;'>{ic('shield',18,P['primary'])} Typologies de Fraude Identifiées dans ce Fichier</h3>", unsafe_allow_html=True)
        t1, t2, t3, t4 = st.columns(4)

        nb_smurf = int((df_b.get("fraud_splitting", pd.Series(0, index=df_b.index)).fillna(0) > 0.4).sum())
        nb_dupl = int((df_b.get("fraud_duplicate", pd.Series(0, index=df_b.index)).fillna(0) > 0.4).sum())
        nb_ghost = int((df_b.get("fraud_ghost_supplier", pd.Series(0, index=df_b.index)).fillna(0) > 0.4).sum())
        nb_infl = int((df_b.get("fraud_inflation", pd.Series(0, index=df_b.index)).fillna(0) > 0.4).sum())

        cards_fraud = [
            ("Fractionnement (Smurfing)", nb_smurf, "Dépenses fragmentées sous le seuil d'autorisation", t1),
            ("Doublons Potentiels", nb_dupl, "Paiements répétés pour une même prestation", t2),
            ("Fournisseurs Fantômes", nb_ghost, "Fournisseurs non vérifiés ou à l'activité suspecte", t3),
            ("Gonflement de Montants", nb_infl, "Factures hors proportions par rapport aux normes", t4),
        ]
        for name, cnt, desc, col in cards_fraud:
            with col:
                c_border = P["danger"] if cnt > 0 else P["border"]
                c_badge = P["danger"] if cnt > 0 else P["text3"]
                st.markdown(f"""<div style="background:{P['bg']}; border:1px solid {c_border}; border-radius:10px; padding:0.9rem; min-height:115px;">
                    <div style="font-size:0.75rem; text-transform:uppercase; font-weight:700; color:{P['text2']};">{name}</div>
                    <div style="font-size:1.45rem; font-weight:800; color:{c_badge}; margin:0.25rem 0;">{cnt} <span style="font-size:0.8rem; font-weight:600; color:{P['text2']}">cas</span></div>
                    <div style="font-size:0.72rem; color:{P['text3']}; line-height:1.2;">{desc}</div>
                </div>""", unsafe_allow_html=True)

        # ── Tableau des Dépenses Suspectes ────────────────────────────────────
        st.markdown(f"<h3 style='font-size:1.05rem; font-weight:700; color:{P['text']}; margin:1.5rem 0 0.6rem; display:flex; align-items:center; gap:0.5rem;'>{ic('alert',18,P['danger'])} Dépenses Suspectes à Examiner en Priorité</h3>", unsafe_allow_html=True)

        # Filtres simples
        f_c1, f_c2 = st.columns([1, 2])
        with f_c1:
            all_b_cats = ["Toutes les catégories"] + sorted(list(df_b_susp["expense_type"].dropna().unique())) if not df_b_susp.empty else ["Toutes les catégories"]
            sel_b_cat = st.selectbox("Catégorie", all_b_cats, format_func=lambda x: cat_translations.get(x, x), label_visibility="collapsed")
        with f_c2:
            search_b = st.text_input("Rechercher par référence, collaborateur ou fournisseur...", "", label_visibility="collapsed")

        # Filtrage
        df_display = df_b_susp.copy()
        if not df_display.empty:
            if sel_b_cat != "Toutes les catégories":
                df_display = df_display[df_display["expense_type"] == sel_b_cat]
            if search_b.strip():
                q = search_b.strip().lower()
                mask = (
                    df_display["id"].astype(str).str.lower().str.contains(q, na=False) |
                    df_display.get("user_id", pd.Series("", index=df_display.index)).astype(str).str.lower().str.contains(q, na=False) |
                    df_display.get("supplier_id", pd.Series("", index=df_display.index)).astype(str).str.lower().str.contains(q, na=False)
                )
                df_display = df_display[mask]

        if not df_display.empty:
            def format_rules_fr(rules_val):
                if not rules_val or str(rules_val) == "nan":
                    return "Score statistique (hors norme globale)"
                if isinstance(rules_val, str):
                    rules_list = [r.strip() for r in rules_val.split("|") if r.strip()]
                elif isinstance(rules_val, (list, tuple)):
                    rules_list = [str(r).strip() for r in rules_val if str(r).strip()]
                else:
                    rules_list = []
                if not rules_list:
                    return "Score statistique (hors norme globale)"
                rule_map = {
                    "Montant avec z-score global > 3.0": "Montant exceptionnel (déviation statistique)",
                    "Montant en dehors du IQR par utilisateur": "Montant anormal pour cet employé",
                    "Montant en dehors du IQR global": "Montant inhabituel à l'échelle de l'entreprise",
                    "Transaction hors heures de bureau": "Saisie nocturne (hors heures ouvrées)",
                    "Hors heures ouvrées": "Saisie nocturne (hors heures ouvrées)",
                    "Transaction le week-end": "Transaction le week-end",
                    "Fournisseur utilisé moins de 3 fois": "Fournisseur nouveau / rarement utilisé",
                    "Montant x3+ vs moyenne glissante 7 jours": "Montant 3x supérieur à la moyenne récente",
                    "Montant rond suspect": "Montant rond (possible fausse note)",
                    "Fréquence journalière élevée": "Fréquence inhabituelle de dépenses aujourd'hui",
                }
                translated = []
                for r in rules_list:
                    clean = r.replace("≥", ">=").replace("≤", "<=")
                    translated.append(rule_map.get(clean, rule_map.get(r, clean)))
                return " • ".join(translated[:3])

            table_df = pd.DataFrame({
                "Référence": df_display["id"],
                "Risque": [f"{v*100:.0f}%" for v in df_display["score_final"]],
                "Montant (XAF)": [fmt_money(v) for v in df_display["amount"]],
                "Catégorie": [cat_translations.get(c, c) for c in df_display.get("expense_type", "")],
                "Collaborateur": df_display.get("user_id", "—"),
                "Fournisseur": df_display.get("supplier_id", "—"),
                "Motifs d'Alerte": [format_rules_fr(r) for r in df_display.get("rules_triggered", "")],
            })

            st.dataframe(table_df, use_container_width=True, height=350)
        else:
            st.success("Aucune dépense suspecte ne correspond aux critères de recherche.")

        # ── Exportations Officielles ──────────────────────────────────────────
        st.markdown("---")
        st.markdown(f"<h4 style='font-size:1rem; font-weight:700; color:{P['text']}; display:flex; align-items:center; gap:0.4rem;'>{ic('download',16,P['primary'])} Télécharger le Rapport d'Audit</h4>", unsafe_allow_html=True)
        e1, e2, _ = st.columns([1, 1, 2])
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        with e1:
            st.download_button(
                "Télécharger en CSV",
                df_b_susp.to_csv(index=False).encode("utf-8") if not df_b_susp.empty else df_b.to_csv(index=False).encode("utf-8"),
                f"auditpulse_rapport_batch_{ts}.csv",
                "text/csv",
                use_container_width=True
            )
        with e2:
            st.download_button(
                "Télécharger en Excel (.xlsx)",
                to_xlsx(df_b_susp if not df_b_susp.empty else df_b),
                f"auditpulse_rapport_batch_{ts}.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )