import html
import json
import re
import subprocess
import unicodedata
import difflib
from datetime import date, datetime
from io import BytesIO
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import streamlit as st
import streamlit.components.v1 as components


st.set_page_config(
    page_title="Paraty by UTMB 2026",
    page_icon="🏔️",
    layout="wide",
    initial_sidebar_state="expanded",
)


EXCHANGE_RATE_USD_TO_BRL = 5.0
YOPP_PRICE_BRL = 159.0
YOPP_PRICE_USD = 32.0
PTR17_BUS_PRICE_BRL = 30.0
PTR17_BUS_PRICE_USD = 6.0
MP_GROSS_TARGET_BRL = 2_000_000.0
EDITION_TO_CURRENCY = {
    "691336e655be7bd5663d55ef": "USD",
    "6985ddfb09a601887cdbec29": "BRL",
}

DEFAULT_MODULES = [
    "KPIs gerais e metas",
    "Projeções e ritmo de vendas",
    "Demografia",
    "Geografia Brasil",
    "Internacional",
    "Comparativo histórico",
    "Padrões de venda",
    "Financeiro",
    "Exportações",
]
REQUIRED_COLUMNS = [
    "Edition ID",
    "Registration date",
    "Registered status",
    "Competition",
    "Registration amount",
    "Total registration amount",
    "Total discounts amount",
]
AI_GENDER_CANDIDATES = ["gender", "gênero", "genero", "sexo", "sex"]
AI_FEMALE_TOKENS = {"f", "female"}
AI_MALE_TOKENS = {"m", "male"}
MIN_VALID_YEAR = 2018
MAX_VALID_YEAR = 2035

PERCURSO_ORDER = ["PTR 108", "PTR 58", "PTR 34", "PTR 25", "PTR 17", "RUN 7"]
DEFAULT_PERCURSO_TARGETS = {
    "PTR 108": 400,
    "PTR 58": 900,
    "PTR 34": 1000,
    "PTR 25": 1000,
    "PTR 17": 600,
    "RUN 7": 600,
}

# Regras para classificar cupons pelo prefixo (3 primeiras letras).
# Atualize este dicionário quando enviar a lista oficial de lógica.
COUPON_PREFIX_RULES = {
    "IFL": "INFLUENCIADORES",
    "IDS": "IDOSO",
    "PCD": "PESSOA COM DEFICIÊNCIA",
    "MDS": "MORADORES DE PARATY E REGIÃO",
    "PRA": "PREFEITURA (CORTESIAS)",
    "CTA": "CORTESIA",
    "ETE": "ATLETA DE ELITE",
    "REM": "REMANEJADAS (VINDOS DE OUTROS ANOS)",
    "VIP": "VIP",
    "EXP": "EXPOSITORES",
    "PAT": "PATROCINADORES",
    "UPG": "UPGRADE (MUDANÇA DE PERCURSO)",
    "DWG": "DOWNGRADE (MUDANÇA DE PERCURSO)",
    "CPD": "CONTRAPARTIDA (PATROCINADORES E PARCEIROS)",
    "IDX": "INDEX (DESCONTO PELO UTMB INDEX DO ATLETA)",
    "NUB": "NUBANK (CONVIDADOS NUBANK OU AJUSTES DE PREÇO)",
    "CEX": "CONCESSÃO EXCEPCIONAL (OUTROS)",
    "COL": "COLLO (VENDA MANUAL PARA ARGENTINOS EM DÓLAR)",
}

HISTORICAL_VENN_FILES = {
    "2023": ["UTMB - 2023 - USD.xlsx"],
    "2024": ["UTMB - 2024 - USD.xlsx"],
    "2025": [
        "UTMB - 2025 - BRL.xlsx",
        "UTMB - 2025 - USD.xlsx",
    ],
}


def apply_theme() -> None:
    st.markdown(
        """
        <style>
        .main {
            background: linear-gradient(180deg, #F7F9FC 0%, #FFFFFF 100%);
        }
        .hero {
            background: linear-gradient(135deg, #0b2447 0%, #19376d 100%);
            border-radius: 16px;
            padding: 20px 24px;
            color: white;
            margin-bottom: 12px;
        }
        .hero h2, .hero p {
            margin: 0;
            padding: 0;
        }
        .hero-report .hero-row {
            display: flex;
            align-items: center;
            gap: 16px;
        }
        .hero-report .hero-badge {
            display: inline-block;
            font-weight: 700;
            font-size: 0.78rem;
            letter-spacing: 1px;
            color: white;
            padding: 6px 12px;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.15);
            white-space: nowrap;
        }
        .hero-report .hero-text {
            flex: 1;
        }
        .subtle {
            color: #6b7280;
            font-size: 0.9rem;
        }
        .module-card {
            background: #f8fafc;
            border: 1px solid #e5e7eb;
            border-radius: 12px;
            padding: 12px 16px;
            margin-bottom: 10px;
        }
        .highlight-card {
            background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
            border: 1px solid #e2e8f0;
            border-left: 4px solid #0b2447;
            border-radius: 10px;
            padding: 10px 14px;
            margin-bottom: 8px;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
        }
        .highlight-card .highlight-title {
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: #64748b;
        }
        .highlight-card .highlight-value {
            font-size: 1.05rem;
            font-weight: 700;
            color: #0b2447;
            margin-top: 2px;
        }
        .highlight-card .highlight-hint {
            font-size: 0.82rem;
            color: #475569;
            margin-top: 2px;
        }
        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #eef2f7;
            border-radius: 10px;
            padding: 10px 12px;
        }
        div[data-testid="stMetric"] label {
            color: #475569 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def apply_print_css(print_mode: bool = False, report_mode: str | None = None) -> None:
    """Injeta CSS de impressão.

    `report_mode` pode ser "diario" (1 página, sem quebras forçadas) ou
    "semanal" (multipágina, quebra antes de cada h2 para manter blocos íntegros).
    Para os demais relatórios o comportamento original é mantido.
    """
    screen_preview_css = """
        [data-testid="stMainBlockContainer"] {
          max-width: 980px !important;
          margin: 0 auto !important;
          padding: 1.2rem 1.2rem 2.2rem 1.2rem !important;
          background: #ffffff !important;
          border-radius: 12px;
          box-shadow: 0 6px 24px rgba(15, 23, 42, 0.08);
        }
    """
    if not print_mode:
        screen_preview_css = ""

    if report_mode == "diario":
        mode_specific_css = """
          /* Diário: caber em uma página A4. Tudo evita break, fontes ligeiramente menores. */
          #print-content { font-size: 11px !important; }
          #print-content h2 { font-size: 18px !important; }
          #print-content h3, #print-content h4 { font-size: 14px !important; }
          #print-content .hero { padding: 10px 14px !important; margin-bottom: 8px !important; }
          #print-content .hero h2 { font-size: 18px !important; }
          #print-content [data-testid="stMetric"] {
            padding-top: 4px !important;
            padding-bottom: 4px !important;
          }
          #print-content .plotly-graph-div { max-height: 280px !important; }
          #print-content,
          #print-content .element-container,
          #print-content .stTable,
          #print-content .plotly-graph-div,
          #print-content div[data-testid="stMetric"],
          #print-content .highlight-card {
            page-break-inside: avoid !important;
            break-inside: avoid-page !important;
          }
          #print-content .mkt-kpi-card, #print-content .mkt-kpi-wrap {
            page-break-inside: avoid !important;
            break-inside: avoid-page !important;
          }
          #print-content h2, #print-content h3 {
            page-break-before: auto !important;
            break-before: auto !important;
          }
        """
    elif report_mode == "semanal":
        mode_specific_css = """
          /* Semanal: paginação por bloco, com quebra antes de cada h2 (exceto o primeiro). */
          #print-content h2 {
            page-break-before: always !important;
            break-before: page !important;
          }
          #print-content > div:first-of-type h2,
          #print-content .hero + * h2:first-of-type {
            page-break-before: avoid !important;
            break-before: avoid-page !important;
          }
          #print-content .element-container,
          #print-content .stTable,
          #print-content .plotly-graph-div,
          #print-content div[data-testid="stMetric"],
          #print-content .highlight-card {
            page-break-inside: avoid !important;
            break-inside: avoid-page !important;
          }
          #print-content .mkt-kpi-card, #print-content .mkt-kpi-wrap {
            page-break-inside: avoid !important;
            break-inside: avoid-page !important;
          }
        """
    else:
        mode_specific_css = ""

    css = """
        <style>
        __SCREEN_PREVIEW_CSS__
        @media print {
          @page {
            size: A4 portrait;
            margin: 12mm 10mm 14mm 10mm;
          }
          html, body, [data-testid="stAppViewContainer"] {
            background: #ffffff !important;
          }
          [data-testid="stSidebar"], [data-testid="stToolbar"], header, footer {
            display: none !important;
          }
          .stDeployButton {
            display: none !important;
          }
          [data-testid="stMainBlockContainer"] {
            max-width: 100% !important;
            padding: 0 !important;
          }
          h1, h2, h3 {
            page-break-after: avoid !important;
            break-after: avoid-page !important;
          }
          .element-container, .stTable, .plotly-graph-div, div[data-testid="stMetric"] {
            page-break-inside: avoid !important;
            break-inside: avoid-page !important;
          }
          .print-page-break {
            page-break-after: always !important;
            break-after: page !important;
            height: 0 !important;
          }
          div[role="tablist"], button[role="tab"] {
            display: none !important;
          }
          div[role="tabpanel"] {
            display: block !important;
            visibility: visible !important;
          }
          [data-testid="stExpander"] details > summary {
            display: none !important;
          }
          [data-testid="stExpander"] details > div {
            display: block !important;
          }
          __MODE_SPECIFIC_CSS__
        }
        </style>
        """.replace("__SCREEN_PREVIEW_CSS__", screen_preview_css).replace(
        "__MODE_SPECIFIC_CSS__", mode_specific_css
    )

    st.markdown(
        css,
        unsafe_allow_html=True,
    )


def insert_print_break(enabled: bool) -> None:
    if enabled:
        st.markdown('<div class="print-page-break"></div>', unsafe_allow_html=True)


def norm_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(text))
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    normalized = normalized.lower().strip()
    return re.sub(r"[^a-z0-9\\s]", "", normalized)


def format_currency(value: float) -> str:
    if pd.isna(value):
        return "R$ 0"
    return f"R$ {float(value):,.0f}".replace(",", ".")


def format_currency_brl_2(value: float) -> str:
    if pd.isna(value):
        value = 0.0
    formatted = f"{float(value):,.2f}"
    return f"R$ {formatted.replace(',', 'X').replace('.', ',').replace('X', '.')}"


def format_int(value: float) -> str:
    if pd.isna(value):
        return "0"
    return f"{int(round(float(value))):,}".replace(",", ".")


def format_pct(value: float) -> str:
    if pd.isna(value):
        return "0,00%"
    return f"{float(value):.2f}%".replace(".", ",")


def style_bar_labels(fig: go.Figure) -> go.Figure:
    # Prioriza rótulos dentro das barras com bom contraste e evita corte no layout.
    fig.update_traces(
        selector=dict(type="bar"),
        textposition="auto",
        insidetextanchor="middle",
        insidetextfont=dict(color="white", size=12),
        outsidetextfont=dict(color="#334155", size=12),
        cliponaxis=False,
        constraintext="both",
    )
    fig.update_layout(uniformtext_minsize=10, uniformtext_mode="hide")
    fig.update_xaxes(automargin=True)
    fig.update_yaxes(automargin=True)
    return fig


def _inside_bar_label_color(marker_hex: str) -> str:
    """Light fills → dark label; dark fills → white (weekly compare charts)."""
    h = str(marker_hex).lstrip("#")
    if len(h) != 6:
        return "#ffffff"
    try:
        r = int(h[0:2], 16)
        g = int(h[2:4], 16)
        b = int(h[4:6], 16)
    except ValueError:
        return "#ffffff"
    luminance = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0
    return "#0f172a" if luminance > 0.62 else "#ffffff"


def _weekly_compare_textposition(value: float, max_pair: float, *, frac: float = 0.16) -> str:
    """Prefer inside; thin bars vs the pair scale get outside labels."""
    if max_pair <= 0:
        return "inside"
    return "outside" if float(value) < frac * float(max_pair) else "inside"


def _polish_weekly_compare_bar_figure(fig: go.Figure) -> None:
    fig.update_traces(
        selector=dict(type="bar"),
        insidetextanchor="middle",
        cliponaxis=False,
        constraintext="both",
    )
    fig.update_layout(uniformtext_minsize=9, uniformtext_mode="hide")
    fig.update_xaxes(automargin=True)
    fig.update_yaxes(automargin=True)


# Paleta institucional aplicada a todos os gráficos (escala azul UTMB + acentos).
BRAND_COLORWAY = [
    "#0b2447",  # navy core
    "#19376d",  # navy mid
    "#576cbc",  # blue light
    "#a5d7e8",  # ice
    "#f59e0b",  # amber accent
    "#dc2626",  # red alert
    "#16a34a",  # green positive
    "#9333ea",  # purple highlight
]


def apply_brand_chart_style(fig: go.Figure, *, height: int | None = None) -> go.Figure:
    """Aplica paleta, fonte e margens consistentes a qualquer figura plotly."""
    fig.update_layout(
        colorway=BRAND_COLORWAY,
        font=dict(family="Inter, system-ui, sans-serif", color="#1f2937", size=13),
        title_font=dict(family="Inter, system-ui, sans-serif", color="#0b2447", size=16),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        margin=dict(l=8, r=8, t=44, b=8),
        legend=dict(orientation="h", yanchor="bottom", y=-0.18, xanchor="left", x=0.0),
    )
    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        linecolor="#e5e7eb",
        tickcolor="#e5e7eb",
        title_font=dict(color="#475569", size=12),
        tickfont=dict(color="#334155", size=12),
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="#f1f5f9",
        zeroline=False,
        linecolor="#e5e7eb",
        tickcolor="#e5e7eb",
        title_font=dict(color="#475569", size=12),
        tickfont=dict(color="#334155", size=12),
    )
    title_obj = fig.layout.title
    title_text = getattr(title_obj, "text", None) if title_obj is not None else None
    if not title_text:
        fig.update_layout(title=None)
    if height is not None:
        fig.update_layout(height=height)
    return fig


def to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    txt = str(value).strip().lower()
    return txt in {"true", "1", "yes", "y", "sim"}


def is_registered_status_series(series: pd.Series) -> pd.Series:
    txt = series.fillna("").astype(str).str.strip().str.lower()
    registered_values = {"true", "1", "yes", "y", "sim", "completed", "paid", "confirmed", "approved"}
    return txt.isin(registered_values)


def normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    renamed = {}
    for col in df.columns:
        clean = str(col).strip()
        if clean == "yopp [retrieved at]":
            clean = "yopp"
        renamed[col] = clean
    return df.rename(columns=renamed)


def parse_currency_from_edition(edition_id) -> str:
    if pd.isna(edition_id):
        return "UNKNOWN"
    return EDITION_TO_CURRENCY.get(str(edition_id).strip(), "UNKNOWN")


def revenue_split_by_currency(df: pd.DataFrame, value_col: str) -> tuple[float, float, float]:
    if value_col not in df.columns:
        return 0.0, 0.0, 0.0
    total = float(df[value_col].sum())
    if "edition_currency" not in df.columns:
        return total, 0.0, total
    usd_mask = df["edition_currency"].eq("USD")
    usd_value = float(df.loc[usd_mask, value_col].sum())
    brl_value = total - usd_value
    return brl_value, usd_value, total


def compute_additional_revenue(df: pd.DataFrame) -> dict[str, float]:
    if df.empty:
        return {
            "yopp_brl": 0.0,
            "yopp_usd_brl": 0.0,
            "yopp_total": 0.0,
            "bus_ptr17_brl": 0.0,
            "bus_ptr17_usd_brl": 0.0,
            "bus_ptr17_total": 0.0,
            "extras_total": 0.0,
        }

    usd_mask = (
        df["edition_currency"].eq("USD")
        if "edition_currency" in df.columns
        else pd.Series(False, index=df.index)
    )
    brl_mask = ~usd_mask

    yopp_mask = (
        df["yopp_flag"].fillna(0).gt(0)
        if "yopp_flag" in df.columns
        else pd.Series(False, index=df.index)
    )
    yopp_brl = float((yopp_mask & brl_mask).sum()) * YOPP_PRICE_BRL
    yopp_usd_brl = float((yopp_mask & usd_mask).sum()) * YOPP_PRICE_USD * EXCHANGE_RATE_USD_TO_BRL
    yopp_total = yopp_brl + yopp_usd_brl

    ptr17_mask = (
        df["Competition"].fillna("").astype(str).eq("PTR 17")
        if "Competition" in df.columns
        else pd.Series(False, index=df.index)
    )
    bus_ptr17_brl = float((ptr17_mask & brl_mask).sum()) * PTR17_BUS_PRICE_BRL
    bus_ptr17_usd_brl = float((ptr17_mask & usd_mask).sum()) * PTR17_BUS_PRICE_USD * EXCHANGE_RATE_USD_TO_BRL
    bus_ptr17_total = bus_ptr17_brl + bus_ptr17_usd_brl

    return {
        "yopp_brl": yopp_brl,
        "yopp_usd_brl": yopp_usd_brl,
        "yopp_total": yopp_total,
        "bus_ptr17_brl": bus_ptr17_brl,
        "bus_ptr17_usd_brl": bus_ptr17_usd_brl,
        "bus_ptr17_total": bus_ptr17_total,
        "extras_total": yopp_total + bus_ptr17_total,
    }


def standardize_competition(value):
    if pd.isna(value):
        return value
    val = str(value).strip().upper()
    mapping = {
        "UTSB 110": "PTR 108",
        "UTSB 100": "PTR 108",
        "UTSB 100 (108KM)": "PTR 108",
        "PTR 108": "PTR 108",
        "PTR108": "PTR 108",
        "PTR 55 (58KM)": "PTR 58",
        "PTR 55": "PTR 58",
        "PTR55": "PTR 58",
        "PTR 58": "PTR 58",
        "PTR58": "PTR 58",
        "PTR 35 (34KM)": "PTR 34",
        "PTR 35": "PTR 34",
        "PTR35": "PTR 34",
        "PTR 34": "PTR 34",
        "PTR34": "PTR 34",
        "PTR 20 (25KM)": "PTR 25",
        "PTR 20": "PTR 25",
        "PTR20": "PTR 25",
        "PTR 25": "PTR 25",
        "PTR25": "PTR 25",
        "PTR 17": "PTR 17",
        "PTR17": "PTR 17",
        "PTR 17KM": "PTR 17",
        "PTR17KM": "PTR 17",
        "PTR 17 (17KM)": "PTR 17",
        "RUN 7": "RUN 7",
        "RUN7": "RUN 7",
        "FUN 7KM": "RUN 7",
        "FUN 7KM ": "RUN 7",
        "FUN 7": "RUN 7",
        "KIDS": "KIDS",
        "KIDS RACE": "KIDS",
    }
    return mapping.get(val, val)


def standardize_nationality(value):
    if pd.isna(value):
        return value
    val = str(value).strip().upper()
    mapping = {
        "BRASIL": "BR",
        "BRAZIL": "BR",
    }
    return mapping.get(val, val)


def normalize_col_name(value: str) -> str:
    return norm_text(str(value)).replace(" ", "_")


def find_column_by_candidates(columns, candidates: list[str]) -> str | None:
    normalized_lookup = {normalize_col_name(col): col for col in columns}
    for candidate in candidates:
        candidate_norm = normalize_col_name(candidate)
        if candidate_norm in normalized_lookup:
            return normalized_lookup[candidate_norm]
    return None


def canonicalize_team_name(value: str) -> str:
    raw = str(value).strip()
    if not raw or raw.lower() in {"nan", "none"}:
        return ""

    cleaned = norm_text(raw)
    cleaned = re.sub(r"\b(assessoria|corrida|running|team)\b", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        cleaned = norm_text(raw)
    return cleaned


def parse_opt_in_series(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    numeric_flag = numeric.fillna(0).gt(0)
    txt = series.astype(str).str.strip().str.lower()
    text_flag = txt.isin(
        {
            "true",
            "1",
            "yes",
            "y",
            "sim",
            "interessado",
            "interessada",
            "quero",
            "i am interested",
            "interested",
        }
    )
    return (numeric_flag | text_flag).astype(int)


def parse_nubank_series(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    numeric_flag = numeric.fillna(0).gt(0)
    txt = series.astype(str).str.strip().str.lower()

    positive_text = txt.isin(
        {
            "true",
            "1",
            "yes",
            "y",
            "sim",
            "quero",
            "interessado",
            "interessada",
            "yes, i will pay with a nubank card",
            "yes, i will pay with nubank card",
        }
    )
    negative_text = txt.isin(
        {
            "",
            "0",
            "false",
            "no",
            "nao",
            "não",
            "none",
            "nan",
            "n/a",
            "na",
            "-",
        }
    ) | txt.str.contains(r"\b(no|nao|não|not)\b", regex=True)
    nubank_mention = txt.str.contains("nubank", na=False)
    return (numeric_flag | positive_text | (nubank_mention & ~negative_text)).astype(int)


def detect_official_bus_column(columns):
    for col in columns:
        col_norm = normalize_col_name(col)
        has_bus = any(token in col_norm for token in ["bus", "onibus", "transporte", "shuttle"])
        has_intent = any(token in col_norm for token in ["official", "oficial", "interesse", "interested", "opt"])
        if has_bus and has_intent:
            return col
    return None


def detect_nubank_column(columns):
    for col in columns:
        col_norm = normalize_col_name(col)
        if "nubank" in col_norm:
            return col
    return None


def detect_coupon_column(columns, df: pd.DataFrame | None = None):
    preferred = {
        "discount_codes",
        "discount_code",
        "discountcodes",
        "discountcode",
        "coupon_code",
        "coupon_codes",
        "couponcode",
        "couponcodes",
    }
    best_col = None
    best_score = -1.0

    for col in columns:
        col_norm = normalize_col_name(col)
        score = 0.0

        if col_norm in preferred:
            score += 200

        has_coupon_term = any(token in col_norm for token in ["coupon", "cupom", "voucher"])
        has_discount_term = any(token in col_norm for token in ["discount", "desconto"])
        has_code_term = any(token in col_norm for token in ["code", "codigo", "codes", "codigos"])
        has_promo_term = any(token in col_norm for token in ["promo", "promoc"])

        if has_coupon_term:
            score += 120
        if has_discount_term and has_code_term:
            score += 120
        if has_discount_term:
            score += 40
        if has_code_term:
            score += 30
        if has_promo_term and has_code_term:
            score += 15

        # Desempate por conteúdo: prioriza colunas com valores com cara de cupom (3+ chars alfanuméricos).
        if df is not None and col in df.columns:
            series = df[col].fillna("").astype(str).str.strip()
            series = series.replace({"nan": "", "None": "", "none": ""})
            non_empty_ratio = float(series.ne("").mean()) if len(series) else 0.0
            normalized = series.str.upper().str.replace(r"[^A-Z0-9]", "", regex=True)
            coupon_like_ratio = float(normalized.str.len().ge(3).mean()) if len(normalized) else 0.0
            score += non_empty_ratio * 20
            score += coupon_like_ratio * 80

        if score > best_score:
            best_score = score
            best_col = col

    # Evita retornar coluna aleatória sem nenhum indício de cupom.
    return best_col if best_score >= 80 else None


def pick_existing_column(columns, candidates):
    normalized_lookup = {normalize_col_name(col): col for col in columns}
    for candidate in candidates:
        if candidate in columns:
            return candidate
        normalized_candidate = normalize_col_name(candidate)
        if normalized_candidate in normalized_lookup:
            return normalized_lookup[normalized_candidate]
    return None


def safe_series(df: pd.DataFrame, column_name: str | None, default_value="") -> pd.Series:
    if column_name and column_name in df.columns:
        return df[column_name]
    return pd.Series(default_value, index=df.index)


def parse_datetime_series_flexible(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(pd.NaT)

    parsed = pd.to_datetime(series, errors="coerce")
    fallback_mask = parsed.isna() & series.notna()
    if fallback_mask.any():
        parsed.loc[fallback_mask] = pd.to_datetime(series.loc[fallback_mask], errors="coerce", dayfirst=True)

    numeric_guess = pd.to_numeric(series, errors="coerce")
    excel_mask = parsed.isna() & numeric_guess.notna()
    if excel_mask.any():
        parsed.loc[excel_mask] = pd.to_datetime(
            numeric_guess.loc[excel_mask],
            errors="coerce",
            origin="1899-12-30",
            unit="D",
        )
    return parsed


def infer_year_from_filename(file_name: str) -> int | None:
    match = re.search(r"(20\d{2})", str(file_name))
    if not match:
        return None
    year = int(match.group(1))
    if MIN_VALID_YEAR <= year <= MAX_VALID_YEAR:
        return year
    return None


def build_year_series(
    registration_dt: pd.Series,
    date_time_dt: pd.Series,
    source_file_name: str,
) -> pd.Series:
    years = pd.Series(pd.NA, index=registration_dt.index, dtype="Int64")
    reg_year = registration_dt.dt.year.astype("Int64")
    dt_year = date_time_dt.dt.year.astype("Int64")

    years.loc[reg_year.notna()] = reg_year.loc[reg_year.notna()]
    dt_fallback_mask = years.isna() & dt_year.notna()
    years.loc[dt_fallback_mask] = dt_year.loc[dt_fallback_mask]

    inferred_year = infer_year_from_filename(source_file_name)
    if inferred_year is not None:
        years.loc[years.isna()] = inferred_year

    valid_range_mask = years.between(MIN_VALID_YEAR, MAX_VALID_YEAR, inclusive="both")
    years.loc[~valid_range_mask.fillna(False)] = pd.NA
    return years


def female_mask_from_series(series: pd.Series) -> pd.Series:
    normalized = series.fillna("").astype(str).str.strip().str.lower()
    return normalized.isin(AI_FEMALE_TOKENS)


def male_mask_from_series(series: pd.Series) -> pd.Series:
    normalized = series.fillna("").astype(str).str.strip().str.lower()
    return normalized.isin(AI_MALE_TOKENS)


def get_gender_series(df: pd.DataFrame) -> pd.Series:
    gender_col = find_column_by_candidates(df.columns, AI_GENDER_CANDIDATES)
    if not gender_col:
        return pd.Series("", index=df.index)
    return df[gender_col]


def count_female_entries(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    return int(female_mask_from_series(get_gender_series(df)).sum())


def count_male_entries(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    return int(male_mask_from_series(get_gender_series(df)).sum())


def get_mp_approved_scope(df_mp: pd.DataFrame) -> pd.DataFrame:
    if "Estado" not in df_mp.columns:
        return df_mp.copy()
    return df_mp[df_mp["Estado"].astype(str).eq("Aprovado")].copy()


def classify_coupon_prefix(code: str) -> str:
    if pd.isna(code):
        return "SEM_CUPOM"
    clean = re.sub(r"[^A-Za-z0-9]", "", str(code).strip().upper())
    if not clean:
        return "SEM_CUPOM"
    prefix = clean[:3]
    return COUPON_PREFIX_RULES.get(prefix, f"OUTROS ({prefix})")


def _coupon_code_normalized_alnum(df: pd.DataFrame) -> pd.Series:
    if "coupon_code" not in df.columns:
        return pd.Series("", index=df.index)
    s = (
        df["coupon_code"]
        .astype(str)
        .str.strip()
        .str.upper()
        .str.replace(r"[^A-Z0-9]", "", regex=True)
        .replace({"NAN": "", "NONE": ""})
    )
    return s


def coupon_usage_registration_mask(df: pd.DataFrame) -> pd.Series:
    """True quando há cupom válido no mesmo critério do estudo analítico de cupons.

    Código normalizado (somente A–Z/0–9) com comprimento ≥ 3 — ver uso em cupons por
    categoria e `_coupon_code_normalized_alnum`.
    """
    norm = _coupon_code_normalized_alnum(df)
    return norm.str.len().ge(3)


def nubank_card_payment_registration_mask(df: pd.DataFrame) -> pd.Series:
    """Pagamento com cartão Nubank (`nubank_flag` > 0), excluindo quem usou cupom na compra."""
    if "nubank_flag" not in df.columns:
        return pd.Series(False, index=df.index)
    nb = pd.to_numeric(df["nubank_flag"], errors="coerce").fillna(0).gt(0)
    return nb & ~coupon_usage_registration_mask(df)


def sort_competitions(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    local = df.copy()
    local["_order"] = pd.Categorical(local[col_name], categories=PERCURSO_ORDER, ordered=True)
    local = local.sort_values("_order", na_position="last").drop(columns="_order")
    return local


def correct_city(city: str, ibge_df: pd.DataFrame, city_choices: list[str], cutoff: float = 0.8) -> str:
    norm = norm_text(city)
    matches = difflib.get_close_matches(norm, city_choices, n=1, cutoff=cutoff)
    if matches:
        return ibge_df.loc[ibge_df["City_norm"] == matches[0], "City"].iat[0]
    return str(city)


def build_top_cities_table(
    df: pd.DataFrame,
    ibge_df: pd.DataFrame,
    total: int,
    count_col: str,
    pct_col: str,
    top_n: int = 5,
) -> pd.DataFrame:
    city_col = "city" if "city" in df.columns else "city_norm"
    if city_col not in df.columns:
        return pd.DataFrame(columns=["Cidade", count_col, pct_col])

    city_series = (
        df[city_col]
        .astype(str)
        .str.strip()
        .replace({"": pd.NA, "nan": pd.NA, "NaN": pd.NA})
        .dropna()
    )
    if city_series.empty:
        return pd.DataFrame(columns=["Cidade", count_col, pct_col])

    if {"City", "City_norm"}.issubset(ibge_df.columns):
        city_choices = ibge_df["City_norm"].tolist()
        city_series = city_series.apply(lambda city: correct_city(city, ibge_df, city_choices))

    top_cities = (
        city_series.value_counts().head(top_n).rename_axis("Cidade").reset_index(name=count_col)
    )
    top_cities[pct_col] = top_cities[count_col].apply(
        lambda value: format_pct((value / total) * 100 if total else 0)
    )
    return top_cities


@st.cache_data(show_spinner=False)
def load_ibge() -> pd.DataFrame:
    ibge_url = (
        "https://raw.githubusercontent.com/rafaelnmiranda/dash_utmb/"
        "de2e7125c2a3c08c7c41be14c43e528b43c2ea58/municipios_IBGE.xlsx"
    )
    resp = requests.get(ibge_url, timeout=15)
    resp.raise_for_status()
    ibge = pd.read_excel(BytesIO(resp.content), engine="openpyxl")
    ibge["City_norm"] = ibge["City"].astype(str).map(norm_text)
    return ibge.drop_duplicates(subset=["City_norm"], keep="first")


@st.cache_data(show_spinner=False)
def get_app_version_stamp() -> str:
    try:
        output = subprocess.check_output(
            [
                "git",
                "log",
                "-1",
                "--format=%h|%cd",
                "--date=format:%Y-%m-%d %H:%M:%S",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        if output and "|" in output:
            commit_hash, commit_dt = output.split("|", maxsplit=1)
            return f"{commit_hash} | {commit_dt}"
    except Exception:  # noqa: BLE001
        pass

    fallback_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"sem-git | {fallback_ts}"


def validate_columns(df: pd.DataFrame, source_name: str) -> list[str]:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        st.warning(
            f"Arquivo `{source_name}` sem colunas obrigatórias: {', '.join(missing)}. "
            "Essas métricas podem ficar incompletas."
        )
    return missing


def preprocess_uploaded_file(uploaded_file, validate_required: bool = True) -> pd.DataFrame:
    df = pd.read_excel(uploaded_file, sheet_name=0)
    df = normalize_headers(df)
    if validate_required:
        validate_columns(df, uploaded_file.name)

    df["source_file"] = uploaded_file.name
    registration_col = pick_existing_column(df.columns, ["Registration date", "Registration Date"])
    registration_raw = safe_series(df, registration_col, default_value=pd.NA)
    df["Registration date"] = parse_datetime_series_flexible(registration_raw)

    birthdate_col = pick_existing_column(df.columns, ["birthdate", "Birthdate", "Birth date"])
    birthdate_raw = safe_series(df, birthdate_col, default_value=pd.NA)
    df["birthdate"] = parse_datetime_series_flexible(birthdate_raw)

    if "date_time" in df.columns:
        df["date_time_parsed"] = parse_datetime_series_flexible(df["date_time"])
    else:
        df["date_time_parsed"] = pd.Series(pd.NaT, index=df.index)

    df["Ano"] = build_year_series(df["Registration date"], df["date_time_parsed"], uploaded_file.name)
    missing_years = int(df["Ano"].isna().sum())
    if missing_years > 0:
        pct_missing = missing_years / len(df) * 100 if len(df) else 0
        st.warning(
            f"Arquivo `{uploaded_file.name}` com {missing_years} registros sem ano inferido "
            f"({pct_missing:.1f}%)."
        )

    df["sale_hour"] = df["date_time_parsed"].dt.hour
    df["sale_period"] = pd.cut(
        df["sale_hour"],
        bins=[-0.1, 5.9, 11.9, 17.9, 23.9],
        labels=["Madrugada", "Manhã", "Tarde", "Noite"],
    ).astype("object")
    df["sale_period"] = df["sale_period"].fillna("Sem horário")

    edition_col = pick_existing_column(df.columns, ["Edition ID"])
    edition_series = safe_series(df, edition_col, default_value=pd.NA)
    df["edition_currency"] = edition_series.apply(parse_currency_from_edition)
    if (df["edition_currency"] == "UNKNOWN").any():
        if edition_col and edition_col in df.columns:
            unknown_ids = sorted(df.loc[df["edition_currency"] == "UNKNOWN", edition_col].dropna().astype(str).unique())
            st.warning(f"Edition ID sem mapeamento de moeda: {', '.join(unknown_ids)}")

    registered_status_col = pick_existing_column(df.columns, ["Registered status", "Status"])
    # KPI oficial: considerar inscrições ativas. Em algumas bases antigas, o status vem como COMPLETED.
    if registered_status_col:
        status_series = df.get(registered_status_col)
        # Usa parser de status textual para cobrir variações como COMPLETED/PAID/CONFIRMED.
        df["is_registered"] = is_registered_status_series(status_series)
    else:
        df["is_registered"] = False
    competition_col = pick_existing_column(df.columns, ["Competition"])
    df["Competition"] = safe_series(df, competition_col, default_value="").apply(standardize_competition)
    df = df[~df["Competition"].astype(str).str.contains("KIDS", case=False, na=False)].copy()

    nationality_col = pick_existing_column(df.columns, ["nationality", "Nationality"])
    country_col = pick_existing_column(df.columns, ["country", "Country"])
    city_col = pick_existing_column(df.columns, ["city", "City"])
    df["nationality_std"] = safe_series(df, nationality_col, default_value="").apply(standardize_nationality)
    df["country_std"] = safe_series(df, country_col, default_value="").astype(str).str.strip().str.upper()
    df["city_norm"] = safe_series(df, city_col, default_value="").astype(str).map(norm_text)

    if "yopp" in df.columns:
        df["yopp_flag"] = pd.to_numeric(df["yopp"], errors="coerce").fillna(0).astype(int).clip(0, 1)
    else:
        df["yopp_flag"] = 0

    nubank_col = "nubank_opt" if "nubank_opt" in df.columns else detect_nubank_column(df.columns)
    if nubank_col:
        df["nubank_flag"] = parse_nubank_series(df[nubank_col])
    else:
        df["nubank_flag"] = 0

    bus_col = detect_official_bus_column(df.columns)
    if bus_col:
        df["official_bus_flag"] = parse_opt_in_series(df[bus_col])
    else:
        df["official_bus_flag"] = 0

    if "bin_number" in df.columns:
        df["bin_number"] = (
            df["bin_number"]
            .astype(str)
            .str.replace(r"\\D", "", regex=True)
            .str[:6]
            .replace("nan", "")
        )
    else:
        df["bin_number"] = ""

    coupon_col = detect_coupon_column(df.columns, df=df)
    if coupon_col:
        df["coupon_code"] = df[coupon_col].astype(str).str.strip()
        df["coupon_code"] = df["coupon_code"].replace({"nan": "", "None": "", "none": ""})
    else:
        df["coupon_code"] = ""

    df["coupon_prefix"] = (
        df["coupon_code"]
        .astype(str)
        .str.upper()
        .str.replace(r"[^A-Z0-9]", "", regex=True)
        .str[:3]
    )
    df["coupon_prefix"] = df["coupon_prefix"].replace("", "SEM")
    df["coupon_family"] = df["coupon_code"].apply(classify_coupon_prefix)

    registration_col = pick_existing_column(df.columns, ["Registration amount"])
    total_registration_col = pick_existing_column(df.columns, ["Total registration amount", "Registration amount"])
    total_discounts_col = pick_existing_column(df.columns, ["Total discounts amount", "Discounts amount"])

    registration_series = (
        pd.to_numeric(df.get(registration_col), errors="coerce").fillna(0) if registration_col else pd.Series(0, index=df.index)
    )
    total_registration_series = (
        pd.to_numeric(df.get(total_registration_col), errors="coerce").fillna(0)
        if total_registration_col
        else pd.Series(0, index=df.index)
    )
    total_discounts_series = (
        pd.to_numeric(df.get(total_discounts_col), errors="coerce").fillna(0)
        if total_discounts_col
        else pd.Series(0, index=df.index)
    )

    usd_factor = df["edition_currency"].eq("USD").astype(int) * (EXCHANGE_RATE_USD_TO_BRL - 1) + 1
    df["registration_brl"] = registration_series * usd_factor
    df["total_registration_brl"] = total_registration_series * usd_factor
    df["total_discounts_brl"] = total_discounts_series * usd_factor
    df["net_revenue_brl"] = df["total_registration_brl"] - df["total_discounts_brl"]

    today = pd.Timestamp.now().normalize()
    df["age"] = ((today - df["birthdate"]).dt.days / 365.25).round(1)

    return df


def get_filtered_base(df: pd.DataFrame) -> pd.DataFrame:
    base = df[df["is_registered"]].copy()
    return base


def get_email_column_name(df: pd.DataFrame) -> str | None:
    candidates = [
        "Email",
        "E-mail",
        "email",
        "e-mail",
        "mail",
        "athlete_email",
        "participant_email",
    ]
    found = find_column_by_candidates(df.columns, candidates)
    if found:
        return found

    for col in df.columns:
        if "mail" in normalize_col_name(col):
            return col
    return None


def extract_unique_emails(df: pd.DataFrame) -> set[str]:
    email_col = get_email_column_name(df)
    if not email_col:
        return set()

    series = df[email_col].dropna().astype(str).str.strip().str.lower()
    series = series[~series.isin(["", "nan", "none", "null"])]
    series = series[series.str.contains("@", regex=False)]
    return set(series.tolist())


def load_historical_venn_sets(base_dir: str) -> tuple[dict[str, set[str]], list[str]]:
    base_path = Path(base_dir)
    email_sets = {edition: set() for edition in HISTORICAL_VENN_FILES}
    warnings: list[str] = []

    for edition, files in HISTORICAL_VENN_FILES.items():
        for filename in files:
            file_path = base_path / filename
            if not file_path.exists():
                warnings.append(f"Arquivo histórico ausente: {filename}")
                continue
            try:
                hist_df = pd.read_excel(file_path, sheet_name=0)
                hist_df = normalize_headers(hist_df)
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"Falha ao ler {filename}: {exc}")
                continue

            email_col = get_email_column_name(hist_df)
            if not email_col:
                warnings.append(f"Coluna de e-mail não encontrada em {filename}")
                continue

            email_sets[edition].update(extract_unique_emails(hist_df))
    return email_sets, warnings


def compute_membership_distribution(email_sets: dict[str, set[str]]) -> dict[int, int]:
    universe = set().union(*email_sets.values()) if email_sets else set()
    membership_counts = {1: 0, 2: 0, 3: 0, 4: 0}
    for email in universe:
        presence = sum(email in edition_set for edition_set in email_sets.values())
        if presence in membership_counts:
            membership_counts[presence] += 1
    return membership_counts


def render_venn_unique_athletes(df_2026_all_rows: pd.DataFrame) -> None:
    st.header("Venn de Atletas Únicos (2023-2026)")

    hist_sets, issues = load_historical_venn_sets(str(Path(__file__).resolve().parent))
    for msg in issues:
        st.warning(msg)

    email_2026 = extract_unique_emails(df_2026_all_rows)
    if not email_2026:
        st.info("Sem e-mails válidos no upload 2026 para montar o Venn.")
        return

    venn_sets = {
        "2023": hist_sets.get("2023", set()),
        "2024": hist_sets.get("2024", set()),
        "2025": hist_sets.get("2025", set()),
        "2026": email_2026,
    }
    totals = {edition: len(values) for edition, values in venn_sets.items()}
    membership = compute_membership_distribution(venn_sets)
    previous_editions = venn_sets["2023"] | venn_sets["2024"] | venn_sets["2025"]
    returning_2026 = len(venn_sets["2026"] & previous_editions)
    return_rate_2026 = (returning_2026 / totals["2026"] * 100) if totals["2026"] else 0
    new_2026 = totals["2026"] - returning_2026

    editions = ["2023", "2024", "2025", "2026"]
    universe = set().union(*venn_sets.values())
    total_unique_general = len(universe)

    t1, t2, t3, t4, t5 = st.columns(5)
    t1.metric("Total únicos 2023", format_int(totals["2023"]))
    t2.metric("Total únicos 2024", format_int(totals["2024"]))
    t3.metric("Total únicos 2025", format_int(totals["2025"]))
    t4.metric("Total únicos 2026", format_int(totals["2026"]))
    t5.metric("Total únicos geral", format_int(total_unique_general))

    combo_counts: dict[tuple[bool, bool, bool, bool], int] = {}
    for email in universe:
        key = tuple(email in venn_sets[edition] for edition in editions)
        combo_counts[key] = combo_counts.get(key, 0) + 1

    venn_table_rows = []
    for key, count in combo_counts.items():
        present = [edition for edition, flag in zip(editions, key) if flag]
        if not present:
            continue
        venn_table_rows.append(
            {
                "Participação": " + ".join(present),
                "Qtde atletas únicos": count,
                "N edições": len(present),
            }
        )
    venn_table = pd.DataFrame(venn_table_rows).sort_values(
        ["N edições", "Qtde atletas únicos", "Participação"],
        ascending=[False, False, True],
    )
    venn_table["Qtde atletas únicos"] = venn_table["Qtde atletas únicos"].map(format_int)
    st.subheader("Tabela Venn (interseções exatas por e-mail)")
    st.dataframe(venn_table[["Participação", "Qtde atletas únicos"]], hide_index=True, use_container_width=True)

    k1, k2 = st.columns(2)
    k1.metric("Taxa de retorno 2026", format_pct(return_rate_2026))
    k2.metric("Atletas retornantes 2026", format_int(returning_2026), delta=f"Novos: {format_int(new_2026)}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Só em 1 edição", format_int(membership[1]))
    c2.metric("Em 2 edições", format_int(membership[2]))
    c3.metric("Em 3 edições", format_int(membership[3]))
    c4.metric("Em 4 edições", format_int(membership[4]))


def render_header(kpi_df: pd.DataFrame, data_base_label: str) -> None:
    total = len(kpi_df)
    unique_emails = len(extract_unique_emails(kpi_df))
    female = count_female_entries(kpi_df)
    pct_female = (female / total * 100) if total else 0
    foreigners = kpi_df[kpi_df["nationality_std"] != "BR"].shape[0] if "nationality_std" in kpi_df.columns else 0
    pct_foreigners = (foreigners / total * 100) if total else 0
    countries = kpi_df["country_std"].replace({"": pd.NA, "NAN": pd.NA}).dropna().nunique()
    net_revenue = kpi_df["net_revenue_brl"].sum()
    avg_ticket = (net_revenue / total) if total else 0
    br_mask = kpi_df["nationality_std"].eq("BR")
    foreign_mask = ~br_mask
    yopp_total = int(kpi_df["yopp_flag"].sum()) if "yopp_flag" in kpi_df.columns else 0
    yopp_br = int(kpi_df.loc[br_mask, "yopp_flag"].sum()) if "yopp_flag" in kpi_df.columns else 0
    yopp_foreign = int(kpi_df.loc[foreign_mask, "yopp_flag"].sum()) if "yopp_flag" in kpi_df.columns else 0
    nubank_total = int(kpi_df["nubank_flag"].sum()) if "nubank_flag" in kpi_df.columns else 0
    bus_total = int(kpi_df["official_bus_flag"].sum()) if "official_bus_flag" in kpi_df.columns else 0
    bus_br = int(kpi_df.loc[br_mask, "official_bus_flag"].sum()) if "official_bus_flag" in kpi_df.columns else 0
    bus_foreign = int(kpi_df.loc[foreign_mask, "official_bus_flag"].sum()) if "official_bus_flag" in kpi_df.columns else 0

    st.markdown(
        f"""
        <div class="hero">
          <h2>Dashboard de Inscrições 2026 - Paraty Brazil by UTMB</h2>
          <p class="subtle">Base atualizada até {data_base_label} | Conversão fixa: 1 USD = R$ 5,00</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4, c5, c6, c7, c8 = st.columns(8)
    c1.metric("Inscritos ativos", format_int(total))
    c2.metric("% mulheres", format_pct(pct_female))
    c3.metric("% estrangeiros", format_pct(pct_foreigners))
    c4.metric("Países distintos", format_int(countries))
    c5.metric("Receita líquida", format_currency(net_revenue))
    c6.metric("Ticket médio", format_currency(avg_ticket))
    c7.metric("Pagaram com cartão Nubank", format_int(nubank_total))
    c8.metric("E-mails únicos", format_int(unique_emails))

    YOPP_META_OCULOS = 300
    pct_venda_yopp = (yopp_total / YOPP_META_OCULOS * 100) if YOPP_META_OCULOS else 0
    pct_atletas_yopp = (yopp_total / total * 100) if total else 0
    y1, y2 = st.columns(2)
    y1.metric("Óculos Yopp vendidos", format_int(yopp_total))
    y1.caption(
        f"% meta venda ({YOPP_META_OCULOS} óculos): {format_pct(pct_venda_yopp)} | "
        f"% atletas que compram: {format_pct(pct_atletas_yopp)} | Atletas totais: {format_int(total)}"
    )
    y2.metric("Interessados no ônibus oficial", format_int(bus_total))
    y2.caption(f"BR: {format_int(bus_br)} | Estrangeiros: {format_int(bus_foreign)}")


PT_WEEKDAYS = [
    "Segunda-feira",
    "Terça-feira",
    "Quarta-feira",
    "Quinta-feira",
    "Sexta-feira",
    "Sábado",
    "Domingo",
]


def _resolve_reference_date(df: pd.DataFrame, ref_ts: pd.Timestamp | None) -> date:
    """Determina a data de referência (preferindo o ts informado, depois max do df)."""
    if ref_ts is not None and pd.notna(ref_ts):
        return pd.Timestamp(ref_ts).date()
    if "Registration date" in df.columns:
        max_ts = pd.to_datetime(df["Registration date"], errors="coerce").max()
        if pd.notna(max_ts):
            return max_ts.date()
    return date.today()


def _filter_by_day(df: pd.DataFrame, day: date) -> pd.DataFrame:
    if "Registration date" not in df.columns:
        return df.iloc[0:0]
    series = pd.to_datetime(df["Registration date"], errors="coerce").dt.date
    return df.loc[series.eq(day)].copy()


def _filter_by_window(df: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    """Inclusivo nas duas pontas."""
    if "Registration date" not in df.columns:
        return df.iloc[0:0]
    series = pd.to_datetime(df["Registration date"], errors="coerce").dt.date
    return df.loc[series.between(start, end)].copy()


def compute_daily_deltas(df: pd.DataFrame, ref_ts: pd.Timestamp | None = None) -> dict[str, float | int | str]:
    """Compara o dia de referência (geralmente data base do dashboard) com o dia anterior."""
    ref_day = _resolve_reference_date(df, ref_ts)
    prev_day = ref_day - pd.Timedelta(days=1).to_pytimedelta()

    today_df = _filter_by_day(df, ref_day)
    yesterday_df = _filter_by_day(df, prev_day)

    inscritos_hoje = int(len(today_df))
    inscritos_ontem = int(len(yesterday_df))
    delta_inscritos = inscritos_hoje - inscritos_ontem

    receita_hoje = float(today_df["net_revenue_brl"].sum()) if "net_revenue_brl" in today_df.columns else 0.0
    receita_ontem = float(yesterday_df["net_revenue_brl"].sum()) if "net_revenue_brl" in yesterday_df.columns else 0.0
    delta_receita = receita_hoje - receita_ontem

    weekday_idx = ref_day.weekday()
    weekday_label = PT_WEEKDAYS[weekday_idx]

    return {
        "ref_day": ref_day,
        "prev_day": prev_day,
        "inscritos_hoje": inscritos_hoje,
        "inscritos_ontem": inscritos_ontem,
        "delta_inscritos": delta_inscritos,
        "receita_hoje": receita_hoje,
        "receita_ontem": receita_ontem,
        "delta_receita": delta_receita,
        "weekday_label": weekday_label,
    }


def compute_weekly_deltas(df: pd.DataFrame, ref_ts: pd.Timestamp | None = None) -> dict[str, object]:
    """Compara semana corrente (segunda→domingo da ref) vs semana anterior."""
    ref_day = _resolve_reference_date(df, ref_ts)
    week_start = ref_day - pd.Timedelta(days=ref_day.weekday()).to_pytimedelta()
    week_end = week_start + pd.Timedelta(days=6).to_pytimedelta()
    prev_week_start = week_start - pd.Timedelta(days=7).to_pytimedelta()
    prev_week_end = week_end - pd.Timedelta(days=7).to_pytimedelta()
    iso_year, iso_week, _ = ref_day.isocalendar()

    week_df = _filter_by_window(df, week_start, week_end)
    prev_week_df = _filter_by_window(df, prev_week_start, prev_week_end)

    inscritos_semana = int(len(week_df))
    inscritos_semana_anterior = int(len(prev_week_df))
    delta_inscritos_pct = (
        ((inscritos_semana - inscritos_semana_anterior) / inscritos_semana_anterior * 100)
        if inscritos_semana_anterior
        else 0.0
    )

    receita_semana = float(week_df["net_revenue_brl"].sum()) if "net_revenue_brl" in week_df.columns else 0.0
    receita_semana_anterior = (
        float(prev_week_df["net_revenue_brl"].sum()) if "net_revenue_brl" in prev_week_df.columns else 0.0
    )
    delta_receita_pct = (
        ((receita_semana - receita_semana_anterior) / receita_semana_anterior * 100)
        if receita_semana_anterior
        else 0.0
    )

    melhor_dia: dict[str, object] | None = None
    pior_dia: dict[str, object] | None = None
    if not week_df.empty and "Registration date" in week_df.columns:
        per_day = (
            week_df.assign(_day=pd.to_datetime(week_df["Registration date"], errors="coerce").dt.date)
            .dropna(subset=["_day"])
            .groupby("_day")
            .size()
            .rename("inscritos")
            .reset_index()
            .sort_values("inscritos", ascending=False)
        )
        if not per_day.empty:
            top = per_day.iloc[0]
            bottom = per_day.iloc[-1]
            melhor_dia = {
                "data": top["_day"],
                "inscritos": int(top["inscritos"]),
                "weekday": PT_WEEKDAYS[top["_day"].weekday()],
            }
            pior_dia = {
                "data": bottom["_day"],
                "inscritos": int(bottom["inscritos"]),
                "weekday": PT_WEEKDAYS[bottom["_day"].weekday()],
            }

    return {
        "ref_day": ref_day,
        "week_start": week_start,
        "week_end": week_end,
        "prev_week_start": prev_week_start,
        "prev_week_end": prev_week_end,
        "iso_year": iso_year,
        "iso_week": iso_week,
        "inscritos_semana": inscritos_semana,
        "inscritos_semana_anterior": inscritos_semana_anterior,
        "delta_inscritos_pct": delta_inscritos_pct,
        "receita_semana": receita_semana,
        "receita_semana_anterior": receita_semana_anterior,
        "delta_receita_pct": delta_receita_pct,
        "melhor_dia": melhor_dia,
        "pior_dia": pior_dia,
        "week_df": week_df,
        "prev_week_df": prev_week_df,
    }


def compute_marketing_highlights(
    df: pd.DataFrame,
    granularidade: str,
    ref_ts: pd.Timestamp | None = None,
    targets: dict[str, int] | None = None,
) -> list[dict[str, str]]:
    """Gera destaques automáticos: percurso em alta, cupom em alta, marco de meta.

    `granularidade` ∈ {"diario", "semanal"} controla a janela de comparação.
    """
    targets = targets or {}
    ref_day = _resolve_reference_date(df, ref_ts)
    if granularidade == "diario":
        current_start = current_end = ref_day
        prev_start = prev_end = ref_day - pd.Timedelta(days=1).to_pytimedelta()
        janela_label = "24h"
    else:
        current_start = ref_day - pd.Timedelta(days=ref_day.weekday()).to_pytimedelta()
        current_end = current_start + pd.Timedelta(days=6).to_pytimedelta()
        prev_start = current_start - pd.Timedelta(days=7).to_pytimedelta()
        prev_end = current_end - pd.Timedelta(days=7).to_pytimedelta()
        janela_label = "semana"

    current_df = _filter_by_window(df, current_start, current_end)
    prev_df = _filter_by_window(df, prev_start, prev_end)

    highlights: list[dict[str, str]] = []

    # Destaque 1 - percurso que mais cresceu na janela.
    if "Competition" in current_df.columns:
        cur_counts = current_df["Competition"].value_counts()
        prev_counts = prev_df["Competition"].value_counts() if "Competition" in prev_df.columns else pd.Series(dtype=int)
        crescimento = (cur_counts - prev_counts.reindex(cur_counts.index, fill_value=0)).sort_values(ascending=False)
        if not crescimento.empty and crescimento.iloc[0] > 0:
            top_percurso = crescimento.index[0]
            highlights.append(
                {
                    "titulo": f"Percurso em alta ({janela_label})",
                    "valor": f"{top_percurso}",
                    "hint": f"+{int(crescimento.iloc[0])} inscritos vs período anterior",
                }
            )

    # Destaque 2 - cupom mais usado na janela atual (>= 3 chars válidos).
    if "coupon_code" in current_df.columns and not current_df.empty:
        codes = current_df["coupon_code"].fillna("").astype(str).str.strip()
        codes = codes[codes.str.len().ge(3)]
        if not codes.empty:
            top_coupon = codes.value_counts().head(1)
            cupom_nome = str(top_coupon.index[0])
            usos = int(top_coupon.iloc[0])
            highlights.append(
                {
                    "titulo": f"Cupom em destaque ({janela_label})",
                    "valor": cupom_nome,
                    "hint": f"{usos} uso(s) na janela",
                }
            )

    # Destaque 3 - percurso mais próximo / acima de 80% da meta acumulada.
    if targets and "Competition" in df.columns:
        accumulated = df["Competition"].value_counts()
        progresso = []
        for percurso, meta in targets.items():
            if not meta:
                continue
            total = int(accumulated.get(percurso, 0))
            pct = total / meta * 100
            progresso.append((percurso, total, meta, pct))
        progresso.sort(key=lambda row: row[3], reverse=True)
        if progresso:
            percurso, total, meta, pct = progresso[0]
            highlights.append(
                {
                    "titulo": "Marco de meta",
                    "valor": f"{percurso} em {pct:.0f}% da meta",
                    "hint": f"{format_int(total)} de {format_int(meta)} inscritos",
                }
            )

    return highlights[:3]


def render_kpi_strip(items: list[dict[str, object]], *, columns: int | None = None) -> None:
    """Faixa padronizada de st.metric. Cada item: {label, value, delta?, help?}."""
    if not items:
        return
    n = columns or len(items)
    cols = st.columns(n)
    for idx, item in enumerate(items):
        col = cols[idx % n]
        col.metric(
            label=str(item.get("label", "")),
            value=str(item.get("value", "")),
            delta=item.get("delta"),
            help=item.get("help"),
        )


def inject_marketing_report_styles() -> None:
    """CSS compartilhado apenas nas visões Marketing Diário / Semanal (cards KPI responsivos)."""
    st.markdown(
        """
        <style>
        :root {
          --mkt-navy: #0b2447;
          --mkt-navy-mid: #19376d;
          --mkt-accent: #576cbc;
          --mkt-surface: #ffffff;
          --mkt-border: #e2e8f0;
          --mkt-muted: #64748b;
          --mkt-text: #1e293b;
        }
        .mkt-kpi-wrap {
          display: flex;
          flex-wrap: wrap;
          gap: 12px;
          margin: 8px 0 18px 0;
          align-items: stretch;
        }
        .mkt-kpi-card {
          flex: 1 1 148px;
          min-width: min(168px, 100%);
          max-width: 100%;
          box-sizing: border-box;
          background: var(--mkt-surface);
          border: 1px solid var(--mkt-border);
          border-radius: 12px;
          padding: 14px 16px;
          border-left: 4px solid var(--mkt-navy);
          box-shadow: 0 1px 3px rgba(15, 23, 42, 0.06);
        }
        .mkt-kpi-label {
          font-family: Inter, system-ui, sans-serif;
          font-size: 0.78rem;
          font-weight: 600;
          color: var(--mkt-muted);
          text-transform: uppercase;
          letter-spacing: 0.04em;
          line-height: 1.35;
          word-wrap: break-word;
          overflow-wrap: anywhere;
          white-space: normal;
          hyphens: auto;
        }
        .mkt-kpi-value {
          font-family: Inter, system-ui, sans-serif;
          font-size: 1.32rem;
          font-weight: 700;
          color: var(--mkt-text);
          margin-top: 6px;
          line-height: 1.28;
          word-wrap: break-word;
          overflow-wrap: anywhere;
          white-space: normal;
        }
        .mkt-kpi-delta {
          font-family: Inter, system-ui, sans-serif;
          font-size: 0.82rem;
          color: #475569;
          margin-top: 6px;
          line-height: 1.35;
          word-wrap: break-word;
          overflow-wrap: anywhere;
          white-space: normal;
        }
        .mkt-section-title {
          font-family: Inter, system-ui, sans-serif;
          font-size: 1.05rem;
          font-weight: 700;
          color: var(--mkt-navy);
          margin: 1.1rem 0 0.4rem 0;
          letter-spacing: -0.01em;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_marketing_kpi_cards(items: list[dict[str, object]]) -> None:
    """KPIs em cartões flexíveis (sem truncagem agressiva de rótulos do st.metric)."""
    if not items:
        return
    parts: list[str] = ['<div class="mkt-kpi-wrap">']
    for item in items:
        label = html.escape(str(item.get("label", "")))
        value = html.escape(str(item.get("value", "")))
        help_txt = item.get("help")
        title_attr = f' title="{html.escape(str(help_txt))}"' if help_txt else ""
        parts.append(f'<div class="mkt-kpi-card"{title_attr}>')
        parts.append(f'<div class="mkt-kpi-label">{label}</div>')
        parts.append(f'<div class="mkt-kpi-value">{value}</div>')
        delta = item.get("delta")
        if delta is not None:
            parts.append(f'<div class="mkt-kpi-delta">{html.escape(str(delta))}</div>')
        parts.append("</div>")
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def _route_pct_meta_color(pct: float) -> str:
    if pct >= 100:
        return BRAND_COLORWAY[6]
    if pct >= 70:
        return BRAND_COLORWAY[4]
    return BRAND_COLORWAY[2]


def build_weekly_metas_bullet_figure(summary: pd.DataFrame) -> go.Figure:
    """Barras horizontais sobrepostas: meta (fundo) vs inscritos (frente), por percurso."""
    order_with_total = list(PERCURSO_ORDER) + ["TOTAL"]
    chart_df = summary.copy()
    chart_df["Percurso"] = pd.Categorical(
        chart_df["Percurso"],
        categories=order_with_total,
        ordered=True,
    )
    chart_df = chart_df.sort_values("Percurso").reset_index(drop=True)
    colors = [_route_pct_meta_color(float(r["pct_meta"])) for _, r in chart_df.iterrows()]
    xmax_raw = max(float(chart_df["meta"].max()), float(chart_df["inscritos"].max()), 1.0)
    # Extra horizontal room so a single outside inscritos label ("n (x,x%)") never grazes the axis edge.
    xmax = xmax_raw * 1.32

    def _inscritos_bar_label(row: pd.Series) -> str:
        """Um único rótulo fora da barra azul; contagem e % também no hover."""
        ins = float(row["inscritos"])
        pct = float(row["pct_meta"])
        return f"{format_int(ins)} ({format_pct(pct)})"

    ins_labels = [_inscritos_bar_label(r) for _, r in chart_df.iterrows()]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=chart_df["Percurso"],
            x=chart_df["meta"],
            orientation="h",
            name="Meta",
            marker_color="rgba(226, 232, 240, 0.92)",
            text=[format_int(m) for m in chart_df["meta"]],
            textposition="inside",
            # Âncora à esquerda do cinza: não compete com o rótulo externo no fim da barra azul.
            insidetextanchor="start",
            insidetextfont=dict(color="#64748b", size=10),
            constraintext="both",
            hovertemplate="%{y}<br>Meta: %{x}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            y=chart_df["Percurso"],
            x=chart_df["inscritos"],
            orientation="h",
            name="Inscritos",
            marker_color=colors,
            text=ins_labels,
            textposition="outside",
            cliponaxis=False,
            outsidetextfont=dict(color="#334155", size=10),
            constraintext="both",
            customdata=chart_df["pct_meta"],
            hovertemplate="%{y}<br>Inscritos: %{x}<br>% da meta: %{customdata:.1f}%<extra></extra>",
        )
    )
    fig.update_layout(
        title="Metas por percurso — inscritos vs meta",
        barmode="overlay",
        height=max(340, 52 * len(chart_df)),
        uniformtext_minsize=9,
        uniformtext_mode="hide",
    )
    fig.update_xaxes(title="Quantidade", range=[0, xmax])
    fig.update_yaxes(autorange="reversed", title="")
    apply_brand_chart_style(fig)
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255,255,255,0.92)",
        ),
        margin=dict(l=16, r=56, t=56, b=48),
    )
    return fig


def build_weekly_ma_projection_figure(
    series: pd.DataFrame,
    cumulative_inscritos: int,
    end_date: date,
    *,
    ma_specs: tuple[tuple[str, str, str], ...] = (
        ("mm7", "MM 7d", BRAND_COLORWAY[0]),
        ("mm15", "MM 15d", BRAND_COLORWAY[1]),
        ("mm30", "MM 30d", BRAND_COLORWAY[6]),
    ),
) -> tuple[go.Figure, dict[str, int]]:
    """Duplo painel: barras diárias + médias móveis com legenda contendo projeção linear ao fim do período."""
    last_day = series["day"].iloc[-1]
    days_left = max((end_date - last_day).days, 0)
    projections: dict[str, int] = {}

    fig = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.44, 0.56],
        vertical_spacing=0.20,
        subplot_titles=(
            "Inscrições por dia",
            "Médias móveis (escala própria — ver legenda para projeção ao fim da campanha)",
        ),
    )
    fig.add_trace(
        go.Bar(
            x=series["day"],
            y=series["inscricoes_diarias"],
            name="Inscrições/dia",
            marker_color=BRAND_COLORWAY[3],
            opacity=0.85,
        ),
        row=1,
        col=1,
    )

    for col_name, label_base, color in ma_specs:
        last_rate = series[col_name].iloc[-1]
        if pd.isna(last_rate):
            last_rate = 0.0
        proj_total = int(round(cumulative_inscritos + float(last_rate) * days_left))
        projections[col_name] = proj_total
        legend_label = f"{label_base} → projeção fim: {format_int(proj_total)}"
        fig.add_trace(
            go.Scatter(
                x=series["day"],
                y=series[col_name],
                name=legend_label,
                mode="lines+markers",
                line=dict(color=color, width=2.6),
                marker=dict(size=5),
                hovertemplate=(
                    f"{label_base}<br>"
                    + "Data: %{x}<br>"
                    + "Média: %{y:.2f}/dia<br>"
                    + f"Dias restantes (até fim campanha): {days_left}<br>"
                    + f"Projeção total inscritos: {format_int(proj_total)}"
                    + "<extra></extra>"
                ),
            ),
            row=2,
            col=1,
        )

    fig.update_xaxes(matches="x")
    fig.update_layout(
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=-0.14, xanchor="center", x=0.5),
        margin=dict(t=12, b=108, l=10, r=10),
    )
    fig.update_yaxes(title="Inscrições", row=1, col=1)
    fig.update_yaxes(title="Média móvel (por dia)", row=2, col=1)
    apply_brand_chart_style(fig)
    fig.update_layout(
        title=None,
        margin=dict(t=64, b=118, l=10, r=10),
        legend=dict(orientation="h", yanchor="bottom", y=-0.12, xanchor="center", x=0.5),
    )
    return fig, projections


def build_marketing_gauges_grid_figure(df: pd.DataFrame, targets: dict[str, int]) -> go.Figure:
    """Barras horizontais compactas (meta vs inscritos), TOTAL + percursos + Óculos Yopp — Marketing Semanal."""
    summary = build_route_summary(df, targets)
    gauge_order = ["TOTAL"] + list(PERCURSO_ORDER)
    gauge_df = summary.copy()
    gauge_df["Percurso"] = pd.Categorical(gauge_df["Percurso"], categories=gauge_order, ordered=True)
    gauge_df = gauge_df.sort_values("Percurso").reset_index(drop=True)

    rows_list: list[dict[str, object]] = []
    for _, row in gauge_df.iterrows():
        rows_list.append(
            {
                "Categoria": str(row["Percurso"]),
                "value": float(row["inscritos"]),
                "meta": float(row["meta"]),
                "pct": float(row["pct_meta"]),
            }
        )

    yopp_total = float(int(df["yopp_flag"].sum())) if "yopp_flag" in df.columns else 0.0
    yopp_meta = 300.0
    rows_list.append(
        {
            "Categoria": "Óculos Yopp",
            "value": yopp_total,
            "meta": yopp_meta,
            "pct": (yopp_total / yopp_meta * 100) if yopp_meta else 0.0,
        }
    )

    chart_df = pd.DataFrame(rows_list)
    xmax = max(float(chart_df["meta"].max()), float(chart_df["value"].max()), 1.0) * 1.22
    colors = [_route_pct_meta_color(float(p)) for p in chart_df["pct"]]

    def _row_outside_label(row: pd.Series, x_max: float) -> str:
        ins = float(row["value"])
        meta_v = float(row["meta"])
        pct = float(row["pct"])
        near_goal = meta_v > 0 and ins >= meta_v * 0.88
        thin = x_max > 0 and ins / x_max < 0.06
        if near_goal or thin:
            return format_pct(pct)
        return f"{format_int(ins)} · {format_pct(pct)}"

    ins_labels = [_row_outside_label(r, xmax) for _, r in chart_df.iterrows()]
    meta_x = chart_df["meta"].clip(lower=0)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=chart_df["Categoria"],
            x=meta_x,
            orientation="h",
            name="Meta",
            marker_color="rgba(226, 232, 240, 0.92)",
            text=[format_int(m) for m in meta_x],
            textposition="inside",
            insidetextanchor="end",
            insidetextfont=dict(color="#64748b", size=10),
            hovertemplate="%{y}<br>Meta: %{x:,.0f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            y=chart_df["Categoria"],
            x=chart_df["value"],
            orientation="h",
            name="Inscritos / vendidos",
            marker_color=colors,
            text=ins_labels,
            textposition="outside",
            cliponaxis=False,
            outsidetextfont=dict(color="#334155", size=11),
            customdata=chart_df["pct"],
            hovertemplate=(
                "%{y}<br>Valor: %{x:,.0f}<br>% da meta: %{customdata:.1f}%<extra></extra>"
            ),
        )
    )
    n_rows = len(chart_df)
    fig.update_layout(
        title="Metas — desempenho vs meta (TOTAL, percursos e Yopp)",
        barmode="overlay",
        height=max(280, 44 * n_rows),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255,255,255,0.92)",
        ),
        margin=dict(l=14, r=120, t=56, b=48),
    )
    fig.update_xaxes(title="Quantidade", range=[0, xmax])
    fig.update_yaxes(autorange="reversed", title="", tickfont=dict(size=12))
    apply_brand_chart_style(fig)
    return fig


def render_section_caption(text: str) -> None:
    """Microcopy padrão abaixo de st.header explicando o porquê do bloco."""
    st.caption(text)


def render_header_marketing(kpi_df: pd.DataFrame, data_base_label: str) -> None:
    total = len(kpi_df)
    female = count_female_entries(kpi_df)
    pct_female = (female / total * 100) if total else 0
    foreigners = kpi_df[kpi_df["nationality_std"] != "BR"].shape[0] if "nationality_std" in kpi_df.columns else 0
    pct_foreigners = (foreigners / total * 100) if total else 0
    countries = kpi_df["country_std"].replace({"": pd.NA, "NAN": pd.NA}).dropna().nunique()
    br_mask = kpi_df["nationality_std"].eq("BR")
    foreign_mask = ~br_mask
    yopp_total = int(kpi_df["yopp_flag"].sum()) if "yopp_flag" in kpi_df.columns else 0
    bus_total = int(kpi_df["official_bus_flag"].sum()) if "official_bus_flag" in kpi_df.columns else 0
    bus_br = int(kpi_df.loc[br_mask, "official_bus_flag"].sum()) if "official_bus_flag" in kpi_df.columns else 0
    bus_foreign = int(kpi_df.loc[foreign_mask, "official_bus_flag"].sum()) if "official_bus_flag" in kpi_df.columns else 0

    st.markdown(
        f"""
        <div class="hero">
          <h2>Dashboard Marketing 2026 - Paraty Brazil by UTMB</h2>
          <p class="subtle">Base atualizada até {data_base_label} | Visão de audiência, perfil e ritmo</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Inscritos ativos", format_int(total))
    c2.metric("% mulheres", format_pct(pct_female))
    c3.metric("% estrangeiros", format_pct(pct_foreigners))
    c4.metric("Países distintos", format_int(countries))

    yopp_pct = (yopp_total / total * 100) if total else 0
    bus_pct = (bus_total / total * 100) if total else 0
    y1, y2 = st.columns(2)
    y1.metric("Óculos Yopp vendidos", format_int(yopp_total))
    y1.caption(f"% dos inscritos: {format_pct(yopp_pct)}")
    y2.metric("Interessados no ônibus oficial", format_int(bus_total))
    y2.caption(
        f"% dos inscritos: {format_pct(bus_pct)} | BR: {format_int(bus_br)} | Estrangeiros: {format_int(bus_foreign)}"
    )


def _render_report_hero(badge: str, badge_color: str, title: str, subtitle: str) -> None:
    """Hero unificado para Diário/Semanal com badge de tipo e subtítulo padronizado."""
    st.markdown(
        f"""
        <div class="hero hero-report">
          <div class="hero-row">
            <span class="hero-badge" style="background:{badge_color};">{badge}</span>
            <div class="hero-text">
              <h2 style="margin:0;">{title}</h2>
              <p class="subtle" style="color:#cbd5e1;margin:4px 0 0 0;">{subtitle}</p>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_pdf_button(label: str, key: str) -> None:
    """Botão dedicado de export para PDF (usa janela de impressão do navegador)."""
    if st.button(label, key=key, use_container_width=True):
        components.html("<script>window.print();</script>", height=0, width=0)


def _build_pace_chart(df: pd.DataFrame, days_window: int = 14, ma_window: int = 7) -> go.Figure | None:
    """Linha + média móvel para os últimos N dias de inscrições. None se não houver datas."""
    if "Registration date" not in df.columns:
        return None
    series = pd.to_datetime(df["Registration date"], errors="coerce")
    valid_df = df.assign(_day=series.dt.date).dropna(subset=["_day"])
    if valid_df.empty:
        return None

    max_day = max(valid_df["_day"])
    start_day = max_day - pd.Timedelta(days=days_window - 1).to_pytimedelta()
    daily = (
        valid_df[valid_df["_day"].between(start_day, max_day)]
        .groupby("_day")
        .size()
        .rename("inscritos")
        .reset_index()
        .sort_values("_day")
    )

    full_index = pd.date_range(start=start_day, end=max_day, freq="D").date
    daily_full = (
        pd.DataFrame({"_day": full_index})
        .merge(daily, on="_day", how="left")
        .fillna({"inscritos": 0})
    )
    daily_full["mm"] = daily_full["inscritos"].rolling(window=ma_window, min_periods=1).mean()

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=daily_full["_day"],
            y=daily_full["inscritos"],
            name="Inscritos/dia",
            marker_color=BRAND_COLORWAY[2],
        )
    )
    fig.add_trace(
        go.Scatter(
            x=daily_full["_day"],
            y=daily_full["mm"],
            name=f"Média móvel ({ma_window}d)",
            mode="lines+markers",
            line=dict(color=BRAND_COLORWAY[0], width=3),
            marker=dict(size=6),
        )
    )
    fig.update_layout(title=f"Ritmo de vendas — últimos {days_window} dias", height=240)
    apply_brand_chart_style(fig)
    return fig


def _build_route_progress_bars(summary: pd.DataFrame) -> go.Figure:
    """Barras horizontais com % de meta por percurso (sem TOTAL)."""
    chart_df = summary[summary["Percurso"] != "TOTAL"].copy()
    chart_df["pct_cap"] = chart_df["pct_meta"].clip(lower=0, upper=120)
    chart_df["color"] = chart_df["pct_meta"].apply(
        lambda pct: BRAND_COLORWAY[6] if pct >= 100 else (BRAND_COLORWAY[4] if pct >= 70 else BRAND_COLORWAY[2])
    )
    chart_df["label"] = chart_df.apply(
        lambda row: f"{format_int(row['inscritos'])} / {format_int(row['meta'])}  ({row['pct_meta']:.0f}%)",
        axis=1,
    )

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=chart_df["pct_cap"],
            y=chart_df["Percurso"],
            orientation="h",
            text=chart_df["label"],
            textposition="outside",
            marker=dict(color=chart_df["color"]),
        )
    )
    fig.add_vline(x=100, line_color="#94a3b8", line_dash="dash", line_width=1)
    fig.update_layout(
        title="Progresso por percurso (% da meta)",
        xaxis=dict(range=[0, 130], ticksuffix="%"),
        yaxis=dict(autorange="reversed"),
        height=320,
        showlegend=False,
    )
    apply_brand_chart_style(fig)
    return fig


def render_marketing_highlights(highlights: list[dict[str, str]]) -> None:
    """Faixa de cards horizontais com os principais destaques."""
    if not highlights:
        st.caption("Sem destaques relevantes para esta janela.")
        return
    cols = st.columns(len(highlights))
    for col, item in zip(cols, highlights):
        col.markdown(
            f"""
            <div class="highlight-card">
              <div class="highlight-title">{item.get('titulo', '')}</div>
              <div class="highlight-value">{item.get('valor', '')}</div>
              <div class="highlight-hint">{item.get('hint', '')}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_marketing_diario(
    scoped: pd.DataFrame,
    full_df: pd.DataFrame,
    percurso_targets: dict[str, int],
    data_base_label: str,
    data_base_ts: pd.Timestamp | None = None,
) -> None:
    """Visão executiva diária — KPIs essenciais + ritmo + 3 destaques + PDF dedicado."""
    inject_marketing_report_styles()
    deltas = compute_daily_deltas(scoped, data_base_ts)
    ref_day: date = deltas["ref_day"]  # type: ignore[assignment]
    weekday_label = deltas["weekday_label"]
    title = "Marketing Diário — Paraty Brazil by UTMB"
    subtitle = (
        f"{weekday_label}, {ref_day.strftime('%d/%m/%Y')} | "
        f"Base atualizada até {data_base_label}"
    )
    _render_report_hero(badge="DIÁRIO", badge_color=BRAND_COLORWAY[0], title=title, subtitle=subtitle)

    with st.container():
        button_col_left, button_col_right = st.columns([3, 1])
        with button_col_left:
            st.caption(
                "Visão de 1 página com inscritos, ritmo e progresso de metas — pensada para checagem rápida e envio."
            )
        with button_col_right:
            _render_pdf_button("Baixar PDF do Diário", key="pdf_marketing_diario")

    total_inscritos = int(len(scoped))
    receita_total = float(scoped["net_revenue_brl"].sum()) if "net_revenue_brl" in scoped.columns else 0.0
    female = count_female_entries(scoped)
    pct_female = (female / total_inscritos * 100) if total_inscritos else 0
    foreigners = (
        scoped[scoped["nationality_std"] != "BR"].shape[0] if "nationality_std" in scoped.columns else 0
    )
    pct_foreigners = (foreigners / total_inscritos * 100) if total_inscritos else 0
    summary = build_route_summary(scoped, percurso_targets)
    total_row = summary[summary["Percurso"] == "TOTAL"].iloc[0] if not summary.empty else None
    pct_meta_total = float(total_row["pct_meta"]) if total_row is not None else 0.0

    delta_inscritos = int(deltas["delta_inscritos"])  # type: ignore[arg-type]
    delta_receita = float(deltas["delta_receita"])  # type: ignore[arg-type]
    inscritos_hoje = int(deltas["inscritos_hoje"])  # type: ignore[arg-type]
    receita_hoje = float(deltas["receita_hoje"])  # type: ignore[arg-type]

    render_marketing_kpi_cards(
        [
            {
                "label": "Inscritos hoje",
                "value": format_int(inscritos_hoje),
                "delta": f"{delta_inscritos:+d} vs ontem",
            },
            {"label": "Inscritos total", "value": format_int(total_inscritos)},
            {
                "label": "% meta total",
                "value": format_pct(pct_meta_total),
                "help": "Inscritos atuais ÷ soma das metas por percurso.",
            },
            {
                "label": "Receita líquida hoje",
                "value": format_currency(receita_hoje),
                "delta": f"{delta_receita:+,.0f}".replace(",", ".") + " R$ vs ontem",
            },
            {"label": "Receita líquida total", "value": format_currency(receita_total)},
            {"label": "% mulheres", "value": format_pct(pct_female)},
            {"label": "% estrangeiros", "value": format_pct(pct_foreigners)},
            {
                "label": "Δ ontem",
                "value": f"{delta_inscritos:+d}",
                "help": "Variação absoluta de inscritos hoje vs ontem.",
            },
        ]
    )

    st.markdown("---")
    chart_col, hl_col = st.columns([3, 2])
    with chart_col:
        progress_fig = _build_route_progress_bars(summary)
        st.plotly_chart(progress_fig, use_container_width=True)
        pace_fig = _build_pace_chart(scoped)
        if pace_fig is not None:
            st.plotly_chart(pace_fig, use_container_width=True)
        else:
            st.info("Sem datas válidas para montar o gráfico de ritmo dos últimos 14 dias.")
    with hl_col:
        st.markdown('<div class="mkt-section-title">Destaques do dia</div>', unsafe_allow_html=True)
        highlights = compute_marketing_highlights(
            scoped, granularidade="diario", ref_ts=data_base_ts, targets=percurso_targets
        )
        if highlights:
            for item in highlights:
                st.markdown(
                    f"""
                    <div class="highlight-card">
                      <div class="highlight-title">{item.get('titulo', '')}</div>
                      <div class="highlight-value">{item.get('valor', '')}</div>
                      <div class="highlight-hint">{item.get('hint', '')}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.caption("Sem destaques relevantes nas últimas 24h.")


def render_marketing_semanal(
    scoped: pd.DataFrame,
    full_df: pd.DataFrame,
    percurso_targets: dict[str, int],
    start_date: date,
    end_date: date,
    data_base_label: str,
    data_base_ts: pd.Timestamp | None,
    ibge_df: pd.DataFrame,
    print_mode: bool,
) -> None:
    """Relatório executivo semanal — para envio toda segunda de manhã."""
    inject_marketing_report_styles()
    weekly = compute_weekly_deltas(scoped, data_base_ts)
    week_start: date = weekly["week_start"]  # type: ignore[assignment]
    week_end: date = weekly["week_end"]  # type: ignore[assignment]
    iso_year = weekly["iso_year"]
    iso_week = weekly["iso_week"]

    badge = f"SEMANA {iso_week:02d}/{iso_year}"
    title = "Marketing Semanal — Paraty Brazil by UTMB"
    subtitle = (
        f"Período: {week_start.strftime('%d/%m')} a {week_end.strftime('%d/%m/%Y')} | "
        f"Base atualizada até {data_base_label}"
    )
    _render_report_hero(badge=badge, badge_color=BRAND_COLORWAY[1], title=title, subtitle=subtitle)

    with st.container():
        button_col_left, button_col_right = st.columns([3, 1])
        with button_col_left:
            st.caption(
                "Relatório completo da semana com KPIs, ritmo, destaques e blocos analíticos — "
                "pronto para envio toda segunda."
            )
        with button_col_right:
            _render_pdf_button("Baixar PDF do Semanal", key="pdf_marketing_semanal")

    inscritos_semana = int(weekly["inscritos_semana"])  # type: ignore[arg-type]
    inscritos_ant = int(weekly["inscritos_semana_anterior"])  # type: ignore[arg-type]
    delta_inscritos_pct = float(weekly["delta_inscritos_pct"])  # type: ignore[arg-type]
    receita_semana = float(weekly["receita_semana"])  # type: ignore[arg-type]
    receita_ant = float(weekly["receita_semana_anterior"])  # type: ignore[arg-type]
    delta_receita_pct = float(weekly["delta_receita_pct"])  # type: ignore[arg-type]

    week_df_obj = weekly.get("week_df")
    week_df = week_df_obj if isinstance(week_df_obj, pd.DataFrame) else scoped.iloc[0:0]

    total_inscritos = int(len(scoped))
    summary = build_route_summary(scoped, percurso_targets)
    total_row = summary[summary["Percurso"] == "TOTAL"].iloc[0] if not summary.empty else None
    pct_meta_total = float(total_row["pct_meta"]) if total_row is not None else 0.0
    female = count_female_entries(scoped)
    pct_female = (female / total_inscritos * 100) if total_inscritos else 0
    foreigners = (
        scoped[scoped["nationality_std"] != "BR"].shape[0] if "nationality_std" in scoped.columns else 0
    )
    pct_foreigners = (foreigners / total_inscritos * 100) if total_inscritos else 0
    coupon_share = (
        int(scoped["total_discounts_brl"].gt(0).sum()) / total_inscritos * 100
        if total_inscritos and "total_discounts_brl" in scoped.columns
        else 0.0
    )

    def _flag_count(frame: pd.DataFrame, col: str) -> int:
        if col not in frame.columns:
            return 0
        return int(pd.to_numeric(frame[col], errors="coerce").fillna(0).gt(0).sum())

    def _nubank_card_payment_count(frame: pd.DataFrame) -> int:
        if frame is None or getattr(frame, "empty", True):
            return 0
        return int(nubank_card_payment_registration_mask(frame).sum())

    yopp_semana = _flag_count(week_df, "yopp_flag")
    yopp_acumulado = _flag_count(scoped, "yopp_flag")
    nubank_semana = _nubank_card_payment_count(week_df)
    nubank_acumulado = _nubank_card_payment_count(scoped)
    # "Interessados no ônibus oficial" = opt-in detectado em coluna de transporte/ônibus
    # (ver detect_official_bus_column + parse_opt_in_series → flag official_bus_flag).
    bus_semana = _flag_count(week_df, "official_bus_flag")
    bus_acumulado = _flag_count(scoped, "official_bus_flag")

    if "country_std" in scoped.columns:
        paises_acumulado = int(
            scoped["country_std"]
            .replace({"": pd.NA, "NAN": pd.NA})
            .dropna()
            .nunique()
        )
    else:
        paises_acumulado = 0

    render_marketing_kpi_cards(
        [
            {
                "label": "Inscritos na semana",
                "value": format_int(inscritos_semana),
                "delta": f"{delta_inscritos_pct:+.1f}% vs semana anterior",
                "help": f"Anterior: {format_int(inscritos_ant)} inscritos.",
            },
            {"label": "% meta total", "value": format_pct(pct_meta_total)},
            {
                "label": "Óculos Yopp vendidos (semana)",
                "value": format_int(yopp_semana),
                "delta": f"Acumulado: {format_int(yopp_acumulado)}",
                "help": "Inscrições com flag Yopp marcada (yopp_flag) registradas na semana corrente.",
            },
            {
                "label": "Pagamentos cartão Nubank (semana)",
                "value": format_int(nubank_semana),
                "delta": f"Acumulado: {format_int(nubank_acumulado)}",
                "help": "nubank_flag ativo e sem cupom válido (código ≥3 caracteres alfanuméricos normalizado). Exclui compras com cupom.",
            },
            {
                "label": "Interessados no ônibus oficial (semana)",
                "value": format_int(bus_semana),
                "delta": f"Acumulado: {format_int(bus_acumulado)}",
                "help": "Opt-in na coluna de transporte/ônibus oficial (official_bus_flag).",
            },
            {
                "label": "Países diferentes (acumulado)",
                "value": format_int(paises_acumulado),
                "help": "Países distintos entre todos os inscritos (campo Country).",
            },
            {"label": "% mulheres (acumulado)", "value": format_pct(pct_female)},
            {"label": "% estrangeiros (acumulado)", "value": format_pct(pct_foreigners)},
            {
                "label": "% c/ cupom (acumulado)",
                "value": format_pct(coupon_share),
                "help": "Inscrições com qualquer desconto aplicado.",
            },
        ]
    )

    st.markdown('<div class="mkt-section-title">Destaques da semana</div>', unsafe_allow_html=True)
    weekly_highlights = compute_marketing_highlights(
        scoped, granularidade="semanal", ref_ts=data_base_ts, targets=percurso_targets
    )
    melhor = weekly.get("melhor_dia")
    pior = weekly.get("pior_dia")
    if isinstance(melhor, dict):
        weekly_highlights.append(
            {
                "titulo": "Melhor dia da semana",
                "valor": f"{melhor['weekday']} ({melhor['data'].strftime('%d/%m')})",
                "hint": f"{format_int(int(melhor['inscritos']))} inscritos",
            }
        )
    if isinstance(pior, dict):
        weekly_highlights.append(
            {
                "titulo": "Dia mais fraco",
                "valor": f"{pior['weekday']} ({pior['data'].strftime('%d/%m')})",
                "hint": f"{format_int(int(pior['inscritos']))} inscritos",
            }
        )
    render_marketing_highlights(weekly_highlights[:5])

    insert_print_break(print_mode)
    render_progress_projection(
        scoped, percurso_targets, start_date, end_date, weekly_polish=True
    )
    render_marketing_target_gauges(scoped, percurso_targets, weekly_polish=True)
    insert_print_break(print_mode)
    render_demography(scoped, expandido=True)
    render_geography(scoped, ibge_df)
    render_international(scoped)
    insert_print_break(print_mode)
    render_marketing_coupon_block(scoped)
    render_team_medical_company(scoped)

    insert_print_break(print_mode)
    st.header("Comparativo: semana atual vs semana anterior")
    st.caption(
        f"Janela atual: {week_start.strftime('%d/%m')} a {week_end.strftime('%d/%m')} | "
        f"Janela anterior: {weekly['prev_week_start'].strftime('%d/%m')} a {weekly['prev_week_end'].strftime('%d/%m')}"
    )
    comparativo_df = pd.DataFrame(
        [
            {
                "Métrica": "Inscritos",
                "Semana atual": inscritos_semana,
                "Semana anterior": inscritos_ant,
                "Δ %": delta_inscritos_pct,
            },
            {
                "Métrica": "Receita líquida (R$)",
                "Semana atual": receita_semana,
                "Semana anterior": receita_ant,
                "Δ %": delta_receita_pct,
            },
        ]
    )
    comparativo_view = comparativo_df.copy()
    comparativo_view["Semana atual"] = comparativo_view.apply(
        lambda row: format_currency(row["Semana atual"]) if "Receita" in row["Métrica"] else format_int(row["Semana atual"]),
        axis=1,
    )
    comparativo_view["Semana anterior"] = comparativo_view.apply(
        lambda row: format_currency(row["Semana anterior"]) if "Receita" in row["Métrica"] else format_int(row["Semana anterior"]),
        axis=1,
    )
    comparativo_view["Δ %"] = comparativo_view["Δ %"].map(lambda v: f"{v:+.1f}%")
    st.dataframe(comparativo_view, hide_index=True, use_container_width=True)

    max_insc_pair = max(float(inscritos_ant), float(inscritos_semana), 1.0)
    col_insc_ant = BRAND_COLORWAY[3]
    col_insc_cur = BRAND_COLORWAY[0]
    fig_insc = go.Figure()
    fig_insc.add_trace(
        go.Bar(
            name="Semana anterior",
            x=["Inscritos"],
            y=[inscritos_ant],
            marker_color=col_insc_ant,
            text=[format_int(inscritos_ant)],
            textposition=_weekly_compare_textposition(inscritos_ant, max_insc_pair),
            insidetextfont=dict(color=_inside_bar_label_color(col_insc_ant), size=12),
            outsidetextfont=dict(color="#334155", size=12),
        )
    )
    fig_insc.add_trace(
        go.Bar(
            name="Semana atual",
            x=["Inscritos"],
            y=[inscritos_semana],
            marker_color=col_insc_cur,
            text=[format_int(inscritos_semana)],
            textposition=_weekly_compare_textposition(inscritos_semana, max_insc_pair),
            insidetextfont=dict(color=_inside_bar_label_color(col_insc_cur), size=12),
            outsidetextfont=dict(color="#334155", size=12),
        )
    )
    fig_insc.update_layout(
        barmode="group",
        title="Inscritos — semana atual vs anterior",
        height=340,
        yaxis_title="Inscritos",
        showlegend=True,
    )
    apply_brand_chart_style(fig_insc)
    fig_insc.update_layout(
        title=dict(text="Inscritos — semana atual vs anterior", x=0.5, xanchor="center"),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.07,
            x=0.5,
            xanchor="center",
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor="#e5e7eb",
            borderwidth=1,
        ),
        margin=dict(t=96, b=48, l=10, r=10),
    )
    _polish_weekly_compare_bar_figure(fig_insc)
    st.plotly_chart(fig_insc, use_container_width=True)

    max_rec_pair = max(float(receita_ant), float(receita_semana), 1.0)
    col_rec_ant = BRAND_COLORWAY[3]
    col_rec_cur = BRAND_COLORWAY[1]
    fig_rec = go.Figure()
    fig_rec.add_trace(
        go.Bar(
            name="Semana anterior",
            x=["Receita líquida (R$)"],
            y=[receita_ant],
            marker_color=col_rec_ant,
            text=[format_currency(receita_ant)],
            textposition=_weekly_compare_textposition(receita_ant, max_rec_pair),
            insidetextfont=dict(color=_inside_bar_label_color(col_rec_ant), size=11),
            outsidetextfont=dict(color="#334155", size=11),
        )
    )
    fig_rec.add_trace(
        go.Bar(
            name="Semana atual",
            x=["Receita líquida (R$)"],
            y=[receita_semana],
            marker_color=col_rec_cur,
            text=[format_currency(receita_semana)],
            textposition=_weekly_compare_textposition(receita_semana, max_rec_pair),
            insidetextfont=dict(color=_inside_bar_label_color(col_rec_cur), size=11),
            outsidetextfont=dict(color="#334155", size=11),
        )
    )
    fig_rec.update_layout(
        barmode="group",
        title="Receita líquida (R$) — semana atual vs anterior",
        height=340,
        yaxis_title="R$",
        showlegend=True,
    )
    apply_brand_chart_style(fig_rec)
    fig_rec.update_layout(
        title=dict(text="Receita líquida (R$) — semana atual vs anterior", x=0.5, xanchor="center"),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.07,
            x=0.5,
            xanchor="center",
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor="#e5e7eb",
            borderwidth=1,
        ),
        margin=dict(t=96, b=48, l=10, r=10),
    )
    _polish_weekly_compare_bar_figure(fig_rec)
    st.plotly_chart(fig_rec, use_container_width=True)


def render_header_financial(kpi_df: pd.DataFrame, data_base_label: str) -> None:
    total = len(kpi_df)
    gross = float(kpi_df["total_registration_brl"].sum())
    discounts = float(kpi_df["total_discounts_brl"].sum())
    net = float(kpi_df["net_revenue_brl"].sum())
    gross_brl, gross_usd_brl, _ = revenue_split_by_currency(kpi_df, "total_registration_brl")
    extras = compute_additional_revenue(kpi_df)
    discount_rate = (discounts / gross * 100) if gross else 0
    paying_with_discount = int(kpi_df["total_discounts_brl"].gt(0).sum())
    pct_discount_orders = (paying_with_discount / total * 100) if total else 0

    st.markdown(
        f"""
        <div class="hero">
          <h2>Dashboard Financeiro 2026 - Paraty Brazil by UTMB</h2>
          <p class="subtle">Base atualizada até {data_base_label} | Foco em faturamento separado por origem de moeda</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Inscritos ativos", format_int(total))
    c2.metric("Faturamento inscrições total", format_currency(gross))
    c3.metric("Descontos totais", format_currency(discounts))
    c4.metric("Faturamento inscrições BRL", format_currency(gross_brl))
    c5.metric("Faturamento inscrições USD (R$)", format_currency(gross_usd_brl))
    c6.metric("% desconto sobre bruto", format_pct(discount_rate))
    x1, x2, x3 = st.columns(3)
    x1.metric("Faturamento líquido inscrições", format_currency(net))
    x2.metric("Faturamento Yopp (R$)", format_currency(extras["yopp_total"]))
    x3.metric("Faturamento Ônibus PTR 17 (R$)", format_currency(extras["bus_ptr17_total"]))
    st.caption(
        f"Inscrições com desconto aplicado: {format_int(paying_with_discount)} "
        f"({format_pct(pct_discount_orders)}) | "
        f"Yopp: R$ {YOPP_PRICE_BRL:,.0f} ou USD {YOPP_PRICE_USD:,.0f} | "
        f"Ônibus PTR 17: R$ {PTR17_BUS_PRICE_BRL:,.0f} ou USD {PTR17_BUS_PRICE_USD:,.0f}"
    )


def build_route_summary(df: pd.DataFrame, targets: dict[str, int]) -> pd.DataFrame:
    local = df.copy()
    local["_female_flag"] = female_mask_from_series(get_gender_series(local)).astype(int)
    grouped = (
        local.groupby("Competition", dropna=False)
        .agg(
            inscritos=("Competition", "size"),
            receita_bruta=("total_registration_brl", "sum"),
            descontos=("total_discounts_brl", "sum"),
            receita_liquida=("net_revenue_brl", "sum"),
            mulheres=("_female_flag", "sum"),
        )
        .reset_index()
        .rename(columns={"Competition": "Percurso"})
    )
    grouped["meta"] = grouped["Percurso"].map(targets).fillna(0).astype(int)
    grouped["pct_meta"] = grouped.apply(
        lambda row: (row["inscritos"] / row["meta"] * 100) if row["meta"] else 0,
        axis=1,
    )
    grouped["pct_mulheres"] = grouped.apply(
        lambda row: (row["mulheres"] / row["inscritos"] * 100) if row["inscritos"] else 0,
        axis=1,
    )

    route_base = pd.DataFrame({"Percurso": PERCURSO_ORDER})
    route_base["meta"] = route_base["Percurso"].map(targets).fillna(0).astype(int)
    route_summary = route_base.merge(
        grouped.drop(columns="meta"),
        on="Percurso",
        how="left",
    ).fillna(
        {
            "inscritos": 0,
            "receita_bruta": 0,
            "descontos": 0,
            "receita_liquida": 0,
            "mulheres": 0,
            "pct_meta": 0,
            "pct_mulheres": 0,
        }
    )
    route_summary["inscritos"] = route_summary["inscritos"].astype(int)
    route_summary["mulheres"] = route_summary["mulheres"].astype(int)

    total_inscritos = int(route_summary["inscritos"].sum())
    total_meta = int(route_summary["meta"].sum())
    total_mulheres = int(route_summary["mulheres"].sum())
    total_row = pd.DataFrame(
        [
            {
                "Percurso": "TOTAL",
                "meta": total_meta,
                "inscritos": total_inscritos,
                "receita_bruta": float(route_summary["receita_bruta"].sum()),
                "descontos": float(route_summary["descontos"].sum()),
                "receita_liquida": float(route_summary["receita_liquida"].sum()),
                "mulheres": total_mulheres,
                "pct_meta": (total_inscritos / total_meta * 100) if total_meta else 0,
                "pct_mulheres": (total_mulheres / total_inscritos * 100) if total_inscritos else 0,
            }
        ]
    )
    return pd.concat([route_summary, total_row], ignore_index=True)


def render_progress_projection(
    df: pd.DataFrame,
    targets: dict[str, int],
    start_date: date,
    end_date: date,
    show_target_gauges: bool = False,
    weekly_polish: bool = False,
) -> None:
    st.header("Projeções e ritmo de vendas")
    summary = build_route_summary(df, targets)
    table = summary.copy()
    table["Meta"] = table["meta"].map(format_int)
    table["Inscritos atuais"] = table["inscritos"].map(format_int)
    table["% meta"] = table["pct_meta"].map(format_pct)
    table["% mulheres"] = table["pct_mulheres"].map(format_pct)
    st.dataframe(
        table[["Percurso", "Meta", "Inscritos atuais", "% meta", "% mulheres"]],
        hide_index=True,
        use_container_width=True,
    )

    order_with_total = list(PERCURSO_ORDER) + ["TOTAL"]
    if weekly_polish:
        st.markdown(
            '<div class="mkt-section-title">Metas por percurso — desempenho vs meta</div>',
            unsafe_allow_html=True,
        )
        st.caption(
            "Faixa cinza: meta de inscritos. Barra colorida: inscritos atuais (cor por faixa de % da meta). "
            "Valores absolutos facilitam comparar PTR e RUN lado a lado."
        )
        fig_metas = build_weekly_metas_bullet_figure(summary)
        st.plotly_chart(fig_metas, use_container_width=True)
    else:
        # Grafico 100% empilhado: todas as barras com mesma altura visual (0 a 100%)
        chart_summary = summary.copy()
        chart_summary["Percurso"] = pd.Categorical(
            chart_summary["Percurso"],
            categories=order_with_total,
            ordered=True,
        )
        chart_summary = chart_summary.sort_values("Percurso").reset_index(drop=True)
        chart_summary["pct_meta_cap"] = chart_summary["pct_meta"].clip(lower=0, upper=100)
        chart_summary["pct_restante"] = 100 - chart_summary["pct_meta_cap"]
        st.subheader("Metas por percurso (inscritos atuais)")
        fig_comp = go.Figure()
        fig_comp.add_trace(
            go.Bar(
                x=chart_summary["Percurso"],
                y=chart_summary["pct_restante"],
                name="Restante até meta",
                marker_color="rgba(200, 200, 200, 0.5)",
                text=[f"Meta: {int(v)}" for v in chart_summary["meta"]],
                textposition="none",
                customdata=chart_summary[["meta", "inscritos", "pct_meta"]].values,
                hovertemplate=(
                    "Percurso: %{x}<br>"
                    "Meta: %{customdata[0]}<br>"
                    "Inscritos: %{customdata[1]}<br>"
                    "% da meta: %{customdata[2]:.1f}%<extra></extra>"
                ),
                showlegend=True,
            )
        )
        fig_comp.add_trace(
            go.Bar(
                x=chart_summary["Percurso"],
                y=chart_summary["pct_meta_cap"],
                name="Inscritos",
                text=[
                    f"{int(r['inscritos'])}/{int(r['meta'])}<br>{format_pct(r['pct_meta'])}"
                    for _, r in chart_summary.iterrows()
                ],
                textposition="inside",
                marker_color="rgba(59, 130, 246, 0.9)",
                customdata=chart_summary[["meta", "inscritos", "pct_meta"]].values,
                hovertemplate=(
                    "Percurso: %{x}<br>"
                    "Meta: %{customdata[0]}<br>"
                    "Inscritos: %{customdata[1]}<br>"
                    "% da meta: %{customdata[2]:.1f}%<extra></extra>"
                ),
                showlegend=True,
            )
        )
        fig_comp = style_bar_labels(fig_comp)
        fig_comp.update_traces(textposition="none", selector=dict(name="Restante até meta"))
        fig_comp.update_layout(
            height=380,
            barmode="stack",
            xaxis={"categoryorder": "array", "categoryarray": order_with_total},
            yaxis_title="% da meta",
            yaxis=dict(range=[0, 100]),
            margin=dict(b=80),
        )
        st.plotly_chart(fig_comp, use_container_width=True)

    if show_target_gauges:
        st.subheader("Gauges de meta por percurso e total")
        gauge_order = ["TOTAL"] + PERCURSO_ORDER
        gauge_df = summary.copy()
        gauge_df["Percurso"] = pd.Categorical(gauge_df["Percurso"], categories=gauge_order, ordered=True)
        gauge_df = gauge_df.sort_values("Percurso").reset_index(drop=True)
        gauge_cols = st.columns(3)
        for idx, row in gauge_df.iterrows():
            percurso = str(row["Percurso"])
            inscritos = float(row["inscritos"])
            meta = float(row["meta"])
            pct_meta = float(row["pct_meta"])
            axis_max = max(meta * 1.05, inscritos * 1.05, 1.0)

            gauge_fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=inscritos,
                    number={"valueformat": ",.0f"},
                    title={"text": f"{percurso}<br>Meta: {int(meta)}"},
                    gauge={
                        "axis": {"range": [0, axis_max]},
                        "bar": {"color": "#2563eb"},
                        "threshold": {
                            "line": {"color": "#ef4444", "width": 3},
                            "thickness": 0.9,
                            "value": meta,
                        },
                    },
                )
            )
            gauge_fig.update_layout(height=250, margin=dict(l=10, r=10, t=60, b=20))

            with gauge_cols[idx % 3]:
                st.plotly_chart(gauge_fig, use_container_width=True)
                st.caption(f"Inscritos: {format_int(inscritos)} | % da meta: {format_pct(pct_meta)}")

    comp_counts = summary[summary["Percurso"] != "TOTAL"]
    meta_total = int(comp_counts["meta"].sum())
    total = len(df)
    elapsed_days = max((date.today() - start_date).days, 0)
    total_days = max((end_date - start_date).days, 1)
    pct_elapsed = min(elapsed_days / total_days * 100, 100)
    pct_meta = (total / meta_total * 100) if meta_total else 0
    col_a, col_b = st.columns(2)
    col_a.metric("% prazo decorrido", format_pct(pct_elapsed))
    col_b.metric("% meta de inscritos", format_pct(pct_meta))

    series = (
        df.dropna(subset=["Registration date"])
        .assign(day=lambda d: d["Registration date"].dt.date)
        .groupby("day")
        .size()
        .reset_index(name="inscricoes_diarias")
        .sort_values("day")
    )
    if series.empty:
        st.info("Sem datas suficientes para projeções.")
        return

    series["mm7"] = series["inscricoes_diarias"].rolling(7, min_periods=1).mean()
    series["mm15"] = series["inscricoes_diarias"].rolling(15, min_periods=1).mean()
    series["mm30"] = series["inscricoes_diarias"].rolling(30, min_periods=1).mean()
    proj_rate = series["mm15"].iloc[-1]
    if pd.isna(proj_rate):
        proj_rate = 0.0
    days_left = max((end_date - series["day"].iloc[-1]).days, 0)
    proj_total = int(round(total + float(proj_rate) * days_left))

    st.subheader("Média móvel e projeção")
    if weekly_polish:
        last_day = series["day"].iloc[-1]
        st.caption(
            f"Projeção ao fim da campanha ({end_date.strftime('%d/%m/%Y')}): "
            "inscritos acumulados até "
            f"{last_day.strftime('%d/%m/%Y')} + (última média móvel em inscrições/dia) × "
            f"{days_left} dia(s) restantes. Traços mostram apenas médias móveis no painel inferior — "
            "eixo Y dedicado para leitura sem compressão pelos picos diários."
        )
        fig_mm, projs = build_weekly_ma_projection_figure(series, total, end_date)
        m1, m2, m3 = st.columns(3)
        m1.metric("Projeção total (MM 7d)", format_int(projs["mm7"]))
        m2.metric("Projeção total (MM 15d)", format_int(projs["mm15"]))
        m3.metric("Projeção total (MM 30d)", format_int(projs["mm30"]))
        st.plotly_chart(fig_mm, use_container_width=True)
    else:
        st.metric("Projeção de inscritos no prazo", format_int(proj_total))
        fig_mm = px.line(
            series,
            x="day",
            y=["inscricoes_diarias", "mm7", "mm15", "mm30"],
            labels={"day": "Data", "value": "Inscrições", "variable": "Série"},
        )
        fig_mm.update_layout(height=360)
        st.plotly_chart(fig_mm, use_container_width=True)


def render_marketing_target_gauges(
    df: pd.DataFrame,
    targets: dict[str, int],
    *,
    weekly_polish: bool = False,
) -> None:
    st.header("Gauges de metas de atletas (Marketing)")
    if weekly_polish:
        st.caption(
            "Mesmo tipo de leitura do gráfico de Metas acima: faixa cinza = meta; barra colorida = "
            "inscritos (ou óculos Yopp), cor por faixa de % da meta; rótulos à direita evitam sobreposição no centro."
        )
        st.plotly_chart(build_marketing_gauges_grid_figure(df, targets), use_container_width=True)
        return

    summary = build_route_summary(df, targets)
    gauge_order = ["TOTAL"] + PERCURSO_ORDER
    gauge_df = summary.copy()
    gauge_df["Percurso"] = pd.Categorical(gauge_df["Percurso"], categories=gauge_order, ordered=True)
    gauge_df = gauge_df.sort_values("Percurso").reset_index(drop=True)

    gauge_cols = st.columns(3)

    for idx, row in gauge_df.iterrows():
        percurso = str(row["Percurso"])
        inscritos = float(row["inscritos"])
        meta = float(row["meta"])
        pct_meta = float(row["pct_meta"])
        axis_max = max(meta * 1.05, inscritos * 1.05, 1.0)

        gauge_fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=inscritos,
                number={"valueformat": ",.0f"},
                gauge={
                    "axis": {"range": [0, axis_max]},
                    "bar": {"color": "#2563eb"},
                    "threshold": {
                        "line": {"color": "#ef4444", "width": 3},
                        "thickness": 0.9,
                        "value": meta,
                    },
                },
            )
        )
        gauge_fig.update_layout(height=250, margin=dict(l=10, r=10, t=20, b=20))
        with gauge_cols[idx % 3]:
            st.markdown(f"**{percurso}**  \nMeta: {format_int(meta)}")
            st.plotly_chart(gauge_fig, use_container_width=True)
            st.caption(f"Inscritos: {format_int(inscritos)} | % da meta: {format_pct(pct_meta)}")

    # Gauge adicional de meta de vendas de Óculos Yopp
    yopp_total = int(df["yopp_flag"].sum()) if "yopp_flag" in df.columns else 0
    yopp_meta = 300
    yopp_pct_meta = (yopp_total / yopp_meta * 100) if yopp_meta else 0
    yopp_axis_max = max(yopp_meta * 1.05, yopp_total * 1.05, 1.0)
    yopp_fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=yopp_total,
            number={"valueformat": ",.0f"},
            gauge={
                "axis": {"range": [0, yopp_axis_max]},
                "bar": {"color": "#2563eb"},
                "threshold": {
                    "line": {"color": "#ef4444", "width": 3},
                    "thickness": 0.9,
                    "value": yopp_meta,
                },
            },
        )
    )
    yopp_fig.update_layout(height=250, margin=dict(l=10, r=10, t=20, b=20))
    with gauge_cols[len(gauge_df) % 3]:
        st.markdown(f"**Óculos Yopp**  \nMeta: {format_int(yopp_meta)}")
        st.plotly_chart(yopp_fig, use_container_width=True)
        st.caption(f"Vendidos: {format_int(yopp_total)} | % da meta: {format_pct(yopp_pct_meta)}")


def render_demography(df: pd.DataFrame, expandido: bool = False) -> None:
    st.header("Demografia")
    valid_age = df["age"].dropna()
    c1, c2, c3 = st.columns(3)
    c1.metric("Idade média", format_int(valid_age.mean() if not valid_age.empty else 0))
    c2.metric("Idade mínima", format_int(valid_age.min() if not valid_age.empty else 0))
    c3.metric("Idade máxima", format_int(valid_age.max() if not valid_age.empty else 0))

    if not valid_age.empty:
        hist = ff.create_distplot([valid_age], ["Idade"], show_hist=True, show_rug=False, bin_size=2)
        hist.update_layout(height=320)
        st.plotly_chart(hist, use_container_width=True)

    if not expandido:
        return

    st.subheader("Gênero")
    gender_col = find_column_by_candidates(df.columns, AI_GENDER_CANDIDATES)
    if not gender_col:
        st.info("Coluna de gênero não disponível para análise.")
    else:
        gender_raw = df[gender_col].astype(str).str.strip().str.lower()
        gender_norm = pd.Series(pd.NA, index=df.index, dtype="object")
        gender_norm.loc[gender_raw.isin(AI_FEMALE_TOKENS)] = "Feminino"
        gender_norm.loc[gender_raw.isin(AI_MALE_TOKENS)] = "Masculino"
        gender_counts = (
            gender_norm.dropna().value_counts().reindex(["Feminino", "Masculino"], fill_value=0)
            .rename_axis("Gênero")
            .reset_index(name="Inscritos")
        )
        total_gender = gender_counts["Inscritos"].sum()
        gender_counts["%"] = gender_counts["Inscritos"].apply(
            lambda v: format_pct((v / total_gender) * 100 if total_gender else 0)
        )
        col_a, col_b = st.columns(2)
        col_a.dataframe(gender_counts, hide_index=True, use_container_width=True)
        pie_gender = px.pie(gender_counts, names="Gênero", values="Inscritos", hole=0.45)
        pie_gender.update_layout(height=320)
        col_b.plotly_chart(pie_gender, use_container_width=True)

    st.subheader("Faixas etárias")
    if valid_age.empty:
        st.info("Sem dados de idade para montar faixas etárias.")
    else:
        age_labels = ["18-24", "25-34", "35-44", "45-54", "55+"]
        age_bins = pd.cut(
            valid_age,
            bins=[17, 24, 34, 44, 54, 120],
            labels=age_labels,
            include_lowest=True,
        )
        age_counts = (
            age_bins.value_counts()
            .reindex(age_labels, fill_value=0)
            .rename_axis("Faixa")
            .reset_index(name="Inscritos")
        )
        total_age = age_counts["Inscritos"].sum()
        age_counts["%"] = age_counts["Inscritos"].apply(lambda v: format_pct((v / total_age) * 100 if total_age else 0))
        col_c, col_d = st.columns(2)
        col_c.dataframe(age_counts, hide_index=True, use_container_width=True)
        fig_age = px.bar(age_counts, x="Faixa", y="Inscritos", text="Inscritos")
        fig_age = style_bar_labels(fig_age)
        fig_age.update_layout(height=320)
        col_d.plotly_chart(fig_age, use_container_width=True)


def render_geography(df: pd.DataFrame, ibge_df: pd.DataFrame) -> None:
    st.header("Geografia Brasil")
    brazil = df[df["country_std"].isin(["BRAZIL", "BRASIL", "BR"])].copy()
    if brazil.empty:
        st.info("Sem inscritos do Brasil para exibir nesta filtragem.")
        return

    city_choices = ibge_df["City_norm"].tolist()
    brazil["city_display"] = brazil["city"].astype(str).apply(lambda c: correct_city(c, ibge_df, city_choices))
    brazil["city_norm"] = brazil["city_display"].map(norm_text)
    total_brazil = len(brazil)
    city_counts = brazil["city_display"].value_counts().rename_axis("Cidade").reset_index(name="Inscritos")
    city_counts["% brasileiros"] = city_counts["Inscritos"].apply(lambda v: format_pct((v / total_brazil) * 100))
    n_cidades = int(brazil["city_display"].nunique())
    st.subheader(f"Top cidades ({n_cidades} cidades)")
    st.dataframe(city_counts, hide_index=True, use_container_width=True)

    merged = brazil.merge(ibge_df[["City_norm", "UF", "Região"]], left_on="city_norm", right_on="City_norm", how="left")
    uf_counts = merged["UF"].fillna("NA").value_counts().rename_axis("UF").reset_index(name="Inscritos")
    uf_counts["% brasileiros"] = uf_counts["Inscritos"].apply(lambda v: format_pct((v / total_brazil) * 100))
    uf_counts = uf_counts.sort_values("Inscritos", ascending=False)
    reg_counts = merged["Região"].fillna("NA").value_counts().rename_axis("Regiao").reset_index(name="Inscritos")

    n_ufs = int(merged["UF"].fillna("NA").nunique())
    st.subheader(f"Top Estados ({n_ufs} estados)")
    st.dataframe(uf_counts, hide_index=True, use_container_width=True)

    st.subheader("Regiões do Brasil")
    col1, col2 = st.columns(2)
    col1.dataframe(reg_counts.sort_values("Inscritos", ascending=False), hide_index=True, use_container_width=True)
    pie = px.pie(reg_counts, names="Regiao", values="Inscritos", hole=0.4)
    pie.update_layout(height=320)
    col2.plotly_chart(pie, use_container_width=True)
    st.caption("NA = cidades sem correspondência no IBGE para mapeamento de região.")


def render_international(df: pd.DataFrame) -> None:
    st.header("Internacional")
    intl = df[df["nationality_std"] != "BR"].copy()
    if intl.empty:
        st.info("Sem estrangeiros para o recorte atual.")
        return

    total_inscritos = len(df)
    nat_counts = intl["nationality_std"].fillna("NA").value_counts().rename_axis("País").reset_index(name="Inscritos")
    top_5 = nat_counts.head(5).copy()
    top_5["% dos inscritos totais"] = top_5["Inscritos"].apply(lambda v: format_pct((v / total_inscritos) * 100))
    st.dataframe(top_5, hide_index=True, use_container_width=True)
    with st.expander("Lista completa de países", expanded=st.session_state.get("print_mode", False)):
        full = nat_counts.copy()
        full["% dos inscritos totais"] = full["Inscritos"].apply(lambda v: format_pct((v / total_inscritos) * 100))
        st.dataframe(full, hide_index=True, use_container_width=True)


def render_historical(df: pd.DataFrame) -> None:
    st.header("Comparativo histórico")
    yearly = df.groupby("Ano").size().rename("Inscritos").reset_index().sort_values("Ano")
    if yearly["Ano"].nunique() <= 1:
        st.info("Somente um ano detectado no upload. Este modulo fica ativo quando houver anos adicionais.")
        return

    st.plotly_chart(px.line(yearly, x="Ano", y="Inscritos", markers=True), use_container_width=True)
    if "Email" in df.columns:
        unique_by_year = df.dropna(subset=["Email"]).groupby("Ano")["Email"].nunique().rename("Atletas únicos").reset_index()
        fig_unique = px.bar(unique_by_year, x="Ano", y="Atletas únicos", text="Atletas únicos")
        fig_unique = style_bar_labels(fig_unique)
        st.plotly_chart(fig_unique, use_container_width=True)


def render_sales_patterns(df: pd.DataFrame) -> None:
    st.header("Padrões de venda")
    dated = df.dropna(subset=["Registration date"]).copy()
    if dated.empty:
        st.info("Sem datas para análise de padrão de vendas.")
        return

    dated["day"] = dated["Registration date"].dt.date
    dated["weekday"] = dated["Registration date"].dt.day_name()

    # Mantém 12 janelas semanais discretas para evitar barra única em eixo contínuo.
    base_day = pd.to_datetime(dated["day"]).max()
    weekly_intervals = []
    for i in range(12):
        end = base_day - pd.Timedelta(days=7 * i)
        start = end - pd.Timedelta(days=6)
        weekly_intervals.append((start.normalize(), end.normalize()))

    weekly_data = []
    day_series = pd.to_datetime(dated["day"])
    for start, end in weekly_intervals:
        cnt = int(((day_series >= start) & (day_series <= end)).sum())
        label = f"{start.strftime('%d/%m')} - {end.strftime('%d/%m')}"
        weekly_data.append({"Semana": label, "Inscrições": cnt})

    week_counts = pd.DataFrame(weekly_data)[::-1].reset_index(drop=True)
    day_counts = dated.groupby("day").size().reset_index(name="Inscrições").sort_values("day")
    weekday_counts = dated.groupby("weekday").size().reset_index(name="Inscrições")
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekday_counts["weekday"] = pd.Categorical(weekday_counts["weekday"], categories=weekday_order, ordered=True)
    weekday_counts = weekday_counts.sort_values("weekday")

    fig_week = px.bar(week_counts, x="Semana", y="Inscrições", title="Últimas 12 semanas", text="Inscrições")
    fig_week = style_bar_labels(fig_week)
    st.plotly_chart(fig_week, use_container_width=True)
    col1, col2 = st.columns(2)
    fig_days = px.bar(day_counts.tail(30), x="day", y="Inscrições", title="Últimos 30 dias", text="Inscrições")
    fig_days = style_bar_labels(fig_days)
    col1.plotly_chart(fig_days, use_container_width=True)
    fig_weekday = px.bar(
        weekday_counts,
        x="weekday",
        y="Inscrições",
        title="Média por dia da semana",
        text="Inscrições",
    )
    fig_weekday = style_bar_labels(fig_weekday)
    col2.plotly_chart(fig_weekday, use_container_width=True)

    day_counts["acumulado"] = day_counts["Inscrições"].cumsum()
    st.plotly_chart(px.line(day_counts, x="day", y="acumulado", title="Inscrições acumuladas"), use_container_width=True)


def render_horarios_venda(df: pd.DataFrame) -> None:
    st.header("Horários de venda")
    if "date_time_parsed" not in df.columns:
        st.info("Coluna date_time não disponível para análise de horários.")
        return

    local = df.dropna(subset=["date_time_parsed"]).copy()
    if local.empty:
        st.info("Sem registros válidos em date_time para análise de horários.")
        return

    if "sale_hour" not in local.columns:
        local["sale_hour"] = local["date_time_parsed"].dt.hour
    hour_counts = local.groupby("sale_hour").size().reindex(range(24), fill_value=0).reset_index(name="Inscrições")
    hour_counts = hour_counts.rename(columns={"sale_hour": "Hora"})
    peak_hour = int(hour_counts.loc[hour_counts["Inscrições"].idxmax(), "Hora"])
    peak_count = int(hour_counts["Inscrições"].max())
    c1, c2 = st.columns(2)
    c1.metric("Horário de pico", f"{peak_hour:02d}h")
    c2.metric("Inscrições no pico", format_int(peak_count))

    fig_hour = px.bar(hour_counts, x="Hora", y="Inscrições", text="Inscrições", title="Inscrições por hora do dia")
    fig_hour = style_bar_labels(fig_hour)
    fig_hour.update_layout(height=340, xaxis=dict(dtick=1))
    st.plotly_chart(fig_hour, use_container_width=True)

    if "sale_period" not in local.columns:
        local["sale_period"] = pd.cut(
            local["sale_hour"],
            bins=[-0.1, 5.9, 11.9, 17.9, 23.9],
            labels=["Madrugada", "Manhã", "Tarde", "Noite"],
        ).astype("object")
        local["sale_period"] = local["sale_period"].fillna("Sem horário")
    period_order = ["Madrugada", "Manhã", "Tarde", "Noite", "Sem horário"]
    period_counts = local["sale_period"].value_counts().reindex(period_order, fill_value=0).rename_axis("Período")
    period_counts = period_counts.reset_index(name="Inscritos")
    total_period = int(period_counts["Inscritos"].sum())
    period_counts["%"] = period_counts["Inscritos"].apply(
        lambda v: format_pct((v / total_period) * 100 if total_period else 0)
    )

    top_period = period_counts.sort_values("Inscritos", ascending=False).iloc[0]
    st.metric("Maior volume por período", f"{top_period['Período']} ({format_int(top_period['Inscritos'])})")
    p1, p2 = st.columns(2)
    p1.dataframe(period_counts, hide_index=True, use_container_width=True)
    fig_period = px.bar(period_counts, x="Período", y="Inscritos", text="Inscritos", title="Inscrições por período do dia")
    fig_period = style_bar_labels(fig_period)
    fig_period.update_layout(height=320)
    p2.plotly_chart(fig_period, use_container_width=True)


def render_team_medical_company(df: pd.DataFrame) -> None:
    st.header("Assessorias, atestado e empresa")
    total = len(df)
    if total == 0:
        st.info("Sem dados para análise de assessorias e cadastro.")
        return

    team_col = find_column_by_candidates(df.columns, ["Team", "Assessoria", "Assesoria", "Training team"])
    medical_col = find_column_by_candidates(df.columns, ["medical_term", "medical term", "medical"])
    company_col = find_column_by_candidates(df.columns, ["company", "empresa"])

    # Indicadores de preenchimento de campos importantes para CRM/compliance.
    if company_col:
        company_txt = df[company_col].astype(str).str.strip()
        company_filled_mask = (~company_txt.isin(["", "nan", "None", "none", "NaN"])) & company_txt.notna()
        company_filled = int(company_filled_mask.sum())
    else:
        company_filled = 0

    if medical_col:
        medical_txt = df[medical_col].astype(str).str.strip()
        medical_link_mask = medical_txt.str.contains(r"(https?://|www\.)", case=False, regex=True, na=False)
        medical_uploaded = int(medical_link_mask.sum())
    else:
        medical_uploaded = 0

    c1, c2, c3 = st.columns(3)
    c1.metric("Inscritos ativos", format_int(total))
    c2.metric("Preencheram company", format_int(company_filled))
    c2.caption(format_pct((company_filled / total) * 100 if total else 0))
    c3.metric("Subiram atestado (medical_term)", format_int(medical_uploaded))
    c3.caption(format_pct((medical_uploaded / total) * 100 if total else 0))

    st.subheader("Top 10 assessorias (Team)")
    if not team_col:
        st.info("Coluna Team não disponível nesta base.")
        return

    team_raw = df[team_col].astype(str).str.strip()
    valid_team = team_raw.replace({"nan": pd.NA, "None": pd.NA, "none": pd.NA, "": pd.NA}).dropna()
    if valid_team.empty:
        st.info("Nenhum valor de Team preenchido no recorte atual.")
        return

    team_df = valid_team.rename("team_raw").to_frame()
    team_df["team_key"] = team_df["team_raw"].apply(canonicalize_team_name)
    team_df = team_df[team_df["team_key"] != ""].copy()
    team_df = team_df[~team_df["team_key"].str.contains(r"\bavulso\b", na=False)].copy()
    if team_df.empty:
        st.info("Não foi possível consolidar os nomes de Team.")
        return

    team_grouped = (
        team_df.groupby("team_key", as_index=False)
        .agg(
            atletas=("team_raw", "size"),
            assessoria=("team_raw", lambda s: s.value_counts().idxmax()),
        )
        .sort_values("atletas", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )
    team_grouped["% dos inscritos"] = team_grouped["atletas"].apply(lambda v: format_pct((v / total) * 100 if total else 0))
    team_grouped = team_grouped.rename(columns={"assessoria": "Assessoria", "atletas": "Atletas"})

    t1, t2 = st.columns(2)
    t1.dataframe(team_grouped[["Assessoria", "Atletas", "% dos inscritos"]], hide_index=True, use_container_width=True)
    fig_team = px.bar(team_grouped, x="Assessoria", y="Atletas", text="Atletas", title="Top 10 assessorias inscritas")
    fig_team = style_bar_labels(fig_team)
    fig_team.update_layout(height=360, xaxis_tickangle=-25)
    t2.plotly_chart(fig_team, use_container_width=True)


def render_perfil_inscrito(df: pd.DataFrame, ibge_df: pd.DataFrame) -> None:
    st.header("Ônibus Oficial")
    total = len(df)
    if total == 0:
        st.info("Sem dados para análise de ônibus oficial no recorte atual.")
        return

    if "official_bus_flag" not in df.columns:
        st.info("Coluna de interesse no ônibus oficial não disponível nesta base.")
        return

    interested = df[df["official_bus_flag"] > 0].copy()
    bus_total = len(interested)
    bus_pct = (bus_total / total * 100) if total else 0
    br_mask = interested["nationality_std"].eq("BR") if "nationality_std" in interested.columns else pd.Series(False, index=interested.index)
    bus_br = int(br_mask.sum())
    bus_foreign = int((~br_mask).sum()) if "nationality_std" in interested.columns else 0

    female_bus = count_female_entries(interested)
    male_bus = count_male_entries(interested)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Interessados", format_int(bus_total))
    c2.metric("% de inscritos interessados", format_pct(bus_pct))
    c3.metric("Mulheres interessadas", format_int(female_bus))
    c4.metric("Homens interessados", format_int(male_bus))
    c5.metric("Brasileiros / Estrangeiros", f"{format_int(bus_br)} / {format_int(bus_foreign)}")

    st.subheader("Interesse por percurso")
    if interested.empty:
        st.info("Nenhum interessado em ônibus oficial no recorte atual.")
    else:
        percurso_counts = (
            interested["Competition"]
            .fillna("NA")
            .value_counts()
            .rename_axis("Percurso")
            .reset_index(name="Interessados")
        )
        order_map = {name: idx for idx, name in enumerate(PERCURSO_ORDER)}
        percurso_counts["ordem"] = percurso_counts["Percurso"].map(order_map).fillna(999).astype(int)
        percurso_counts = percurso_counts.sort_values(["ordem", "Percurso"]).drop(columns="ordem").reset_index(drop=True)
        percurso_counts["% dos interessados"] = percurso_counts["Interessados"].apply(
            lambda v: format_pct((v / bus_total) * 100 if bus_total else 0)
        )

        p1, p2 = st.columns(2)
        p1.dataframe(percurso_counts, hide_index=True, use_container_width=True)
        fig_percurso = px.bar(percurso_counts, x="Percurso", y="Interessados", text="Interessados")
        fig_percurso = style_bar_labels(fig_percurso)
        fig_percurso.update_layout(height=340)
        p2.plotly_chart(fig_percurso, use_container_width=True)

    st.subheader("Top 5 cidades com interesse")
    if interested.empty:
        st.info("Sem interessados para montar ranking de cidades.")
        return

    top_cities = build_top_cities_table(
        interested,
        ibge_df=ibge_df,
        total=bus_total,
        count_col="Interessados",
        pct_col="% dos interessados",
    )
    st.dataframe(top_cities, hide_index=True, use_container_width=True)


def render_yopp_section(df: pd.DataFrame, ibge_df: pd.DataFrame) -> None:
    st.header("Yopp")
    total = len(df)
    if total == 0:
        st.info("Sem dados para análise de Yopp no recorte atual.")
        return

    if "yopp_flag" not in df.columns:
        st.info("Coluna Yopp não disponível nesta base.")
        return

    yopp_buyers = df[df["yopp_flag"] > 0].copy()
    yopp_total = len(yopp_buyers)
    pct_yopp = (yopp_total / total * 100) if total else 0

    female_yopp = count_female_entries(yopp_buyers)
    male_yopp = count_male_entries(yopp_buyers)

    br_mask = yopp_buyers["nationality_std"].eq("BR") if "nationality_std" in yopp_buyers.columns else pd.Series(False, index=yopp_buyers.index)
    br_yopp = int(br_mask.sum())
    foreign_yopp = int((~br_mask).sum()) if "nationality_std" in yopp_buyers.columns else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Óculos vendidos", format_int(yopp_total))
    c2.metric("% inscritos que compraram", format_pct(pct_yopp))
    c3.metric("Mulheres", format_int(female_yopp))
    c4.metric("Homens", format_int(male_yopp))
    c5.metric("Brasileiros / Estrangeiros", f"{format_int(br_yopp)} / {format_int(foreign_yopp)}")

    st.subheader("Óculos vendidos por percurso")
    if yopp_buyers.empty:
        st.info("Nenhum óculos Yopp vendido no recorte atual.")
        return

    yopp_by_course = (
        yopp_buyers["Competition"]
        .fillna("NA")
        .value_counts()
        .rename_axis("Percurso")
        .reset_index(name="Óculos vendidos")
    )
    order_map = {name: idx for idx, name in enumerate(PERCURSO_ORDER)}
    yopp_by_course["ordem"] = yopp_by_course["Percurso"].map(order_map).fillna(999).astype(int)
    yopp_by_course = yopp_by_course.sort_values(["ordem", "Percurso"]).drop(columns="ordem").reset_index(drop=True)
    yopp_by_course["% do total Yopp"] = yopp_by_course["Óculos vendidos"].apply(
        lambda v: format_pct((v / yopp_total) * 100 if yopp_total else 0)
    )

    y1, y2 = st.columns(2)
    y1.dataframe(yopp_by_course, hide_index=True, use_container_width=True)
    fig_yopp = px.bar(yopp_by_course, x="Percurso", y="Óculos vendidos", text="Óculos vendidos")
    fig_yopp = style_bar_labels(fig_yopp)
    fig_yopp.update_layout(height=340)
    y2.plotly_chart(fig_yopp, use_container_width=True)

    st.subheader("Top 5 cidades (Yopp)")
    top_cities = build_top_cities_table(
        yopp_buyers,
        ibge_df=ibge_df,
        total=yopp_total,
        count_col="Óculos vendidos",
        pct_col="% do total Yopp",
    )
    if top_cities.empty:
        st.info("Sem cidades válidas para montar ranking de Yopp.")
    else:
        st.dataframe(top_cities, hide_index=True, use_container_width=True)


def render_nubank_section(df: pd.DataFrame, ibge_df: pd.DataFrame) -> None:
    st.header("Nubank")
    if df.empty:
        st.info("Sem dados para análise de Nubank no recorte atual.")
        return

    if "nubank_flag" not in df.columns:
        st.info("Coluna `nubank_opt` não disponível nesta base.")
        return

    if "source_file" not in df.columns:
        st.info("Não foi possível identificar o arquivo de origem para aplicar o recorte BRL_FULL.")
        return

    br_full = df[df["source_file"].astype(str).str.contains(r"brl[\s_-]*full", case=False, na=False, regex=True)].copy()
    if br_full.empty:
        st.info("Nenhum registro BRL_FULL encontrado no upload atual.")
        return

    nubank_buyers = br_full[br_full["nubank_flag"] > 0].copy()
    nubank_total = len(nubank_buyers)
    total_br_full = len(br_full)
    nubank_pct = (nubank_total / total_br_full * 100) if total_br_full else 0

    female_nubank = count_female_entries(nubank_buyers)
    male_nubank = count_male_entries(nubank_buyers)
    valid_age = nubank_buyers["age"].dropna() if "age" in nubank_buyers.columns else pd.Series(dtype="float64")
    avg_age = valid_age.mean() if not valid_age.empty else 0
    min_age = valid_age.min() if not valid_age.empty else 0
    max_age = valid_age.max() if not valid_age.empty else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Inscritos BRL_FULL", format_int(total_br_full))
    c2.metric("Compraram com Nubank", format_int(nubank_total))
    c3.metric("% BRL_FULL com Nubank", format_pct(nubank_pct))
    c4.metric("Mulheres / Homens", f"{format_int(female_nubank)} / {format_int(male_nubank)}")

    a1, a2, a3 = st.columns(3)
    a1.metric("Idade média (Nubank)", format_int(avg_age))
    a2.metric("Idade mínima", format_int(min_age))
    a3.metric("Idade máxima", format_int(max_age))

    if nubank_buyers.empty:
        st.info("Nenhum inscrito BRL_FULL comprou com desconto Nubank no recorte atual.")
        return

    st.subheader("Perfil de quem comprou com desconto Nubank (BRL_FULL)")
    p1, p2 = st.columns(2)

    gender_profile = pd.DataFrame(
        {
            "Perfil": ["Feminino", "Masculino"],
            "Inscritos": [female_nubank, male_nubank],
        }
    )
    gender_total = female_nubank + male_nubank
    gender_profile["%"] = gender_profile["Inscritos"].apply(
        lambda v: format_pct((v / gender_total) * 100 if gender_total else 0)
    )
    p1.dataframe(gender_profile, hide_index=True, use_container_width=True)

    if not valid_age.empty:
        age_labels = ["18-24", "25-34", "35-44", "45-54", "55+"]
        age_bins = pd.cut(
            valid_age,
            bins=[17, 24, 34, 44, 54, 120],
            labels=age_labels,
            include_lowest=True,
        )
        age_counts = (
            age_bins.value_counts()
            .reindex(age_labels, fill_value=0)
            .rename_axis("Faixa etária")
            .reset_index(name="Inscritos")
        )
        age_counts["%"] = age_counts["Inscritos"].apply(
            lambda v: format_pct((v / nubank_total) * 100 if nubank_total else 0)
        )
        p2.dataframe(age_counts, hide_index=True, use_container_width=True)
    else:
        p2.info("Sem idades válidas para montar faixas etárias.")

    st.subheader("Compras Nubank por percurso")
    by_route = (
        nubank_buyers["Competition"]
        .fillna("NA")
        .value_counts()
        .rename_axis("Percurso")
        .reset_index(name="Inscritos")
    )
    order_map = {name: idx for idx, name in enumerate(PERCURSO_ORDER)}
    by_route["ordem"] = by_route["Percurso"].map(order_map).fillna(999).astype(int)
    by_route = by_route.sort_values(["ordem", "Percurso"]).drop(columns="ordem").reset_index(drop=True)
    by_route["% do total Nubank"] = by_route["Inscritos"].apply(
        lambda v: format_pct((v / nubank_total) * 100 if nubank_total else 0)
    )
    r1, r2 = st.columns(2)
    r1.dataframe(by_route, hide_index=True, use_container_width=True)
    fig_route = px.bar(by_route, x="Percurso", y="Inscritos", text="Inscritos")
    fig_route = style_bar_labels(fig_route)
    fig_route.update_layout(height=320)
    r2.plotly_chart(fig_route, use_container_width=True)

    st.subheader("Top 5 cidades (compras com desconto Nubank)")
    top_cities = build_top_cities_table(
        nubank_buyers,
        ibge_df=ibge_df,
        total=nubank_total,
        count_col="Inscritos",
        pct_col="% do total Nubank",
    )
    if top_cities.empty:
        st.info("Sem cidades válidas para montar ranking de Nubank.")
    else:
        st.dataframe(top_cities, hide_index=True, use_container_width=True)


def render_financial(df: pd.DataFrame) -> None:
    st.header("Financeiro")
    gross = df["total_registration_brl"].sum()
    discounts = df["total_discounts_brl"].sum()
    net = df["net_revenue_brl"].sum()
    c1, c2, c3 = st.columns(3)
    c1.metric("Receita bruta", format_currency(gross))
    c2.metric("Descontos", format_currency(discounts))
    c3.metric("Receita líquida", format_currency(net))

    comp = (
        df.groupby("Competition", dropna=False)
        .agg(
            inscritos=("Competition", "size"),
            receita_bruta=("total_registration_brl", "sum"),
            descontos=("total_discounts_brl", "sum"),
            receita_liquida=("net_revenue_brl", "sum"),
        )
        .reset_index()
    )
    comp = sort_competitions(comp, "Competition")
    comp_table = comp.rename(
        columns={
            "Competition": "Percurso",
            "inscritos": "Inscritos",
            "receita_bruta": "Receita Bruta (R$)",
            "descontos": "Descontos (R$)",
            "receita_liquida": "Receita Líquida (R$)",
        }
    )
    for col in ["Inscritos", "Receita Bruta (R$)", "Descontos (R$)", "Receita Líquida (R$)"]:
        comp_table[col] = comp_table[col].map(format_int)
    st.dataframe(comp_table, hide_index=True, use_container_width=True)

    if not comp.empty:
        st.caption("% de contribuição em receita de cada percurso")
        fig = px.pie(comp, names="Competition", values="receita_liquida", hole=0.45)
        fig.update_layout(height=360)
        st.plotly_chart(fig, use_container_width=True)


def render_financial_report(df: pd.DataFrame) -> None:
    st.header("Visão consolidada financeira")
    gross = float(df["total_registration_brl"].sum())
    discounts = float(df["total_discounts_brl"].sum())
    net = float(df["net_revenue_brl"].sum())
    gross_brl, gross_usd_brl, _ = revenue_split_by_currency(df, "total_registration_brl")
    net_brl, net_usd_brl, _ = revenue_split_by_currency(df, "net_revenue_brl")
    extras = compute_additional_revenue(df)
    total_orders = len(df)
    discount_rate = (discounts / gross * 100) if gross else 0
    orders_with_discount = int(df["total_discounts_brl"].gt(0).sum())
    pct_orders_with_discount = (orders_with_discount / total_orders * 100) if total_orders else 0

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Receita bruta total", format_currency(gross))
    k2.metric("Receita líquida total", format_currency(net))
    k3.metric("Receita bruta BRL", format_currency(gross_brl))
    k4.metric("Receita bruta USD (R$)", format_currency(gross_usd_brl))
    k5, k6 = st.columns(2)
    k5.metric("Taxa de desconto total", format_pct(discount_rate))
    k6.metric("Inscrições com cupom/desconto", format_pct(pct_orders_with_discount))
    k7, k8 = st.columns(2)
    k7.metric("Receita líquida BRL", format_currency(net_brl))
    k8.metric("Receita líquida USD (R$)", format_currency(net_usd_brl))

    st.subheader("Receitas adicionais (fora da inscrição)")
    a1, a2, a3 = st.columns(3)
    a1.metric("Yopp total (R$)", format_currency(extras["yopp_total"]))
    a2.metric("Yopp BRL / USD (R$)", f"{format_currency(extras['yopp_brl'])} / {format_currency(extras['yopp_usd_brl'])}")
    a3.metric("Preço Yopp", f"R$ {YOPP_PRICE_BRL:,.0f} | USD {YOPP_PRICE_USD:,.0f}")
    b1, b2, b3 = st.columns(3)
    b1.metric("Ônibus PTR 17 total (R$)", format_currency(extras["bus_ptr17_total"]))
    b2.metric(
        "Ônibus BRL / USD (R$)",
        f"{format_currency(extras['bus_ptr17_brl'])} / {format_currency(extras['bus_ptr17_usd_brl'])}",
    )
    b3.metric("Preço Ônibus PTR 17", f"R$ {PTR17_BUS_PRICE_BRL:,.0f} | USD {PTR17_BUS_PRICE_USD:,.0f}")
    st.metric("Total receitas adicionais", format_currency(extras["extras_total"]))

    st.subheader("Performance por percurso")
    by_route = (
        df.groupby("Competition", dropna=False)
        .agg(
            inscritos=("Competition", "size"),
            receita_bruta=("total_registration_brl", "sum"),
            descontos=("total_discounts_brl", "sum"),
            receita_liquida=("net_revenue_brl", "sum"),
        )
        .reset_index()
    )
    by_route = sort_competitions(by_route, "Competition")
    by_route["taxa_desconto"] = by_route.apply(
        lambda row: (row["descontos"] / row["receita_bruta"] * 100) if row["receita_bruta"] else 0,
        axis=1,
    )

    route_table = by_route.rename(
        columns={
            "Competition": "Percurso",
            "inscritos": "Inscritos",
            "receita_bruta": "Receita Bruta (R$)",
            "descontos": "Descontos (R$)",
            "receita_liquida": "Receita Líquida (R$)",
            "taxa_desconto": "Taxa de Desconto",
        }
    )
    route_table["Inscritos"] = route_table["Inscritos"].map(format_int)
    for col in ["Receita Bruta (R$)", "Descontos (R$)", "Receita Líquida (R$)"]:
        route_table[col] = route_table[col].map(format_currency)
    route_table["Taxa de Desconto"] = route_table["Taxa de Desconto"].map(format_pct)
    st.dataframe(route_table, hide_index=True, use_container_width=True)

    g1, g2 = st.columns(2)
    fig_net_route = px.bar(
        by_route,
        x="Competition",
        y="receita_liquida",
        text="receita_liquida",
        title="Receita líquida por percurso",
    )
    fig_net_route = style_bar_labels(fig_net_route)
    fig_net_route.update_traces(texttemplate="%{text:,.0f}")
    fig_net_route.update_layout(height=340)
    g1.plotly_chart(fig_net_route, use_container_width=True)

    fig_discount_route = px.bar(
        by_route,
        x="Competition",
        y="taxa_desconto",
        text="taxa_desconto",
        title="Taxa de desconto por percurso",
    )
    fig_discount_route = style_bar_labels(fig_discount_route)
    fig_discount_route.update_traces(texttemplate="%{text:.1f}%")
    fig_discount_route.update_layout(height=340, yaxis_title="%")
    g2.plotly_chart(fig_discount_route, use_container_width=True)

    st.subheader("Evolução diária de faturamento")
    dated = df.dropna(subset=["Registration date"]).copy()
    if dated.empty:
        st.info("Sem datas válidas para evolução diária financeira.")
    else:
        dated["day"] = dated["Registration date"].dt.date
        daily = (
            dated.groupby("day")
            .agg(
                receita_bruta=("total_registration_brl", "sum"),
                descontos=("total_discounts_brl", "sum"),
                receita_liquida=("net_revenue_brl", "sum"),
                inscritos=("day", "size"),
            )
            .reset_index()
            .sort_values("day")
        )
        fig_daily = px.line(
            daily,
            x="day",
            y=["receita_bruta", "descontos", "receita_liquida"],
            labels={"day": "Data", "value": "Valor (R$)", "variable": "Série"},
        )
        fig_daily.update_layout(height=360)
        st.plotly_chart(fig_daily, use_container_width=True)

    st.subheader("Estudo de cupons de desconto")
    coupon_code_clean = df["coupon_code"].fillna("").astype(str).str.strip()
    coupon_code_clean = coupon_code_clean.replace({"nan": "", "None": "", "none": ""})
    coupon_code_len = coupon_code_clean.str.len()
    coupons = df[(coupon_code_len > 2) & (df["total_discounts_brl"] > 0)].copy()
    if coupons.empty:
        st.info(
            "Nenhuma inscrição com cupom válido (3+ caracteres) e desconto > 0 no recorte atual."
        )
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Inscrições com cupom (>=3) e desconto", format_int(len(coupons)))
    c2.metric("Total de desconto (R$)", format_currency(coupons["total_discounts_brl"].sum()))
    c3.metric("Prefixos únicos (3 letras)", format_int(coupons["coupon_prefix"].nunique()))

    by_family = (
        coupons.groupby("coupon_family", dropna=False)
        .agg(
            inscricoes=("coupon_family", "size"),
            desconto_total=("total_discounts_brl", "sum"),
            receita_liquida=("net_revenue_brl", "sum"),
        )
        .reset_index()
        .sort_values("desconto_total", ascending=False)
    )
    by_family["desconto_medio"] = by_family.apply(
        lambda row: row["desconto_total"] / row["inscricoes"] if row["inscricoes"] else 0,
        axis=1,
    )
    by_family["% do desconto total"] = by_family["desconto_total"].apply(
        lambda value: format_pct((value / coupons["total_discounts_brl"].sum() * 100) if coupons["total_discounts_brl"].sum() else 0)
    )

    table_family = by_family.rename(
        columns={
            "coupon_family": "Família de cupom",
            "inscricoes": "Inscrições",
            "desconto_total": "Desconto Total (R$)",
            "receita_liquida": "Receita Líquida (R$)",
            "desconto_medio": "Desconto Médio (R$)",
        }
    )
    table_family["Inscrições"] = table_family["Inscrições"].map(format_int)
    for col in ["Desconto Total (R$)", "Receita Líquida (R$)", "Desconto Médio (R$)"]:
        table_family[col] = table_family[col].map(format_currency)
    st.dataframe(table_family, hide_index=True, use_container_width=True)

    top_prefix = (
        coupons.groupby("coupon_prefix", dropna=False)
        .agg(
            inscricoes=("coupon_prefix", "size"),
            desconto_total=("total_discounts_brl", "sum"),
        )
        .reset_index()
        .sort_values("desconto_total", ascending=False)
        .head(12)
    )
    fig_prefix = px.bar(
        top_prefix,
        x="coupon_prefix",
        y="desconto_total",
        text="desconto_total",
        title="Top prefixos por desconto total",
    )
    fig_prefix = style_bar_labels(fig_prefix)
    fig_prefix.update_traces(texttemplate="%{text:,.0f}")
    fig_prefix.update_layout(height=330, xaxis_title="Prefixo (3 letras)", yaxis_title="Desconto total (R$)")
    st.plotly_chart(fig_prefix, use_container_width=True)

    st.caption(
        "Este estudo considera cupom com 3+ caracteres e desconto > 0. "
        "Classificação atual usa `COUPON_PREFIX_RULES`."
    )


def render_marketing_coupon_block(df: pd.DataFrame) -> None:
    st.header("CUPOM DE DESCONTOS")
    coupon_code_clean = df["coupon_code"].fillna("").astype(str).str.strip()
    coupon_code_clean = coupon_code_clean.replace({"nan": "", "None": "", "none": ""})
    coupon_code_norm = coupon_code_clean.str.upper().str.replace(r"[^A-Z0-9]", "", regex=True)
    coupons = df[coupon_code_norm.str.len().ge(3)].copy()
    if coupons.empty:
        st.info("Nenhum cupom válido (3+ caracteres) encontrado no recorte atual.")
        return

    coupons["coupon_code_norm"] = coupon_code_norm.loc[coupons.index]
    coupons["coupon_category"] = coupons["coupon_code_norm"].str[:3]

    total_coupons_used = int(len(coupons))
    unique_coupons = int(coupons["coupon_code_norm"].nunique())
    unique_categories = int(coupons["coupon_category"].nunique())
    brl_used = int(coupons["edition_currency"].eq("BRL").sum())
    usd_used = int(coupons["edition_currency"].eq("USD").sum())

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Cupons usados (inscrições)", format_int(total_coupons_used))
    m2.metric("Cupons únicos", format_int(unique_coupons))
    m3.metric("Categorias únicas (3 letras)", format_int(unique_categories))
    m4.metric("Uso em BRL", format_int(brl_used))
    m5.metric("Uso em USD", format_int(usd_used))

    by_currency = (
        coupons.groupby("edition_currency", dropna=False)
        .agg(usos=("edition_currency", "size"))
        .reset_index()
        .sort_values("usos", ascending=False)
    )
    by_currency["% do uso"] = by_currency["usos"].apply(
        lambda value: format_pct((value / total_coupons_used * 100) if total_coupons_used else 0)
    )
    currency_table = by_currency.rename(
        columns={
            "edition_currency": "Planilha",
            "usos": "Cupons usados",
        }
    )
    currency_table["Planilha"] = currency_table["Planilha"].replace({"BRL": "BRL", "USD": "US"})
    currency_table["Cupons usados"] = currency_table["Cupons usados"].map(format_int)
    st.subheader("Uso por planilha (BRL e US)")
    st.dataframe(currency_table, hide_index=True, use_container_width=True)

    by_category = (
        coupons.groupby("coupon_category", dropna=False)
        .agg(
            usos=("coupon_category", "size"),
            cupons_unicos=("coupon_code_norm", "nunique"),
        )
        .reset_index()
        .sort_values("usos", ascending=False)
    )
    by_category["% do uso"] = by_category["usos"].apply(
        lambda value: format_pct((value / total_coupons_used * 100) if total_coupons_used else 0)
    )
    category_table = by_category.rename(
        columns={
            "coupon_category": "Categoria (3 letras)",
            "usos": "Cupons usados",
            "cupons_unicos": "Cupons únicos",
        }
    )
    category_table["Cupons usados"] = category_table["Cupons usados"].map(format_int)
    category_table["Cupons únicos"] = category_table["Cupons únicos"].map(format_int)
    st.subheader("Uso por tipo/categoria de cupom")
    st.dataframe(category_table, hide_index=True, use_container_width=True)

    by_route = (
        coupons.groupby("Competition", dropna=False)
        .agg(
            usos=("Competition", "size"),
            categorias=("coupon_category", "nunique"),
        )
        .reset_index()
    )
    by_route = sort_competitions(by_route, "Competition")
    by_route["% do uso"] = by_route["usos"].apply(
        lambda value: format_pct((value / total_coupons_used * 100) if total_coupons_used else 0)
    )
    route_table = by_route.rename(
        columns={
            "Competition": "Percurso",
            "usos": "Cupons usados",
            "categorias": "Categorias ativas",
        }
    )
    route_table["Cupons usados"] = route_table["Cupons usados"].map(format_int)
    route_table["Categorias ativas"] = route_table["Categorias ativas"].map(format_int)
    st.subheader("Uso por percurso")
    st.dataframe(route_table, hide_index=True, use_container_width=True)

    by_category_route = (
        coupons.groupby(["coupon_category", "Competition"], dropna=False)
        .agg(usos=("coupon_category", "size"))
        .reset_index()
        .sort_values("usos", ascending=False)
        .head(20)
    )
    by_category_route = sort_competitions(by_category_route, "Competition")
    cruzamento_table = by_category_route.rename(
        columns={
            "coupon_category": "Categoria (3 letras)",
            "Competition": "Percurso",
            "usos": "Cupons usados",
        }
    )
    cruzamento_table["Cupons usados"] = cruzamento_table["Cupons usados"].map(format_int)
    st.subheader("Cruzamento categoria x percurso (Top 20)")
    st.dataframe(cruzamento_table, hide_index=True, use_container_width=True)

    fig_category = px.bar(
        by_category.head(15),
        x="coupon_category",
        y="usos",
        text="usos",
        title="Top categorias de cupom por quantidade de uso",
    )
    fig_category = style_bar_labels(fig_category)
    fig_category.update_traces(texttemplate="%{text:,.0f}")
    fig_category.update_layout(height=320, xaxis_title="Categoria (3 letras)", yaxis_title="Quantidade")
    st.plotly_chart(fig_category, use_container_width=True)

    st.caption(
        "Análise gerencial baseada na coluna `Discount codes` (BRL e US). "
        "Categoria definida pelas 3 primeiras letras do cupom."
    )


def parse_brl_currency_series(series: pd.Series) -> pd.Series:
    txt = (
        series.fillna("")
        .astype(str)
        .str.replace("R$", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
    )
    txt = txt.replace({"": "0", "-": "0", "nan": "0", "None": "0"})
    return pd.to_numeric(txt, errors="coerce").fillna(0.0)


def parse_mercado_pago_datetime_series(series: pd.Series) -> pd.Series:
    month_map = {
        "jan": 1,
        "fev": 2,
        "mar": 3,
        "abr": 4,
        "mai": 5,
        "jun": 6,
        "jul": 7,
        "ago": 8,
        "set": 9,
        "out": 10,
        "nov": 11,
        "dez": 12,
    }

    parts = series.fillna("").astype(str).str.lower().str.extract(
        r"(?P<day>\d{1,2})\s+(?P<month>[a-z]{3})\s+(?P<hour>\d{1,2}):(?P<minute>\d{2})"
    )
    parsed: list[pd.Timestamp] = []
    current_year = date.today().year
    rolling_year = current_year
    prev_dt: datetime | None = None

    for _, row in parts.iterrows():
        if row.isna().any():
            parsed.append(pd.NaT)
            continue
        month = month_map.get(str(row["month"]).strip())
        if month is None:
            parsed.append(pd.NaT)
            continue

        day = int(row["day"])
        hour = int(row["hour"])
        minute = int(row["minute"])

        try:
            candidate = datetime(rolling_year, month, day, hour, minute)
        except ValueError:
            parsed.append(pd.NaT)
            continue

        # Export do MP costuma vir em ordem mais recente -> mais antiga.
        # Quando a data "volta para frente", assume virada de ano.
        if prev_dt is not None and candidate > prev_dt:
            rolling_year -= 1
            try:
                candidate = datetime(rolling_year, month, day, hour, minute)
            except ValueError:
                parsed.append(pd.NaT)
                continue

        prev_dt = candidate
        parsed.append(pd.Timestamp(candidate))

    return pd.Series(parsed, index=series.index)


def classify_mp_payment_family(value) -> str:
    txt = str(value).strip().lower()
    if not txt or txt in {"nan", "none"}:
        return "Outros"
    if "pix" in txt:
        return "PIX"
    if "boleto" in txt:
        return "Boleto"
    if "nubank" in txt:
        return "Nubank"
    if "mastercard" in txt:
        return "Mastercard"
    if "visa" in txt:
        return "Visa"
    if "elo" in txt:
        return "Elo"
    if "saldo" in txt and "mercado pago" in txt:
        return "Saldo MP"
    return "Outros"


@st.cache_data(show_spinner=False)
def load_mercado_pago_file(uploaded_file) -> pd.DataFrame:
    df = pd.read_excel(uploaded_file, sheet_name=0, header=3)
    df = normalize_headers(df)
    df = df.dropna(how="all").copy()

    for col in [
        "Recebimento",
        "Tarifas e impostos",
        "Cancelamentos e reembolsos",
        "Total a receber",
    ]:
        if col in df.columns:
            df[f"{col}_num"] = parse_brl_currency_series(df[col])
        else:
            df[f"{col}_num"] = 0.0

    if "Data da transação" in df.columns:
        df["data_transacao_dt"] = parse_mercado_pago_datetime_series(df["Data da transação"])
    else:
        df["data_transacao_dt"] = pd.NaT

    weekday_map = {
        0: "Segunda",
        1: "Terça",
        2: "Quarta",
        3: "Quinta",
        4: "Sexta",
        5: "Sábado",
        6: "Domingo",
    }
    df["hora_transacao"] = df["data_transacao_dt"].dt.hour
    df["dia_semana_num"] = df["data_transacao_dt"].dt.dayofweek
    df["dia_semana"] = df["dia_semana_num"].map(weekday_map)
    df["dia_data"] = df["data_transacao_dt"].dt.date

    if "Meio de pagamento" in df.columns:
        df["meio_pagamento_familia"] = df["Meio de pagamento"].apply(classify_mp_payment_family)
    else:
        df["meio_pagamento_familia"] = "Outros"

    return df


def render_dashboard_mercado_pago(df_mp: pd.DataFrame, source_name: str) -> None:
    st.markdown(
        f"""
        <div class="hero">
          <h2>Dashboard Mercado Pago - Vendas e Recebimentos</h2>
          <p class="subtle">Arquivo: {source_name} | Foco no que foi efetivamente vendido/recebido no MP</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if df_mp.empty:
        st.warning("Arquivo do Mercado Pago sem dados para análise.")
        return

    available_status = sorted(df_mp["Estado"].dropna().astype(str).unique().tolist()) if "Estado" in df_mp.columns else []
    default_status = ["Aprovado"] if "Aprovado" in available_status else available_status
    available_payment_families = (
        sorted(df_mp["meio_pagamento_familia"].dropna().astype(str).unique().tolist())
        if "meio_pagamento_familia" in df_mp.columns
        else []
    )
    filter_col_1, filter_col_2 = st.columns(2)
    with filter_col_1:
        selected_status = st.multiselect(
            "Filtrar por estado da venda",
            options=available_status,
            default=default_status,
        )
    with filter_col_2:
        selected_payment_families = st.multiselect(
            "Filtrar por tipo de pagamento",
            options=available_payment_families,
            default=available_payment_families,
        )

    scoped_base = df_mp.copy()
    if selected_payment_families and "meio_pagamento_familia" in scoped_base.columns:
        scoped_base = scoped_base[scoped_base["meio_pagamento_familia"].astype(str).isin(selected_payment_families)].copy()
    if "data_transacao_dt" in scoped_base.columns and scoped_base["data_transacao_dt"].notna().any():
        min_date = scoped_base["data_transacao_dt"].dt.date.min()
        max_date = scoped_base["data_transacao_dt"].dt.date.max()
        date_range = st.date_input(
            "Período da análise",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date, end_date = date_range
            scoped_base = scoped_base[
                scoped_base["data_transacao_dt"].dt.date.between(start_date, end_date, inclusive="both")
            ].copy()

    scoped = scoped_base.copy()
    if selected_status and "Estado" in scoped.columns:
        scoped = scoped[scoped["Estado"].astype(str).isin(selected_status)].copy()

    if scoped.empty:
        st.info("Sem registros para os filtros selecionados.")
        return

    approved_scope = get_mp_approved_scope(scoped_base)

    if approved_scope.empty:
        st.info("Sem vendas com status Aprovado para o período/filtros selecionados.")
        return

    gross = float(approved_scope["Recebimento_num"].sum())
    fee_signed = float(approved_scope["Tarifas e impostos_num"].sum())
    fee_cost = abs(fee_signed)
    refunds = float(approved_scope["Cancelamentos e reembolsos_num"].sum())
    net = float(approved_scope["Total a receber_num"].sum())
    orders = len(approved_scope)
    selected_orders = len(scoped)
    total_orders_base = len(scoped_base)
    avg_ticket = net / orders if orders else 0
    fee_rate = (fee_cost / gross * 100) if gross else 0
    gross_target_pct = (gross / MP_GROSS_TARGET_BRL * 100) if MP_GROSS_TARGET_BRL else 0
    approval_rate = (
        (approved_scope.shape[0] / total_orders_base * 100)
        if total_orders_base
        else 0
    )
    refund_count = int(approved_scope["Cancelamentos e reembolsos_num"].lt(0).sum())
    pix_share = (
        (approved_scope["meio_pagamento_familia"].eq("PIX").sum() / orders * 100)
        if orders and "meio_pagamento_familia" in approved_scope.columns
        else 0
    )
    card_families = {"Nubank", "Mastercard", "Visa", "Elo"}
    card_share = (
        (approved_scope["meio_pagamento_familia"].isin(card_families).sum() / orders * 100)
        if orders and "meio_pagamento_familia" in approved_scope.columns
        else 0
    )

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Transações aprovadas", format_int(orders))
    c2.metric("Bruto recebido", format_currency(gross))
    c3.metric("Custo MP (tarifas)", format_currency(fee_cost))
    c4.metric("Líquido recebido", format_currency(net))
    c5.metric("Reembolsos/cancelamentos", format_currency(refunds))
    c6.metric("Ticket médio líquido", format_currency(avg_ticket))
    meta_col_1, meta_col_2 = st.columns([2, 1])
    with meta_col_1:
        st.metric(
            "Meta bruto Mercado Pago",
            format_currency(gross),
            delta=f"{format_pct(gross_target_pct)} da meta de {format_currency(MP_GROSS_TARGET_BRL)}",
        )
        st.progress(max(min(gross_target_pct / 100, 1), 0))
    with meta_col_2:
        fig_meta = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=gross,
                number={"prefix": "R$ ", "valueformat": ",.0f"},
                gauge={
                    "axis": {"range": [0, MP_GROSS_TARGET_BRL]},
                    "bar": {"color": "#2563eb"},
                    "threshold": {
                        "line": {"color": "#ef4444", "width": 4},
                        "thickness": 0.9,
                        "value": MP_GROSS_TARGET_BRL,
                    },
                },
                title={"text": "Marcador visual da meta bruta"},
            )
        )
        fig_meta.update_layout(height=220, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig_meta, use_container_width=True)

    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Taxa média de comissão", format_pct(fee_rate))
    e2.metric("Taxa de aprovação (filtro atual)", format_pct(approval_rate))
    e3.metric("Participação PIX", format_pct(pix_share))
    e4.metric("Participação cartão", format_pct(card_share))
    st.caption(
        "Faturamento considera sempre apenas status Aprovado. "
        f"Registros no filtro de status: {format_int(selected_orders)} | "
        f"Registros totais no período/filtros: {format_int(total_orders_base)} | "
        f"Qtde com cancelamento/reembolso (aprovadas): {format_int(refund_count)}"
    )

    if st.session_state.get("print_mode", False):
        st.info("Modo impressão ativo: ao exportar, as três abas serão incluídas no PDF.")

    tab_exec, tab_fin, tab_ops = st.tabs(
        ["Resumo executivo", "Financeiro detalhado", "Comportamento de vendas"]
    )

    with tab_exec:
        st.subheader("Leitura executiva")
        refund_rate = (refund_count / orders * 100) if orders else 0

        def semaforo_indicator(value: float, good_limit: float, warn_limit: float, reverse: bool = False) -> str:
            if reverse:
                if value >= good_limit:
                    return "Verde"
                if value >= warn_limit:
                    return "Amarelo"
                return "Vermelho"
            if value <= good_limit:
                return "Verde"
            if value <= warn_limit:
                return "Amarelo"
            return "Vermelho"

        semaforo = pd.DataFrame(
            [
                {
                    "Indicador": "Taxa de comissão",
                    "Valor": format_pct(fee_rate),
                    "Status": semaforo_indicator(fee_rate, good_limit=5.0, warn_limit=7.0),
                    "Meta sugerida": "<= 5,00%",
                },
                {
                    "Indicador": "Taxa de aprovação",
                    "Valor": format_pct(approval_rate),
                    "Status": semaforo_indicator(approval_rate, good_limit=90.0, warn_limit=80.0, reverse=True),
                    "Meta sugerida": ">= 90,00%",
                },
                {
                    "Indicador": "Taxa de reembolso (qtd)",
                    "Valor": format_pct(refund_rate),
                    "Status": semaforo_indicator(refund_rate, good_limit=1.0, warn_limit=3.0),
                    "Meta sugerida": "<= 1,00%",
                },
                {
                    "Indicador": "Participação PIX",
                    "Valor": format_pct(pix_share),
                    "Status": semaforo_indicator(pix_share, good_limit=35.0, warn_limit=20.0, reverse=True),
                    "Meta sugerida": ">= 35,00%",
                },
            ]
        )
        st.dataframe(semaforo, hide_index=True, use_container_width=True)

        red_count = int(semaforo["Status"].eq("Vermelho").sum())
        yellow_count = int(semaforo["Status"].eq("Amarelo").sum())
        if red_count > 0:
            st.error(f"Semáforo: {red_count} indicador(es) crítico(s) e {yellow_count} em atenção.")
        elif yellow_count > 0:
            st.warning(f"Semáforo: {yellow_count} indicador(es) em atenção, sem críticos.")
        else:
            st.success("Semáforo: indicadores dentro do intervalo esperado.")

        insights = []
        if fee_rate >= 8:
            insights.append(
                f"Custo de comissão elevado ({format_pct(fee_rate)}). Vale revisar mix de meios de pagamento."
            )
        elif fee_rate <= 4:
            insights.append(
                f"Comissão em patamar saudável ({format_pct(fee_rate)})."
            )
        if pix_share >= 45:
            insights.append(
                f"PIX representa parcela relevante das vendas ({format_pct(pix_share)}), ajudando custo financeiro."
            )
        if refund_count > 0:
            insights.append(
                f"Foram detectadas {format_int(refund_count)} transações com reembolso/cancelamento no período."
            )
        if not insights:
            insights.append("Não há alertas críticos no recorte atual.")

        for msg in insights:
            st.markdown(f"- {msg}")

        action_rows: list[dict[str, str]] = []
        if fee_rate > 7:
            action_rows.append(
                {
                    "Prioridade": "Alta",
                    "Frente": "Custo financeiro",
                    "Ação recomendada": "Aumentar participação de PIX e revisar condições de parcelamento/captura com maior tarifa.",
                    "Impacto esperado": "Reduzir custo de comissão e elevar margem líquida.",
                }
            )
        if approval_rate < 80:
            action_rows.append(
                {
                    "Prioridade": "Alta",
                    "Frente": "Conversão de pagamento",
                    "Ação recomendada": "Analisar causas de recusa por estado e reforçar meios alternativos (PIX/Boleto) na jornada.",
                    "Impacto esperado": "Recuperar vendas perdidas por baixa aprovação.",
                }
            )
        if refund_rate > 3:
            action_rows.append(
                {
                    "Prioridade": "Média",
                    "Frente": "Qualidade de venda",
                    "Ação recomendada": "Auditar itens/campanhas com maior reembolso e revisar comunicação de oferta.",
                    "Impacto esperado": "Menor devolução e maior previsibilidade de caixa.",
                }
            )
        if pix_share < 20:
            action_rows.append(
                {
                    "Prioridade": "Média",
                    "Frente": "Mix de pagamento",
                    "Ação recomendada": "Criar incentivo comercial para PIX (desconto controlado ou destaque visual no checkout).",
                    "Impacto esperado": "Aumentar conversão com menor custo por transação.",
                }
            )
        if not action_rows:
            action_rows.append(
                {
                    "Prioridade": "Monitorar",
                    "Frente": "Performance geral",
                    "Ação recomendada": "Manter estratégia atual e acompanhar semanalmente os 4 indicadores do semáforo.",
                    "Impacto esperado": "Sustentar resultado com controle contínuo.",
                }
            )

        st.subheader("Ações recomendadas")
        st.dataframe(pd.DataFrame(action_rows), hide_index=True, use_container_width=True)

        split_col_1, split_col_2 = st.columns(2)
        financial_bridge = pd.DataFrame(
            {
                "etapa": ["Bruto", "Tarifas", "Reembolsos", "Líquido"],
                "valor": [gross, -fee_cost, refunds, net],
            }
        )
        fig_bridge = go.Figure(
            go.Waterfall(
                name="Fluxo financeiro",
                orientation="v",
                x=financial_bridge["etapa"],
                text=[format_currency(v) for v in financial_bridge["valor"]],
                y=financial_bridge["valor"],
                connector={"line": {"color": "rgb(63, 63, 63)"}},
            )
        )
        fig_bridge.update_layout(height=340, title="Ponte financeira: bruto até líquido")
        split_col_1.plotly_chart(fig_bridge, use_container_width=True)

        payment_mix = (
            approved_scope.groupby("meio_pagamento_familia", dropna=False)
            .agg(transacoes=("meio_pagamento_familia", "size"), liquido=("Total a receber_num", "sum"))
            .reset_index()
            .sort_values("transacoes", ascending=False)
        )
        fig_mix = px.bar(
            payment_mix,
            x="meio_pagamento_familia",
            y="transacoes",
            text="transacoes",
            title="Mix de meios de pagamento (quantidade)",
        )
        fig_mix = style_bar_labels(fig_mix)
        fig_mix.update_layout(height=340, yaxis_title="Transações", xaxis_title="Tipo")
        split_col_2.plotly_chart(fig_mix, use_container_width=True)

    with tab_fin:
        st.subheader("Análises financeiras")
        by_status = (
            scoped.groupby("Estado", dropna=False)
            .agg(
                transacoes=("Estado", "size"),
            )
            .reset_index()
            .sort_values("transacoes", ascending=False)
        )
        by_status["pct"] = by_status["transacoes"].apply(
            lambda value: (value / selected_orders * 100) if selected_orders else 0
        )

        status_table = by_status.rename(
            columns={
                "Estado": "Estado",
                "transacoes": "Transações",
                "pct": "% no filtro de status",
            }
        )
        status_table["Transações"] = status_table["Transações"].map(format_int)
        status_table["% no filtro de status"] = status_table["% no filtro de status"].map(format_pct)
        st.dataframe(
            status_table[["Estado", "Transações", "% no filtro de status"]],
            hide_index=True,
            use_container_width=True,
        )

        payment_value = (
            approved_scope.groupby("meio_pagamento_familia", dropna=False)
            .agg(
                transacoes=("meio_pagamento_familia", "size"),
                bruto=("Recebimento_num", "sum"),
                liquido=("Total a receber_num", "sum"),
                tarifas=("Tarifas e impostos_num", "sum"),
            )
            .reset_index()
            .sort_values("liquido", ascending=False)
        )
        payment_value["tarifas_abs"] = payment_value["tarifas"].abs()
        payment_value["taxa_comissao"] = payment_value.apply(
            lambda row: (row["tarifas_abs"] / row["bruto"] * 100) if row["bruto"] else 0,
            axis=1,
        )

        p1, p2 = st.columns(2)
        fig_payment_value = px.bar(
            payment_value,
            x="meio_pagamento_familia",
            y="liquido",
            text="liquido",
            title="Líquido por tipo de meio de pagamento",
        )
        fig_payment_value = style_bar_labels(fig_payment_value)
        fig_payment_value.update_traces(texttemplate="%{text:,.0f}")
        fig_payment_value.update_layout(height=340, xaxis_title="Tipo", yaxis_title="Líquido (R$)")
        p1.plotly_chart(fig_payment_value, use_container_width=True)

        fig_payment_qty = px.pie(
            payment_value,
            names="meio_pagamento_familia",
            values="transacoes",
            hole=0.45,
            title="Distribuição de transações por tipo",
        )
        fig_payment_qty.update_layout(height=340)
        p2.plotly_chart(fig_payment_qty, use_container_width=True)

        fee_table = payment_value.rename(
            columns={
                "meio_pagamento_familia": "Tipo de pagamento",
                "transacoes": "Transações",
                "bruto": "Bruto (R$)",
                "tarifas_abs": "Tarifas (R$)",
                "liquido": "Líquido (R$)",
                "taxa_comissao": "Taxa comissão",
            }
        )
        fee_table["Transações"] = fee_table["Transações"].map(format_int)
        for col in ["Bruto (R$)", "Tarifas (R$)", "Líquido (R$)"]:
            fee_table[col] = fee_table[col].map(format_currency)
        fee_table["Taxa comissão"] = fee_table["Taxa comissão"].map(format_pct)
        st.dataframe(
            fee_table[["Tipo de pagamento", "Transações", "Bruto (R$)", "Tarifas (R$)", "Líquido (R$)", "Taxa comissão"]],
            hide_index=True,
            use_container_width=True,
        )

    with tab_ops:
        st.subheader("Padrões de venda")
        dated = approved_scope.dropna(subset=["data_transacao_dt"]).copy()
        if dated.empty:
            st.info("Sem datas de transação válidas para os gráficos de horário e dia da semana.")
        else:
            st.subheader("Vendas aprovadas: dia a dia, dia da semana e hora a hora")
            daily_counts = (
                dated.groupby("dia_data")
                .agg(transacoes=("dia_data", "size"), bruto=("Recebimento_num", "sum"), liquido=("Total a receber_num", "sum"))
                .reset_index()
                .sort_values("dia_data")
            )
            t1, t2 = st.columns(2)
            fig_daily_qty = px.bar(
                daily_counts,
                x="dia_data",
                y="transacoes",
                text="transacoes",
                title="Dia a dia (quantidade de vendas aprovadas)",
            )
            fig_daily_qty = style_bar_labels(fig_daily_qty)
            fig_daily_qty.update_layout(height=320, xaxis_title="Data", yaxis_title="Transações")
            t1.plotly_chart(fig_daily_qty, use_container_width=True)

            fig_daily_gross = px.line(
                daily_counts,
                x="dia_data",
                y="bruto",
                markers=True,
                title="Dia a dia (faturamento bruto aprovado)",
            )
            fig_daily_gross.update_layout(height=320, xaxis_title="Data", yaxis_title="Bruto (R$)")
            t2.plotly_chart(fig_daily_gross, use_container_width=True)

            hour_counts = (
                dated.groupby("hora_transacao")
                .agg(transacoes=("hora_transacao", "size"), liquido=("Total a receber_num", "sum"))
                .reindex(range(24), fill_value=0)
                .reset_index()
                .rename(columns={"hora_transacao": "Hora"})
            )
            h1, h2 = st.columns(2)
            fig_hour_qty = px.bar(
                hour_counts, x="Hora", y="transacoes", text="transacoes", title="Vendas por hora (quantidade)"
            )
            fig_hour_qty = style_bar_labels(fig_hour_qty)
            fig_hour_qty.update_layout(height=330, xaxis=dict(dtick=1), yaxis_title="Transações")
            h1.plotly_chart(fig_hour_qty, use_container_width=True)

            fig_hour_value = px.bar(
                hour_counts, x="Hora", y="liquido", text="liquido", title="Vendas por hora (líquido)"
            )
            fig_hour_value = style_bar_labels(fig_hour_value)
            fig_hour_value.update_traces(texttemplate="%{text:,.0f}")
            fig_hour_value.update_layout(height=330, xaxis=dict(dtick=1), yaxis_title="Líquido (R$)")
            h2.plotly_chart(fig_hour_value, use_container_width=True)

            weekday_order = ["Segunda", "Terça", "Quarta", "Quinta", "Sexta", "Sábado", "Domingo"]
            weekday_counts = (
                dated.groupby("dia_semana")
                .agg(transacoes=("dia_semana", "size"), liquido=("Total a receber_num", "sum"))
                .reindex(weekday_order, fill_value=0)
                .reset_index()
                .rename(columns={"dia_semana": "Dia da semana"})
            )
            d1, d2 = st.columns(2)
            fig_weekday_qty = px.bar(
                weekday_counts,
                x="Dia da semana",
                y="transacoes",
                text="transacoes",
                title="Vendas por dia da semana (quantidade)",
            )
            fig_weekday_qty = style_bar_labels(fig_weekday_qty)
            fig_weekday_qty.update_layout(height=330, yaxis_title="Transações")
            d1.plotly_chart(fig_weekday_qty, use_container_width=True)

            fig_weekday_value = px.bar(
                weekday_counts,
                x="Dia da semana",
                y="liquido",
                text="liquido",
                title="Vendas por dia da semana (líquido)",
            )
            fig_weekday_value = style_bar_labels(fig_weekday_value)
            fig_weekday_value.update_traces(texttemplate="%{text:,.0f}")
            fig_weekday_value.update_layout(height=330, yaxis_title="Líquido (R$)")
            d2.plotly_chart(fig_weekday_value, use_container_width=True)

            # Heatmap para identificar horários fortes por dia da semana.
            heatmap_data = (
                dated.groupby(["dia_semana", "hora_transacao"])
                .size()
                .reset_index(name="transacoes")
                .pivot(index="dia_semana", columns="hora_transacao", values="transacoes")
                .reindex(index=weekday_order, fill_value=0)
                .fillna(0)
            )
            fig_heatmap = px.imshow(
                heatmap_data.values,
                x=list(heatmap_data.columns),
                y=list(heatmap_data.index),
                aspect="auto",
                labels={"x": "Hora", "y": "Dia da semana", "color": "Transações"},
                title="Mapa de calor: concentração de vendas",
            )
            fig_heatmap.update_layout(height=360)
            st.plotly_chart(fig_heatmap, use_container_width=True)

            daily_gross = (
                dated.groupby("dia_data")
                .agg(bruto=("Recebimento_num", "sum"))
                .reset_index()
                .sort_values("dia_data")
            )
            daily_gross["bruto_acumulado"] = daily_gross["bruto"].cumsum()
            fig_cumulative = px.line(
                daily_gross,
                x="dia_data",
                y="bruto_acumulado",
                markers=True,
                title="Evolução do bruto acumulado vs meta",
                labels={"dia_data": "Data", "bruto_acumulado": "Bruto acumulado (R$)"},
            )
            fig_cumulative.add_hline(
                y=MP_GROSS_TARGET_BRL,
                line_color="#ef4444",
                line_dash="dash",
                annotation_text=f"Meta R$ {format_int(MP_GROSS_TARGET_BRL)}",
                annotation_position="top left",
            )
            fig_cumulative.update_layout(height=360)
            st.plotly_chart(fig_cumulative, use_container_width=True)

            reached_target = daily_gross[daily_gross["bruto_acumulado"] >= MP_GROSS_TARGET_BRL]
            if not reached_target.empty:
                reached_date = reached_target.iloc[0]["dia_data"]
                st.success(f"Meta de bruto atingida em {reached_date.strftime('%d/%m/%Y')}.")
            else:
                remaining = MP_GROSS_TARGET_BRL - float(daily_gross["bruto_acumulado"].iloc[-1])
                st.info(f"Faltam {format_currency(remaining)} para atingir a meta de bruto.")

        if "Descrição do item" in scoped.columns:
            st.subheader("Top itens por faturamento líquido")
            top_items = (
                approved_scope.groupby("Descrição do item", dropna=False)
                .agg(transacoes=("Descrição do item", "size"), liquido=("Total a receber_num", "sum"))
                .reset_index()
                .sort_values("liquido", ascending=False)
                .head(10)
            )
            fig_items = px.bar(top_items, x="Descrição do item", y="liquido", text="liquido")
            fig_items = style_bar_labels(fig_items)
            fig_items.update_traces(texttemplate="%{text:,.0f}")
            fig_items.update_layout(height=360, xaxis_tickangle=-25, yaxis_title="Líquido (R$)")
            st.plotly_chart(fig_items, use_container_width=True)

    export_cols = [
        "Número da transação",
        "Data da transação",
        "data_transacao_dt",
        "Estado",
        "Meio de pagamento",
        "meio_pagamento_familia",
        "Recebimento_num",
        "Tarifas e impostos_num",
        "Cancelamentos e reembolsos_num",
        "Total a receber_num",
        "Descrição do item",
        "Referência externa",
    ]
    export_cols = [col for col in export_cols if col in scoped.columns]
    st.download_button(
        "Baixar base Mercado Pago processada (CSV)",
        data=scoped[export_cols].to_csv(index=False).encode("utf-8"),
        file_name="mercado_pago_base_processada.csv",
        mime="text/csv",
        use_container_width=True,
    )


def render_exports(raw_df: pd.DataFrame, kpi_df: pd.DataFrame) -> None:
    st.header("Exportações")
    export_df = kpi_df.copy()
    csv_bytes = export_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Baixar base processada (CSV)",
        data=csv_bytes,
        file_name="dashboard_2026_base_processada.csv",
        mime="text/csv",
        use_container_width=True,
    )

    summary = {
        "registros_upload": int(len(raw_df)),
        "registros_kpi": int(len(kpi_df)),
        "receita_liquida_brl": float(kpi_df["net_revenue_brl"].sum()),
        "ticket_medio_brl": float(kpi_df["net_revenue_brl"].mean() if len(kpi_df) else 0),
    }
    json_bytes = json.dumps(summary, indent=2, ensure_ascii=False).encode("utf-8")
    st.download_button(
        "Baixar resumo executivo (JSON)",
        data=json_bytes,
        file_name="dashboard_2026_resumo.json",
        mime="application/json",
        use_container_width=True,
    )


def main() -> None:
    apply_theme()
    st.sidebar.title("Configurações 2026")
    st.sidebar.caption("Dashboard único em Streamlit com upload manual BRL/USD.")
    st.sidebar.caption(f"Versão: {get_app_version_stamp()}")

    uploaded_files = st.sidebar.file_uploader(
        "Envie os arquivos 2026 (BR_FULL / US_FULL)",
        type=["xlsx"],
        accept_multiple_files=True,
    )
    st.sidebar.markdown("---")
    tipo_relatorio = st.sidebar.radio(
        "Selecione o relatório",
        options=[
            "Marketing Diário",
            "Marketing Semanal",
            "Marketing",
            "Geral",
            "Financeiro",
            "Mercado Pago",
        ],
        index=0,
        label_visibility="collapsed",
    )
    st.sidebar.markdown("### Arquivo acessório Mercado Pago")
    uploaded_mp_input = st.sidebar.file_uploader(
        "Arquivo de vendas Mercado Pago (xlsx)",
        type=["xlsx"],
        accept_multiple_files=False,
        key="mercado_pago_file",
    )
    if uploaded_mp_input is not None:
        st.session_state["mercado_pago_file_bytes"] = uploaded_mp_input.getvalue()
        st.session_state["mercado_pago_file_name"] = uploaded_mp_input.name

    uploaded_mp_file = uploaded_mp_input
    if (
        uploaded_mp_file is None
        and "mercado_pago_file_bytes" in st.session_state
        and "mercado_pago_file_name" in st.session_state
    ):
        uploaded_mp_file = BytesIO(st.session_state["mercado_pago_file_bytes"])
        uploaded_mp_file.name = st.session_state["mercado_pago_file_name"]

    percurso_targets = DEFAULT_PERCURSO_TARGETS.copy()
    start_date = date(2026, 2, 23)
    end_date = date(2026, 8, 14)
    if tipo_relatorio != "Mercado Pago":
        st.sidebar.markdown("### Metas por percurso")
        percurso_targets = {}
        for percurso in PERCURSO_ORDER:
            percurso_targets[percurso] = st.sidebar.number_input(
                f"Meta {percurso}",
                min_value=0,
                value=DEFAULT_PERCURSO_TARGETS[percurso],
                step=50,
            )
        start_date = st.sidebar.date_input("Início da campanha", value=date(2026, 2, 23))
        end_date = st.sidebar.date_input("Fim da campanha", value=date(2026, 8, 14))
    st.sidebar.markdown("### PDF")
    print_mode = st.sidebar.toggle(
        "Modo impressão otimizado (A4)",
        value=False,
        help="Ajusta largura, expande seções e melhora paginação para exportação em PDF.",
    )
    st.session_state["print_mode"] = print_mode
    report_mode_for_pdf: str | None = None
    if tipo_relatorio == "Marketing Diário":
        report_mode_for_pdf = "diario"
    elif tipo_relatorio == "Marketing Semanal":
        report_mode_for_pdf = "semanal"
    apply_print_css(print_mode, report_mode=report_mode_for_pdf)
    if print_mode:
        st.sidebar.caption("Recomendado: usar escala entre 95% e 100% no salvar como PDF do navegador.")
    print_button_label = "Gerar PDF (layout otimizado)" if print_mode else "Imprimir / Exportar PDF"
    if st.sidebar.button(print_button_label, use_container_width=True):
        components.html("<script>window.print();</script>", height=0, width=0)

    full_df: pd.DataFrame | None = None
    filtered: pd.DataFrame | None = None
    if uploaded_files:
        dfs = [preprocess_uploaded_file(file) for file in uploaded_files]
        full_df = pd.concat(dfs, ignore_index=True)
        filtered = get_filtered_base(full_df)

    df_mp: pd.DataFrame | None = None
    mp_source_name = ""
    if uploaded_mp_file:
        df_mp = load_mercado_pago_file(uploaded_mp_file)
        mp_source_name = uploaded_mp_file.name

    if tipo_relatorio == "Mercado Pago":
        st.markdown("<div id='print-content'>", unsafe_allow_html=True)
        if df_mp is None:
            st.info("Envie o arquivo de vendas Mercado Pago para ver esta aba.")
        else:
            render_dashboard_mercado_pago(df_mp, mp_source_name)
        st.markdown("</div>", unsafe_allow_html=True)
        return

    if full_df is None or filtered is None:
        st.info("Envie as planilhas para iniciar o dashboard 2026.")
        return

    if filtered.empty:
        st.warning("Não há registros com `Registered status = True` no recorte atual.")
        return

    # Data base do relatório: última inscrição válida com horário (prioriza date_time).
    data_base_ts = pd.NaT
    if "date_time_parsed" in filtered.columns:
        data_base_ts = filtered["date_time_parsed"].max()
    if pd.isna(data_base_ts):
        data_base_ts = filtered["Registration date"].max()
    if pd.notna(data_base_ts):
        data_base_label = data_base_ts.strftime("%d/%m/%Y às %H:%M")
    else:
        data_base_label = f"{date.today():%d/%m/%Y} às 00:00"

    scoped = filtered.copy()
    if scoped.empty:
        st.warning("Os filtros atuais não retornaram dados.")
        return

    ibge_df = load_ibge()
    st.markdown("<div id='print-content'>", unsafe_allow_html=True)
    if tipo_relatorio == "Marketing Diário":
        render_marketing_diario(
            scoped=scoped,
            full_df=full_df,
            percurso_targets=percurso_targets,
            data_base_label=data_base_label,
            data_base_ts=data_base_ts,
        )
    elif tipo_relatorio == "Marketing Semanal":
        render_marketing_semanal(
            scoped=scoped,
            full_df=full_df,
            percurso_targets=percurso_targets,
            start_date=start_date,
            end_date=end_date,
            data_base_label=data_base_label,
            data_base_ts=data_base_ts,
            ibge_df=ibge_df,
            print_mode=print_mode,
        )
    elif tipo_relatorio == "Marketing":
        render_header_marketing(scoped, data_base_label)
        render_progress_projection(scoped, percurso_targets, start_date, end_date)
        render_marketing_target_gauges(scoped, percurso_targets)
        render_marketing_coupon_block(scoped)
        insert_print_break(print_mode)
        render_demography(scoped, expandido=True)
        render_geography(scoped, ibge_df)
        render_international(scoped)
        insert_print_break(print_mode)
        render_sales_patterns(scoped)
        render_horarios_venda(scoped)
        render_team_medical_company(scoped)
        render_yopp_section(scoped, ibge_df)
        render_nubank_section(scoped, ibge_df)
        render_perfil_inscrito(scoped, ibge_df)
    elif tipo_relatorio == "Geral":
        render_header(scoped, data_base_label)
        render_venn_unique_athletes(full_df)
        insert_print_break(print_mode)
        render_progress_projection(scoped, percurso_targets, start_date, end_date)
        render_demography(scoped)
        insert_print_break(print_mode)
        render_geography(scoped, ibge_df)
        render_international(scoped)
        render_sales_patterns(scoped)
        insert_print_break(print_mode)
        render_financial(scoped)
        render_nubank_section(scoped, ibge_df)
        render_exports(full_df, scoped)
    else:
        render_header_financial(scoped, data_base_label)
        insert_print_break(print_mode)
        render_financial_report(scoped)
        render_exports(full_df, scoped)

    with st.expander("Diagnóstico técnico dos uploads", expanded=st.session_state.get("print_mode", False)):
        diag = (
            full_df.groupby(["source_file", "edition_currency"])
            .size()
            .rename("registros")
            .reset_index()
            .sort_values("registros", ascending=False)
        )
        st.dataframe(diag, hide_index=True, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
