import json
import re
import unicodedata
import difflib
from datetime import date
from io import BytesIO
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
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
EDITION_TO_CURRENCY = {
    "691336e655be7bd5663d55ef": "USD",
    "6985ddfb09a601887cdbec29": "BRL",
}

DEFAULT_MODULES = [
    "KPIs gerais e metas",
    "Projecoes e ritmo de vendas",
    "Demografia",
    "Geografia Brasil",
    "Internacional",
    "Comparativo historico",
    "Padroes de venda",
    "Financeiro",
    "Exportacoes",
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
# Atualize este dicionario quando enviar a lista oficial de logica.
COUPON_PREFIX_RULES = {
    "IFL": "INFLUENCIADORES",
    "IDS": "IDOSO",
    "PCD": "PESSOA COM DEFICIENCIA",
    "MDS": "MORADORES DE PARATY E REGIAO",
    "PRA": "PREFEITURA (CORTESIAS)",
    "CTA": "CORTESIA",
    "ETE": "ATLETA DE ELITE",
    "REM": "REMANEJADAS (VINDOS DE OUTROS ANOS)",
    "VIP": "VIP",
    "EXP": "EXPOSITORES",
    "PAT": "PATROCINADORES",
    "UPG": "UPGRADE (MUDANCA DE PERCURSO)",
    "DWG": "DOWNGRADE (MUDANCA DE PERCURSO)",
    "CPD": "CONTRAPARTIDA (PATROCINADORES E PARCEIROS)",
    "IDX": "INDEX (DESCONTO PELO UTMB INDEX DO ATLETA)",
    "NUB": "NUBANK (CONVIDADOS NUBANK OU AJUSTES DE PRECO)",
    "CEX": "CONCESSAO EXCEPCIONAL (OUTROS)",
    "COL": "COLLO (VENDA MANUAL PARA ARGENTINOS EM DOLAR)",
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
        .subtle {
            color: #6b7280;
            font-size: 0.9rem;
        }
        .module-card {
            background: #f8fafc;
            border: 1px solid #e5e7eb;
            border-radius: 12px;
            padding: 10px 12px;
            margin-bottom: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def apply_print_css() -> None:
    st.markdown(
        """
        <style>
        @media print {
          [data-testid="stSidebar"], [data-testid="stToolbar"], header, footer {
            display: none !important;
          }
          .element-container, .stTable, .plotly-graph-div {
            page-break-inside: avoid !important;
          }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def norm_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(text))
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    normalized = normalized.lower().strip()
    return re.sub(r"[^a-z0-9\\s]", "", normalized)


def format_currency(value: float) -> str:
    if pd.isna(value):
        return "R$ 0"
    return f"R$ {float(value):,.0f}".replace(",", ".")


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


def to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    txt = str(value).strip().lower()
    return txt in {"true", "1", "yes", "y", "sim"}


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


def detect_coupon_column(columns):
    for col in columns:
        col_norm = normalize_col_name(col)
        has_coupon = any(token in col_norm for token in ["coupon", "cupom", "voucher", "promo", "discount_code"])
        if has_coupon:
            return col
    return None


def classify_coupon_prefix(code: str) -> str:
    if pd.isna(code):
        return "SEM_CUPOM"
    clean = re.sub(r"[^A-Za-z0-9]", "", str(code).strip().upper())
    if not clean:
        return "SEM_CUPOM"
    prefix = clean[:3]
    return COUPON_PREFIX_RULES.get(prefix, f"OUTROS ({prefix})")


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


def validate_columns(df: pd.DataFrame, source_name: str) -> list[str]:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        st.warning(
            f"Arquivo `{source_name}` sem colunas obrigatorias: {', '.join(missing)}. "
            "Essas metricas podem ficar incompletas."
        )
    return missing


def preprocess_uploaded_file(uploaded_file) -> pd.DataFrame:
    df = pd.read_excel(uploaded_file, sheet_name=0)
    df = normalize_headers(df)
    validate_columns(df, uploaded_file.name)

    df["source_file"] = uploaded_file.name
    df["Registration date"] = pd.to_datetime(df.get("Registration date"), errors="coerce")
    df["birthdate"] = pd.to_datetime(df.get("birthdate"), errors="coerce")
    df["Ano"] = df["Registration date"].dt.year.fillna(2026).astype(int)
    if "date_time" in df.columns:
        parsed_date_time = pd.to_datetime(df["date_time"], format="%d-%m-%Y %H:%M:%S", errors="coerce")
        fallback_mask = parsed_date_time.isna() & df["date_time"].notna()
        if fallback_mask.any():
            parsed_date_time.loc[fallback_mask] = pd.to_datetime(
                df.loc[fallback_mask, "date_time"],
                errors="coerce",
                dayfirst=True,
            )
        df["date_time_parsed"] = parsed_date_time
    else:
        df["date_time_parsed"] = pd.NaT
    df["sale_hour"] = df["date_time_parsed"].dt.hour
    df["sale_period"] = pd.cut(
        df["sale_hour"],
        bins=[-0.1, 5.9, 11.9, 17.9, 23.9],
        labels=["Madrugada", "Manha", "Tarde", "Noite"],
    ).astype("object")
    df["sale_period"] = df["sale_period"].fillna("Sem horario")

    df["edition_currency"] = df.get("Edition ID").apply(parse_currency_from_edition)
    if (df["edition_currency"] == "UNKNOWN").any():
        unknown_ids = sorted(df.loc[df["edition_currency"] == "UNKNOWN", "Edition ID"].dropna().astype(str).unique())
        st.warning(f"Edition ID sem mapeamento de moeda: {', '.join(unknown_ids)}")

    # KPI oficial: considerar inscricoes com Registered status = True
    df["is_registered"] = df.get("Registered status").apply(to_bool)
    df["Competition"] = df.get("Competition").apply(standardize_competition)
    df = df[~df["Competition"].astype(str).str.contains("KIDS", case=False, na=False)].copy()

    df["nationality_std"] = df.get("nationality").apply(standardize_nationality)
    df["country_std"] = df.get("country").astype(str).str.strip().str.upper()
    df["city_norm"] = df.get("city").astype(str).map(norm_text)

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

    coupon_col = detect_coupon_column(df.columns)
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

    for numeric_col in ["Registration amount", "Total registration amount", "Total discounts amount"]:
        df[numeric_col] = pd.to_numeric(df.get(numeric_col), errors="coerce").fillna(0)

    usd_factor = df["edition_currency"].eq("USD").astype(int) * (EXCHANGE_RATE_USD_TO_BRL - 1) + 1
    df["registration_brl"] = df["Registration amount"] * usd_factor
    df["total_registration_brl"] = df["Total registration amount"] * usd_factor
    df["total_discounts_brl"] = df["Total discounts amount"] * usd_factor
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


@st.cache_data(show_spinner=False)
def load_historical_venn_sets(base_dir: str) -> tuple[dict[str, set[str]], list[str]]:
    base_path = Path(base_dir)
    email_sets = {edition: set() for edition in HISTORICAL_VENN_FILES}
    warnings: list[str] = []

    for edition, files in HISTORICAL_VENN_FILES.items():
        for filename in files:
            file_path = base_path / filename
            if not file_path.exists():
                warnings.append(f"Arquivo historico ausente: {filename}")
                continue
            try:
                hist_df = pd.read_excel(file_path, sheet_name=0)
                hist_df = normalize_headers(hist_df)
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"Falha ao ler {filename}: {exc}")
                continue

            email_col = get_email_column_name(hist_df)
            if not email_col:
                warnings.append(f"Coluna de e-mail nao encontrada em {filename}")
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
    st.header("Venn de Atletas Unicos (2023-2026)")

    hist_sets, issues = load_historical_venn_sets(str(Path(__file__).resolve().parent))
    for msg in issues:
        st.warning(msg)

    email_2026 = extract_unique_emails(df_2026_all_rows)
    if not email_2026:
        st.info("Sem e-mails validos no upload 2026 para montar o Venn.")
        return

    venn_sets = {
        "2023": hist_sets.get("2023", set()),
        "2024": hist_sets.get("2024", set()),
        "2025": hist_sets.get("2025", set()),
        "2026": email_2026,
    }
    totals = {edition: len(values) for edition, values in venn_sets.items()}
    membership = compute_membership_distribution(venn_sets)

    circle_layout = {
        "2023": {"x": 0.34, "y": 0.60, "color": "rgba(59,130,246,0.25)", "stroke": "#3b82f6"},
        "2024": {"x": 0.56, "y": 0.60, "color": "rgba(16,185,129,0.25)", "stroke": "#10b981"},
        "2025": {"x": 0.44, "y": 0.42, "color": "rgba(245,158,11,0.25)", "stroke": "#f59e0b"},
        "2026": {"x": 0.66, "y": 0.42, "color": "rgba(239,68,68,0.25)", "stroke": "#ef4444"},
    }
    radius = 0.22

    fig = go.Figure()
    for edition, layout in circle_layout.items():
        fig.add_shape(
            type="circle",
            xref="paper",
            yref="paper",
            x0=layout["x"] - radius,
            y0=layout["y"] - radius,
            x1=layout["x"] + radius,
            y1=layout["y"] + radius,
            fillcolor=layout["color"],
            line=dict(color=layout["stroke"], width=2),
        )
        fig.add_annotation(
            x=layout["x"],
            y=layout["y"] + radius + 0.08,
            xref="paper",
            yref="paper",
            text=f"{edition}<br>Total: {format_int(totals[edition])}",
            showarrow=False,
            align="center",
            font=dict(size=12),
        )

    fig.add_annotation(
        x=0.50,
        y=0.52,
        xref="paper",
        yref="paper",
        text="Sobreposicao por e-mail",
        showarrow=False,
        font=dict(size=12, color="#334155"),
    )
    fig.update_layout(
        height=520,
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis=dict(visible=False, range=[0, 1]),
        yaxis=dict(visible=False, range=[0, 1]),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("So em 1 edicao", format_int(membership[1]))
    c2.metric("Em 2 edicoes", format_int(membership[2]))
    c3.metric("Em 3 edicoes", format_int(membership[3]))
    c4.metric("Em 4 edicoes", format_int(membership[4]))


def render_header(kpi_df: pd.DataFrame, data_base_label: str) -> None:
    total = len(kpi_df)
    female = kpi_df["gender"].astype(str).str.upper().isin(["F", "FEMALE"]).sum() if "gender" in kpi_df.columns else 0
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
          <h2>Dashboard de Inscricoes 2026 - Paraty Brazil by UTMB</h2>
          <p class="subtle">Base atualizada ate {data_base_label} | Conversao fixa: 1 USD = R$ 5,00</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    c1.metric("Inscritos ativos", format_int(total))
    c2.metric("% mulheres", format_pct(pct_female))
    c3.metric("% estrangeiros", format_pct(pct_foreigners))
    c4.metric("Paises distintos", format_int(countries))
    c5.metric("Receita liquida", format_currency(net_revenue))
    c6.metric("Ticket medio", format_currency(avg_ticket))
    c7.metric("Pagaram com cartao Nubank", format_int(nubank_total))

    YOPP_META_OCULOS = 300
    pct_venda_yopp = (yopp_total / YOPP_META_OCULOS * 100) if YOPP_META_OCULOS else 0
    pct_atletas_yopp = (yopp_total / total * 100) if total else 0
    y1, y2 = st.columns(2)
    y1.metric("Oculos Yopp vendidos", format_int(yopp_total))
    y1.caption(
        f"% meta venda ({YOPP_META_OCULOS} óculos): {format_pct(pct_venda_yopp)} | "
        f"% atletas que compram: {format_pct(pct_atletas_yopp)} | Atletas totais: {format_int(total)}"
    )
    y2.metric("Interessados no onibus oficial", format_int(bus_total))
    y2.caption(f"BR: {format_int(bus_br)} | Estrangeiros: {format_int(bus_foreign)}")


def render_header_marketing(kpi_df: pd.DataFrame, data_base_label: str) -> None:
    total = len(kpi_df)
    female = kpi_df["gender"].astype(str).str.upper().isin(["F", "FEMALE"]).sum() if "gender" in kpi_df.columns else 0
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
          <p class="subtle">Base atualizada ate {data_base_label} | Visao de audiencia, perfil e ritmo</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Inscritos ativos", format_int(total))
    c2.metric("% mulheres", format_pct(pct_female))
    c3.metric("% estrangeiros", format_pct(pct_foreigners))
    c4.metric("Paises distintos", format_int(countries))

    yopp_pct = (yopp_total / total * 100) if total else 0
    bus_pct = (bus_total / total * 100) if total else 0
    y1, y2 = st.columns(2)
    y1.metric("Oculos Yopp vendidos", format_int(yopp_total))
    y1.caption(f"% dos inscritos: {format_pct(yopp_pct)}")
    y2.metric("Interessados no onibus oficial", format_int(bus_total))
    y2.caption(
        f"% dos inscritos: {format_pct(bus_pct)} | BR: {format_int(bus_br)} | Estrangeiros: {format_int(bus_foreign)}"
    )


def render_header_financial(kpi_df: pd.DataFrame, data_base_label: str) -> None:
    total = len(kpi_df)
    gross = float(kpi_df["total_registration_brl"].sum())
    discounts = float(kpi_df["total_discounts_brl"].sum())
    net = float(kpi_df["net_revenue_brl"].sum())
    avg_ticket_net = (net / total) if total else 0
    discount_rate = (discounts / gross * 100) if gross else 0
    paying_with_discount = int(kpi_df["total_discounts_brl"].gt(0).sum())
    pct_discount_orders = (paying_with_discount / total * 100) if total else 0

    st.markdown(
        f"""
        <div class="hero">
          <h2>Dashboard Financeiro 2026 - Paraty Brazil by UTMB</h2>
          <p class="subtle">Base atualizada ate {data_base_label} | Foco em faturamento, descontos e ticket medio</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Inscritos ativos", format_int(total))
    c2.metric("Faturamento total", format_currency(gross))
    c3.metric("Descontos totais", format_currency(discounts))
    c4.metric("Faturamento liquido", format_currency(net))
    c5.metric("Ticket medio liquido", format_currency(avg_ticket_net))
    c6.metric("% desconto sobre bruto", format_pct(discount_rate))
    st.caption(
        f"Inscricoes com desconto aplicado: {format_int(paying_with_discount)} "
        f"({format_pct(pct_discount_orders)})"
    )


def build_route_summary(df: pd.DataFrame, targets: dict[str, int]) -> pd.DataFrame:
    local = df.copy()
    if "gender" not in local.columns:
        local["gender"] = ""
    grouped = (
        local.groupby("Competition", dropna=False)
        .agg(
            inscritos=("Competition", "size"),
            receita_bruta=("total_registration_brl", "sum"),
            descontos=("total_discounts_brl", "sum"),
            receita_liquida=("net_revenue_brl", "sum"),
            mulheres=("gender", lambda s: s.astype(str).str.upper().isin(["F", "FEMALE"]).sum()),
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


def render_progress_projection(df: pd.DataFrame, targets: dict[str, int], start_date: date, end_date: date) -> None:
    st.header("Projecoes e ritmo de vendas")
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

    # Grafico 100% empilhado: todas as barras com mesma altura visual (0 a 100%)
    chart_summary = summary.copy()
    order_with_total = list(PERCURSO_ORDER) + ["TOTAL"]
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
    # Parte restante ate 100% (cinza)
    fig_comp.add_trace(
        go.Bar(
            x=chart_summary["Percurso"],
            y=chart_summary["pct_restante"],
            name="Restante ate meta",
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
    # Parte inscrita (azul), limitada visualmente a 100%
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
    fig_comp.update_traces(textposition="none", selector=dict(name="Restante ate meta"))
    fig_comp.update_layout(
        height=380,
        barmode="stack",
        xaxis={"categoryorder": "array", "categoryarray": order_with_total},
        yaxis_title="% da meta",
        yaxis=dict(range=[0, 100]),
        margin=dict(b=80),
    )
    st.plotly_chart(fig_comp, use_container_width=True)

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
        st.info("Sem datas suficientes para projecoes.")
        return

    series["mm7"] = series["inscricoes_diarias"].rolling(7, min_periods=1).mean()
    series["mm15"] = series["inscricoes_diarias"].rolling(15, min_periods=1).mean()
    series["mm30"] = series["inscricoes_diarias"].rolling(30, min_periods=1).mean()
    proj_rate = series["mm15"].iloc[-1]
    days_left = max((end_date - series["day"].iloc[-1]).days, 0)
    proj_total = int(round(total + proj_rate * days_left))

    st.subheader("Media movel e projecao")
    st.metric("Projecao de inscritos no prazo", format_int(proj_total))
    fig_mm = px.line(
        series,
        x="day",
        y=["inscricoes_diarias", "mm7", "mm15", "mm30"],
        labels={"day": "Data", "value": "Inscricoes", "variable": "Serie"},
    )
    fig_mm.update_layout(height=360)
    st.plotly_chart(fig_mm, use_container_width=True)


def render_demography(df: pd.DataFrame, expandido: bool = False) -> None:
    st.header("Demografia")
    valid_age = df["age"].dropna()
    c1, c2, c3 = st.columns(3)
    c1.metric("Idade media", format_int(valid_age.mean() if not valid_age.empty else 0))
    c2.metric("Idade minima", format_int(valid_age.min() if not valid_age.empty else 0))
    c3.metric("Idade maxima", format_int(valid_age.max() if not valid_age.empty else 0))

    if not valid_age.empty:
        hist = ff.create_distplot([valid_age], ["Idade"], show_hist=True, show_rug=False, bin_size=2)
        hist.update_layout(height=320)
        st.plotly_chart(hist, use_container_width=True)

    if not expandido:
        return

    st.subheader("Genero")
    if "gender" not in df.columns:
        st.info("Coluna de genero nao disponivel para analise.")
    else:
        gender_raw = df["gender"].astype(str).str.strip().str.upper()
        gender_norm = pd.Series("Outros/NA", index=df.index, dtype="object")
        gender_norm.loc[gender_raw.isin(["F", "FEMALE"])] = "Feminino"
        gender_norm.loc[gender_raw.isin(["M", "MALE"])] = "Masculino"
        gender_counts = gender_norm.value_counts().rename_axis("Genero").reset_index(name="Inscritos")
        total_gender = gender_counts["Inscritos"].sum()
        gender_counts["%"] = gender_counts["Inscritos"].apply(
            lambda v: format_pct((v / total_gender) * 100 if total_gender else 0)
        )
        col_a, col_b = st.columns(2)
        col_a.dataframe(gender_counts, hide_index=True, use_container_width=True)
        pie_gender = px.pie(gender_counts, names="Genero", values="Inscritos", hole=0.45)
        pie_gender.update_layout(height=320)
        col_b.plotly_chart(pie_gender, use_container_width=True)

    st.subheader("Faixas etarias")
    if valid_age.empty:
        st.info("Sem dados de idade para montar faixas etarias.")
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
    st.subheader("Top cidades")
    st.dataframe(city_counts, hide_index=True, use_container_width=True)

    merged = brazil.merge(ibge_df[["City_norm", "UF", "Região"]], left_on="city_norm", right_on="City_norm", how="left")
    uf_counts = merged["UF"].fillna("NA").value_counts().rename_axis("UF").reset_index(name="Inscritos")
    uf_counts["% brasileiros"] = uf_counts["Inscritos"].apply(lambda v: format_pct((v / total_brazil) * 100))
    uf_counts = uf_counts.sort_values("Inscritos", ascending=False)
    reg_counts = merged["Região"].fillna("NA").value_counts().rename_axis("Regiao").reset_index(name="Inscritos")

    st.subheader("Top Estados")
    st.dataframe(uf_counts, hide_index=True, use_container_width=True)

    st.subheader("Regioes do Brasil")
    col1, col2 = st.columns(2)
    col1.dataframe(reg_counts.sort_values("Inscritos", ascending=False), hide_index=True, use_container_width=True)
    pie = px.pie(reg_counts, names="Regiao", values="Inscritos", hole=0.4)
    pie.update_layout(height=320)
    col2.plotly_chart(pie, use_container_width=True)
    st.caption("NA = cidades sem correspondencia no IBGE para mapeamento de regiao.")


def render_international(df: pd.DataFrame) -> None:
    st.header("Internacional")
    intl = df[df["nationality_std"] != "BR"].copy()
    if intl.empty:
        st.info("Sem estrangeiros para o recorte atual.")
        return

    total_inscritos = len(df)
    nat_counts = intl["nationality_std"].fillna("NA").value_counts().rename_axis("Pais").reset_index(name="Inscritos")
    top_5 = nat_counts.head(5).copy()
    top_5["% dos inscritos totais"] = top_5["Inscritos"].apply(lambda v: format_pct((v / total_inscritos) * 100))
    st.dataframe(top_5, hide_index=True, use_container_width=True)
    with st.expander("Lista completa de paises"):
        full = nat_counts.copy()
        full["% dos inscritos totais"] = full["Inscritos"].apply(lambda v: format_pct((v / total_inscritos) * 100))
        st.dataframe(full, hide_index=True, use_container_width=True)


def render_historical(df: pd.DataFrame) -> None:
    st.header("Comparativo historico")
    yearly = df.groupby("Ano").size().rename("Inscritos").reset_index().sort_values("Ano")
    if yearly["Ano"].nunique() <= 1:
        st.info("Somente um ano detectado no upload. Este modulo fica ativo quando houver anos adicionais.")
        return

    st.plotly_chart(px.line(yearly, x="Ano", y="Inscritos", markers=True), use_container_width=True)
    if "Email" in df.columns:
        unique_by_year = df.dropna(subset=["Email"]).groupby("Ano")["Email"].nunique().rename("Atletas unicos").reset_index()
        fig_unique = px.bar(unique_by_year, x="Ano", y="Atletas unicos", text="Atletas unicos")
        fig_unique = style_bar_labels(fig_unique)
        st.plotly_chart(fig_unique, use_container_width=True)


def render_sales_patterns(df: pd.DataFrame) -> None:
    st.header("Padroes de venda")
    dated = df.dropna(subset=["Registration date"]).copy()
    if dated.empty:
        st.info("Sem datas para analise de padrao de vendas.")
        return

    dated["week"] = dated["Registration date"].dt.to_period("W").dt.start_time
    dated["day"] = dated["Registration date"].dt.date
    dated["weekday"] = dated["Registration date"].dt.day_name()

    week_counts = dated.groupby("week").size().reset_index(name="Inscricoes").sort_values("week").tail(12)
    day_counts = dated.groupby("day").size().reset_index(name="Inscricoes").sort_values("day")
    weekday_counts = dated.groupby("weekday").size().reset_index(name="Inscricoes")
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekday_counts["weekday"] = pd.Categorical(weekday_counts["weekday"], categories=weekday_order, ordered=True)
    weekday_counts = weekday_counts.sort_values("weekday")

    fig_week = px.bar(week_counts, x="week", y="Inscricoes", title="Ultimas 12 semanas", text="Inscricoes")
    fig_week = style_bar_labels(fig_week)
    st.plotly_chart(fig_week, use_container_width=True)
    col1, col2 = st.columns(2)
    fig_days = px.bar(day_counts.tail(30), x="day", y="Inscricoes", title="Ultimos 30 dias", text="Inscricoes")
    fig_days = style_bar_labels(fig_days)
    col1.plotly_chart(fig_days, use_container_width=True)
    fig_weekday = px.bar(
        weekday_counts,
        x="weekday",
        y="Inscricoes",
        title="Media por dia da semana",
        text="Inscricoes",
    )
    fig_weekday = style_bar_labels(fig_weekday)
    col2.plotly_chart(fig_weekday, use_container_width=True)

    day_counts["acumulado"] = day_counts["Inscricoes"].cumsum()
    st.plotly_chart(px.line(day_counts, x="day", y="acumulado", title="Inscricoes acumuladas"), use_container_width=True)


def render_horarios_venda(df: pd.DataFrame) -> None:
    st.header("Horarios de venda")
    if "date_time_parsed" not in df.columns:
        st.info("Coluna date_time nao disponivel para analise de horarios.")
        return

    local = df.dropna(subset=["date_time_parsed"]).copy()
    if local.empty:
        st.info("Sem registros validos em date_time para analise de horarios.")
        return

    if "sale_hour" not in local.columns:
        local["sale_hour"] = local["date_time_parsed"].dt.hour
    hour_counts = local.groupby("sale_hour").size().reindex(range(24), fill_value=0).reset_index(name="Inscricoes")
    hour_counts = hour_counts.rename(columns={"sale_hour": "Hora"})
    peak_hour = int(hour_counts.loc[hour_counts["Inscricoes"].idxmax(), "Hora"])
    peak_count = int(hour_counts["Inscricoes"].max())
    c1, c2 = st.columns(2)
    c1.metric("Horario de pico", f"{peak_hour:02d}h")
    c2.metric("Inscricoes no pico", format_int(peak_count))

    fig_hour = px.bar(hour_counts, x="Hora", y="Inscricoes", text="Inscricoes", title="Inscricoes por hora do dia")
    fig_hour = style_bar_labels(fig_hour)
    fig_hour.update_layout(height=340, xaxis=dict(dtick=1))
    st.plotly_chart(fig_hour, use_container_width=True)

    if "sale_period" not in local.columns:
        local["sale_period"] = pd.cut(
            local["sale_hour"],
            bins=[-0.1, 5.9, 11.9, 17.9, 23.9],
            labels=["Madrugada", "Manha", "Tarde", "Noite"],
        ).astype("object")
        local["sale_period"] = local["sale_period"].fillna("Sem horario")
    period_order = ["Madrugada", "Manha", "Tarde", "Noite", "Sem horario"]
    period_counts = local["sale_period"].value_counts().reindex(period_order, fill_value=0).rename_axis("Periodo")
    period_counts = period_counts.reset_index(name="Inscritos")
    total_period = int(period_counts["Inscritos"].sum())
    period_counts["%"] = period_counts["Inscritos"].apply(
        lambda v: format_pct((v / total_period) * 100 if total_period else 0)
    )

    top_period = period_counts.sort_values("Inscritos", ascending=False).iloc[0]
    st.metric("Maior volume por periodo", f"{top_period['Periodo']} ({format_int(top_period['Inscritos'])})")
    p1, p2 = st.columns(2)
    p1.dataframe(period_counts, hide_index=True, use_container_width=True)
    fig_period = px.bar(period_counts, x="Periodo", y="Inscritos", text="Inscritos", title="Inscricoes por periodo do dia")
    fig_period = style_bar_labels(fig_period)
    fig_period.update_layout(height=320)
    p2.plotly_chart(fig_period, use_container_width=True)


def render_team_medical_company(df: pd.DataFrame) -> None:
    st.header("Assessorias, atestado e empresa")
    total = len(df)
    if total == 0:
        st.info("Sem dados para analise de assessorias e cadastro.")
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
        st.info("Coluna Team nao disponivel nesta base.")
        return

    team_raw = df[team_col].astype(str).str.strip()
    valid_team = team_raw.replace({"nan": pd.NA, "None": pd.NA, "none": pd.NA, "": pd.NA}).dropna()
    if valid_team.empty:
        st.info("Nenhum valor de Team preenchido no recorte atual.")
        return

    team_df = valid_team.rename("team_raw").to_frame()
    team_df["team_key"] = team_df["team_raw"].apply(canonicalize_team_name)
    team_df = team_df[team_df["team_key"] != ""].copy()
    if team_df.empty:
        st.info("Nao foi possivel consolidar os nomes de Team.")
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


def render_perfil_inscrito(df: pd.DataFrame) -> None:
    st.header("Onibus Oficial")
    total = len(df)
    if total == 0:
        st.info("Sem dados para analise de onibus oficial no recorte atual.")
        return

    if "official_bus_flag" not in df.columns:
        st.info("Coluna de interesse no onibus oficial nao disponivel nesta base.")
        return

    interested = df[df["official_bus_flag"] > 0].copy()
    bus_total = len(interested)
    bus_pct = (bus_total / total * 100) if total else 0
    br_mask = interested["nationality_std"].eq("BR") if "nationality_std" in interested.columns else pd.Series(False, index=interested.index)
    bus_br = int(br_mask.sum())
    bus_foreign = int((~br_mask).sum()) if "nationality_std" in interested.columns else 0

    gender_raw = interested["gender"].astype(str).str.strip().str.upper() if "gender" in interested.columns else pd.Series(dtype="object")
    female_bus = int(gender_raw.isin(["F", "FEMALE"]).sum()) if not gender_raw.empty else 0
    male_bus = int(gender_raw.isin(["M", "MALE"]).sum()) if not gender_raw.empty else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Interessados", format_int(bus_total))
    c2.metric("% de inscritos interessados", format_pct(bus_pct))
    c3.metric("Mulheres interessadas", format_int(female_bus))
    c4.metric("Homens interessados", format_int(male_bus))
    c5.metric("Brasileiros / Estrangeiros", f"{format_int(bus_br)} / {format_int(bus_foreign)}")

    st.subheader("Interesse por percurso")
    if interested.empty:
        st.info("Nenhum interessado em onibus oficial no recorte atual.")
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

    city_col = "city" if "city" in interested.columns else "city_norm"
    top_cities = (
        interested[city_col]
        .astype(str)
        .str.strip()
        .replace({"": pd.NA, "nan": pd.NA, "NaN": pd.NA})
        .dropna()
        .value_counts()
        .head(5)
        .rename_axis("Cidade")
        .reset_index(name="Interessados")
    )
    top_cities["% dos interessados"] = top_cities["Interessados"].apply(
        lambda v: format_pct((v / bus_total) * 100 if bus_total else 0)
    )
    st.dataframe(top_cities, hide_index=True, use_container_width=True)


def render_yopp_section(df: pd.DataFrame, ibge_df: pd.DataFrame) -> None:
    st.header("Yopp")
    total = len(df)
    if total == 0:
        st.info("Sem dados para analise de Yopp no recorte atual.")
        return

    if "yopp_flag" not in df.columns:
        st.info("Coluna Yopp nao disponivel nesta base.")
        return

    yopp_buyers = df[df["yopp_flag"] > 0].copy()
    yopp_total = len(yopp_buyers)
    pct_yopp = (yopp_total / total * 100) if total else 0

    gender_raw = yopp_buyers["gender"].astype(str).str.strip().str.upper() if "gender" in yopp_buyers.columns else pd.Series(dtype="object")
    female_yopp = int(gender_raw.isin(["F", "FEMALE"]).sum()) if not gender_raw.empty else 0
    male_yopp = int(gender_raw.isin(["M", "MALE"]).sum()) if not gender_raw.empty else 0

    br_mask = yopp_buyers["nationality_std"].eq("BR") if "nationality_std" in yopp_buyers.columns else pd.Series(False, index=yopp_buyers.index)
    br_yopp = int(br_mask.sum())
    foreign_yopp = int((~br_mask).sum()) if "nationality_std" in yopp_buyers.columns else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Oculos vendidos", format_int(yopp_total))
    c2.metric("% inscritos que compraram", format_pct(pct_yopp))
    c3.metric("Mulheres", format_int(female_yopp))
    c4.metric("Homens", format_int(male_yopp))
    c5.metric("Brasileiros / Estrangeiros", f"{format_int(br_yopp)} / {format_int(foreign_yopp)}")

    st.subheader("Oculos vendidos por percurso")
    if yopp_buyers.empty:
        st.info("Nenhum oculos Yopp vendido no recorte atual.")
        return

    yopp_by_course = (
        yopp_buyers["Competition"]
        .fillna("NA")
        .value_counts()
        .rename_axis("Percurso")
        .reset_index(name="Oculos vendidos")
    )
    order_map = {name: idx for idx, name in enumerate(PERCURSO_ORDER)}
    yopp_by_course["ordem"] = yopp_by_course["Percurso"].map(order_map).fillna(999).astype(int)
    yopp_by_course = yopp_by_course.sort_values(["ordem", "Percurso"]).drop(columns="ordem").reset_index(drop=True)
    yopp_by_course["% do total Yopp"] = yopp_by_course["Oculos vendidos"].apply(
        lambda v: format_pct((v / yopp_total) * 100 if yopp_total else 0)
    )

    y1, y2 = st.columns(2)
    y1.dataframe(yopp_by_course, hide_index=True, use_container_width=True)
    fig_yopp = px.bar(yopp_by_course, x="Percurso", y="Oculos vendidos", text="Oculos vendidos")
    fig_yopp = style_bar_labels(fig_yopp)
    fig_yopp.update_layout(height=340)
    y2.plotly_chart(fig_yopp, use_container_width=True)

    st.subheader("Top 5 cidades (Yopp)")
    top_cities = build_top_cities_table(
        yopp_buyers,
        ibge_df=ibge_df,
        total=yopp_total,
        count_col="Oculos vendidos",
        pct_col="% do total Yopp",
    )
    if top_cities.empty:
        st.info("Sem cidades validas para montar ranking de Yopp.")
    else:
        st.dataframe(top_cities, hide_index=True, use_container_width=True)


def render_nubank_section(df: pd.DataFrame, ibge_df: pd.DataFrame) -> None:
    st.header("Nubank")
    if df.empty:
        st.info("Sem dados para analise de Nubank no recorte atual.")
        return

    if "nubank_flag" not in df.columns:
        st.info("Coluna `nubank_opt` nao disponivel nesta base.")
        return

    if "source_file" not in df.columns:
        st.info("Nao foi possivel identificar o arquivo de origem para aplicar o recorte BRL_FULL.")
        return

    br_full = df[df["source_file"].astype(str).str.contains(r"brl[\s_-]*full", case=False, na=False, regex=True)].copy()
    if br_full.empty:
        st.info("Nenhum registro BRL_FULL encontrado no upload atual.")
        return

    nubank_buyers = br_full[br_full["nubank_flag"] > 0].copy()
    nubank_total = len(nubank_buyers)
    total_br_full = len(br_full)
    nubank_pct = (nubank_total / total_br_full * 100) if total_br_full else 0

    gender_raw = nubank_buyers["gender"].astype(str).str.strip().str.upper() if "gender" in nubank_buyers.columns else pd.Series(dtype="object")
    female_nubank = int(gender_raw.isin(["F", "FEMALE"]).sum()) if not gender_raw.empty else 0
    male_nubank = int(gender_raw.isin(["M", "MALE"]).sum()) if not gender_raw.empty else 0
    other_nubank = max(nubank_total - female_nubank - male_nubank, 0)

    valid_age = nubank_buyers["age"].dropna() if "age" in nubank_buyers.columns else pd.Series(dtype="float64")
    avg_age = valid_age.mean() if not valid_age.empty else 0
    min_age = valid_age.min() if not valid_age.empty else 0
    max_age = valid_age.max() if not valid_age.empty else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Inscritos BRL_FULL", format_int(total_br_full))
    c2.metric("Compraram com Nubank", format_int(nubank_total))
    c3.metric("% BRL_FULL com Nubank", format_pct(nubank_pct))
    c4.metric("Mulheres / Homens", f"{format_int(female_nubank)} / {format_int(male_nubank)}")
    c5.metric("Outros/NA", format_int(other_nubank))

    a1, a2, a3 = st.columns(3)
    a1.metric("Idade media (Nubank)", format_int(avg_age))
    a2.metric("Idade minima", format_int(min_age))
    a3.metric("Idade maxima", format_int(max_age))

    if nubank_buyers.empty:
        st.info("Nenhum inscrito BRL_FULL comprou com desconto Nubank no recorte atual.")
        return

    st.subheader("Perfil de quem comprou com desconto Nubank (BRL_FULL)")
    p1, p2 = st.columns(2)

    gender_profile = pd.DataFrame(
        {
            "Perfil": ["Feminino", "Masculino", "Outros/NA"],
            "Inscritos": [female_nubank, male_nubank, other_nubank],
        }
    )
    gender_profile["%"] = gender_profile["Inscritos"].apply(
        lambda v: format_pct((v / nubank_total) * 100 if nubank_total else 0)
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
            .rename_axis("Faixa etaria")
            .reset_index(name="Inscritos")
        )
        age_counts["%"] = age_counts["Inscritos"].apply(
            lambda v: format_pct((v / nubank_total) * 100 if nubank_total else 0)
        )
        p2.dataframe(age_counts, hide_index=True, use_container_width=True)
    else:
        p2.info("Sem idades validas para montar faixas etarias.")

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
        st.info("Sem cidades validas para montar ranking de Nubank.")
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
    c3.metric("Receita liquida", format_currency(net))

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
    comp["ticket_medio"] = comp.apply(
        lambda row: row["receita_liquida"] / row["inscritos"] if row["inscritos"] else 0, axis=1
    )
    comp_table = comp.rename(
        columns={
            "Competition": "Percurso",
            "inscritos": "Inscritos",
            "receita_bruta": "Receita Bruta (R$)",
            "descontos": "Descontos (R$)",
            "receita_liquida": "Receita Liquida (R$)",
            "ticket_medio": "Ticket Medio (R$)",
        }
    )
    for col in ["Inscritos", "Receita Bruta (R$)", "Descontos (R$)", "Receita Liquida (R$)", "Ticket Medio (R$)"]:
        comp_table[col] = comp_table[col].map(format_int)
    st.dataframe(comp_table, hide_index=True, use_container_width=True)

    if not comp.empty:
        st.caption("% de contribuicao em receita de cada percurso")
        fig = px.pie(comp, names="Competition", values="receita_liquida", hole=0.45)
        fig.update_layout(height=360)
        st.plotly_chart(fig, use_container_width=True)


def render_financial_report(df: pd.DataFrame) -> None:
    st.header("Visao consolidada financeira")
    gross = float(df["total_registration_brl"].sum())
    discounts = float(df["total_discounts_brl"].sum())
    net = float(df["net_revenue_brl"].sum())
    total_orders = len(df)
    avg_gross = (gross / total_orders) if total_orders else 0
    avg_net = (net / total_orders) if total_orders else 0
    discount_rate = (discounts / gross * 100) if gross else 0
    orders_with_discount = int(df["total_discounts_brl"].gt(0).sum())
    pct_orders_with_discount = (orders_with_discount / total_orders * 100) if total_orders else 0

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Receita bruta total", format_currency(gross))
    k2.metric("Receita liquida total", format_currency(net))
    k3.metric("Ticket medio bruto", format_currency(avg_gross))
    k4.metric("Ticket medio liquido", format_currency(avg_net))
    k5, k6 = st.columns(2)
    k5.metric("Taxa de desconto total", format_pct(discount_rate))
    k6.metric("Inscricoes com cupom/desconto", format_pct(pct_orders_with_discount))

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
    by_route["ticket_medio_liquido"] = by_route.apply(
        lambda row: row["receita_liquida"] / row["inscritos"] if row["inscritos"] else 0,
        axis=1,
    )
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
            "receita_liquida": "Receita Liquida (R$)",
            "ticket_medio_liquido": "Ticket Medio Liquido (R$)",
            "taxa_desconto": "Taxa de Desconto",
        }
    )
    route_table["Inscritos"] = route_table["Inscritos"].map(format_int)
    for col in ["Receita Bruta (R$)", "Descontos (R$)", "Receita Liquida (R$)", "Ticket Medio Liquido (R$)"]:
        route_table[col] = route_table[col].map(format_currency)
    route_table["Taxa de Desconto"] = route_table["Taxa de Desconto"].map(format_pct)
    st.dataframe(route_table, hide_index=True, use_container_width=True)

    g1, g2 = st.columns(2)
    fig_net_route = px.bar(
        by_route,
        x="Competition",
        y="receita_liquida",
        text="receita_liquida",
        title="Receita liquida por percurso",
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

    st.subheader("Evolucao diaria de faturamento")
    dated = df.dropna(subset=["Registration date"]).copy()
    if dated.empty:
        st.info("Sem datas validas para evolucao diaria financeira.")
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
            labels={"day": "Data", "value": "Valor (R$)", "variable": "Serie"},
        )
        fig_daily.update_layout(height=360)
        st.plotly_chart(fig_daily, use_container_width=True)

    st.subheader("Estudo de cupons de desconto")
    discounted = df[df["total_discounts_brl"] > 0].copy()
    if discounted.empty:
        st.info("Nenhuma inscricao com desconto no recorte atual.")
        return

    coupon_code_clean = discounted["coupon_code"].fillna("").astype(str).str.strip()
    coupon_code_clean = coupon_code_clean.replace({"nan": "", "None": "", "none": ""})
    coupons = discounted[coupon_code_clean != ""].copy()
    if coupons.empty:
        st.info(
            "Ha descontos no recorte atual, mas sem codigo de cupom preenchido. "
            "Por isso este estudo considera zero registros."
        )
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Inscricoes com desconto", format_int(len(coupons)))
    c2.metric("Total de desconto (R$)", format_currency(coupons["total_discounts_brl"].sum()))
    c3.metric("Prefixos unicos (3 letras)", format_int(coupons["coupon_prefix"].nunique()))

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
            "coupon_family": "Familia de cupom",
            "inscricoes": "Inscricoes",
            "desconto_total": "Desconto Total (R$)",
            "receita_liquida": "Receita Liquida (R$)",
            "desconto_medio": "Desconto Medio (R$)",
        }
    )
    table_family["Inscricoes"] = table_family["Inscricoes"].map(format_int)
    for col in ["Desconto Total (R$)", "Receita Liquida (R$)", "Desconto Medio (R$)"]:
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
        "Este estudo considera apenas inscricoes com desconto e `coupon_code` preenchido. "
        "Classificacao atual usa `COUPON_PREFIX_RULES`."
    )


def render_exports(raw_df: pd.DataFrame, kpi_df: pd.DataFrame) -> None:
    st.header("Exportacoes")
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
    apply_print_css()
    st.sidebar.title("Configuracoes 2026")
    st.sidebar.caption("Dashboard unico em Streamlit com upload manual BRL/USD.")

    uploaded_files = st.sidebar.file_uploader(
        "Envie os arquivos 2026 (BR_FULL / US_FULL)",
        type=["xlsx"],
        accept_multiple_files=True,
    )
    st.sidebar.markdown("---")
    tipo_relatorio = st.sidebar.radio(
        "Selecione o relatorio",
        options=["Geral", "Marketing", "Financeiro"],
        index=0,
        label_visibility="collapsed",
    )

    st.sidebar.markdown("### Metas por percurso")
    percurso_targets = {}
    for percurso in PERCURSO_ORDER:
        percurso_targets[percurso] = st.sidebar.number_input(
            f"Meta {percurso}",
            min_value=0,
            value=DEFAULT_PERCURSO_TARGETS[percurso],
            step=50,
        )
    start_date = st.sidebar.date_input("Inicio da campanha", value=date(2026, 2, 23))
    end_date = st.sidebar.date_input("Fim da campanha", value=date(2026, 8, 14))
    if st.sidebar.button("Imprimir / Exportar PDF", use_container_width=True):
        components.html("<script>window.print();</script>", height=0, width=0)

    if not uploaded_files:
        st.info("Envie as planilhas para iniciar o dashboard 2026.")
        return

    dfs = [preprocess_uploaded_file(file) for file in uploaded_files]
    full_df = pd.concat(dfs, ignore_index=True)
    filtered = get_filtered_base(full_df)

    if filtered.empty:
        st.warning("Nao ha registros com `Registered status = True` no recorte atual.")
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
        st.warning("Os filtros atuais nao retornaram dados.")
        return

    ibge_df = load_ibge()
    st.markdown("<div id='print-content'>", unsafe_allow_html=True)
    if tipo_relatorio == "Geral":
        render_header(scoped, data_base_label)
        render_venn_unique_athletes(full_df)
        render_progress_projection(scoped, percurso_targets, start_date, end_date)
        render_demography(scoped)
        render_geography(scoped, ibge_df)
        render_international(scoped)
        render_sales_patterns(scoped)
        render_financial(scoped)
        render_nubank_section(scoped, ibge_df)
        render_exports(full_df, scoped)
    elif tipo_relatorio == "Marketing":
        render_header_marketing(scoped, data_base_label)
        render_progress_projection(scoped, percurso_targets, start_date, end_date)
        render_demography(scoped, expandido=True)
        render_geography(scoped, ibge_df)
        render_international(scoped)
        render_sales_patterns(scoped)
        render_horarios_venda(scoped)
        render_team_medical_company(scoped)
        render_yopp_section(scoped, ibge_df)
        render_nubank_section(scoped, ibge_df)
        render_perfil_inscrito(scoped)
    else:
        render_header_financial(scoped, data_base_label)
        render_financial_report(scoped)
        render_exports(full_df, scoped)

    with st.expander("Diagnostico tecnico dos uploads"):
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
