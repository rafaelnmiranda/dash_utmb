import json
import re
import unicodedata
import difflib
from datetime import date
from io import BytesIO

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


def detect_official_bus_column(columns):
    for col in columns:
        col_norm = normalize_col_name(col)
        has_bus = any(token in col_norm for token in ["bus", "onibus", "transporte", "shuttle"])
        has_intent = any(token in col_norm for token in ["official", "oficial", "interesse", "interested", "opt"])
        if has_bus and has_intent:
            return col
    return None


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

    if "nubank_opt" in df.columns:
        df["nubank_flag"] = parse_opt_in_series(df["nubank_opt"])
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


def render_header(kpi_df: pd.DataFrame, data_base: date) -> None:
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
          <p class="subtle">Base atualizada ate {data_base:%d/%m/%Y} | Conversao fixa: 1 USD = R$ 5,00</p>
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


def render_header_marketing(kpi_df: pd.DataFrame, data_base: date) -> None:
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
          <p class="subtle">Base atualizada ate {data_base:%d/%m/%Y} | Visao de audiencia, perfil e ritmo</p>
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
        fig_age.update_traces(textposition="outside")
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
        st.plotly_chart(px.bar(unique_by_year, x="Ano", y="Atletas unicos"), use_container_width=True)


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
    fig_week.update_traces(textposition="outside")
    st.plotly_chart(fig_week, use_container_width=True)
    col1, col2 = st.columns(2)
    fig_days = px.bar(day_counts.tail(30), x="day", y="Inscricoes", title="Ultimos 30 dias", text="Inscricoes")
    fig_days.update_traces(textposition="outside")
    col1.plotly_chart(fig_days, use_container_width=True)
    fig_weekday = px.bar(
        weekday_counts,
        x="weekday",
        y="Inscricoes",
        title="Media por dia da semana",
        text="Inscricoes",
    )
    fig_weekday.update_traces(textposition="outside")
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
    fig_hour.update_traces(textposition="outside")
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
    fig_period.update_traces(textposition="outside")
    fig_period.update_layout(height=320)
    p2.plotly_chart(fig_period, use_container_width=True)


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
        fig_percurso.update_traces(textposition="outside")
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


def render_yopp_section(df: pd.DataFrame) -> None:
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
    fig_yopp.update_traces(textposition="outside")
    fig_yopp.update_layout(height=340)
    y2.plotly_chart(fig_yopp, use_container_width=True)


def render_nubank_section(df: pd.DataFrame) -> None:
    st.header("Nubank")
    if df.empty:
        st.info("Sem dados para analise de Nubank no recorte atual.")
        return

    if "nubank_flag" not in df.columns:
        st.info("Coluna `nubank_opt` nao disponivel nesta base.")
        return

    if "source_file" not in df.columns:
        st.info("Nao foi possivel identificar o arquivo de origem para aplicar o recorte BR_FULL.")
        return

    br_full = df[df["source_file"].astype(str).str.contains("BRL_FULL", case=False, na=False)].copy()
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
    fig_route.update_traces(textposition="outside")
    fig_route.update_layout(height=320)
    r2.plotly_chart(fig_route, use_container_width=True)

    city_col = "city" if "city" in nubank_buyers.columns else "city_norm"
    if city_col in nubank_buyers.columns:
        st.subheader("Top 5 cidades (compras com desconto Nubank)")
        top_cities = (
            nubank_buyers[city_col]
            .astype(str)
            .str.strip()
            .replace({"": pd.NA, "nan": pd.NA, "NaN": pd.NA})
            .dropna()
            .value_counts()
            .head(5)
            .rename_axis("Cidade")
            .reset_index(name="Inscritos")
        )
        top_cities["% do total Nubank"] = top_cities["Inscritos"].apply(
            lambda v: format_pct((v / nubank_total) * 100 if nubank_total else 0)
        )
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
        options=["Geral", "Marketing"],
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

    # Data base do relatório: maior data de inscrição entre BR e US (já é o max do concat)
    data_base = filtered["Registration date"].max()
    data_base = data_base.date() if pd.notna(data_base) else date.today()

    scoped = filtered.copy()
    if scoped.empty:
        st.warning("Os filtros atuais nao retornaram dados.")
        return

    ibge_df = load_ibge()
    st.markdown("<div id='print-content'>", unsafe_allow_html=True)
    if tipo_relatorio == "Geral":
        render_header(scoped, data_base)
        render_progress_projection(scoped, percurso_targets, start_date, end_date)
        render_demography(scoped)
        render_geography(scoped, ibge_df)
        render_international(scoped)
        render_sales_patterns(scoped)
        render_financial(scoped)
        render_nubank_section(scoped)
        render_exports(full_df, scoped)
    else:
        render_header_marketing(scoped, data_base)
        render_progress_projection(scoped, percurso_targets, start_date, end_date)
        render_demography(scoped, expandido=True)
        render_geography(scoped, ibge_df)
        render_international(scoped)
        render_sales_patterns(scoped)
        render_horarios_venda(scoped)
        render_yopp_section(scoped)
        render_nubank_section(scoped)
        render_perfil_inscrito(scoped)

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
