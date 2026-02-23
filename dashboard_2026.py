import json
import re
import unicodedata
from datetime import date
from io import BytesIO

import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import requests
import streamlit as st


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

MODULE_DECISION_OPTIONS = ["Manter", "Ajustar", "Remover"]
REQUIRED_COLUMNS = [
    "Edition ID",
    "Registration date",
    "Registered status",
    "Competition",
    "Registration amount",
    "Total registration amount",
    "Total discounts amount",
]


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
        "UTSB 110": "UTSB 100",
        "UTSB 100 (108KM)": "UTSB 100",
        "PTR 55 (58KM)": "PTR 55",
        "PTR 35 (34KM)": "PTR 35",
        "PTR 20 (25KM)": "PTR 20",
        "FUN 7KM": "FUN 7KM",
        "FUN 7KM ": "FUN 7KM",
        "FUN 7": "FUN 7KM",
        "PTR20": "PTR 20",
        "PTR35": "PTR 35",
        "PTR55": "PTR 55",
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
        df["nubank_flag"] = (
            df["nubank_opt"]
            .astype(str)
            .str.strip()
            .str.lower()
            .eq("yes, i will pay with a nubank card")
            .astype(int)
        )
    else:
        df["nubank_flag"] = 0

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

    st.markdown(
        f"""
        <div class="hero">
          <h2>Dashboard de Inscricoes 2026 - Paraty Brazil by UTMB</h2>
          <p class="subtle">Base atualizada ate {data_base:%d/%m/%Y} | Conversao fixa: 1 USD = R$ 5,00</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Inscritos ativos", format_int(total))
    c2.metric("% mulheres", format_pct(pct_female))
    c3.metric("% estrangeiros", format_pct(pct_foreigners))
    c4.metric("Paises distintos", format_int(countries))
    c5.metric("Receita liquida", format_currency(net_revenue))
    c6.metric("Ticket medio", format_currency(avg_ticket))


def render_module_review(decisions: dict[str, str]) -> None:
    rows = [{"Modulo": module, "Decisao 2026": decision} for module, decision in decisions.items()]
    st.markdown("### Checkpoint de modulos 2026")
    st.caption("Use este quadro para revisar comigo modulo por modulo: Manter, Ajustar ou Remover.")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_progress_projection(df: pd.DataFrame, meta_total: int, start_date: date, end_date: date) -> None:
    st.header("Projecoes e ritmo de vendas")
    comp_counts = (
        df["Competition"].fillna("NA")
        .value_counts()
        .rename_axis("Competition")
        .reset_index(name="Inscritos")
        .sort_values("Inscritos", ascending=False)
    )
    st.subheader("Metas por percurso (inscritos atuais)")
    fig_comp = px.bar(comp_counts, x="Competition", y="Inscritos", text="Inscritos", color="Competition")
    fig_comp.update_layout(showlegend=False, height=350)
    st.plotly_chart(fig_comp, use_container_width=True)

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


def render_demography(df: pd.DataFrame) -> None:
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

    if "gender" in df.columns:
        gender_counts = (
            df["gender"]
            .astype(str)
            .str.upper()
            .replace({"FEMALE": "F", "MALE": "M"})
            .value_counts()
            .rename_axis("Genero")
            .reset_index(name="Qtd")
        )
        if not gender_counts.empty:
            fig_gender = px.pie(gender_counts, names="Genero", values="Qtd", hole=0.45)
            fig_gender.update_layout(height=320)
            st.plotly_chart(fig_gender, use_container_width=True)


def render_geography(df: pd.DataFrame, ibge_df: pd.DataFrame) -> None:
    st.header("Geografia Brasil")
    brazil = df[df["country_std"].isin(["BRAZIL", "BRASIL", "BR"])].copy()
    if brazil.empty:
        st.info("Sem inscritos do Brasil para exibir nesta filtragem.")
        return

    city_counts = (
        brazil["city"].astype(str).value_counts().head(15).rename_axis("Cidade").reset_index(name="Inscritos")
    )
    st.subheader("Top cidades")
    st.plotly_chart(px.bar(city_counts, x="Inscritos", y="Cidade", orientation="h"), use_container_width=True)

    merged = brazil.merge(ibge_df[["City_norm", "UF", "Região"]], left_on="city_norm", right_on="City_norm", how="left")
    uf_counts = merged["UF"].fillna("NA").value_counts().rename_axis("UF").reset_index(name="Inscritos")
    reg_counts = merged["Região"].fillna("NA").value_counts().rename_axis("Regiao").reset_index(name="Inscritos")

    col1, col2 = st.columns(2)
    col1.plotly_chart(px.bar(uf_counts, x="UF", y="Inscritos"), use_container_width=True)
    col2.plotly_chart(px.pie(reg_counts, names="Regiao", values="Inscritos", hole=0.4), use_container_width=True)


def render_international(df: pd.DataFrame) -> None:
    st.header("Internacional")
    intl = df[df["nationality_std"] != "BR"].copy()
    if intl.empty:
        st.info("Sem estrangeiros para o recorte atual.")
        return

    nat_counts = intl["nationality_std"].fillna("NA").value_counts().head(15).rename_axis("Pais").reset_index(name="Inscritos")
    st.plotly_chart(px.bar(nat_counts, x="Pais", y="Inscritos"), use_container_width=True)
    with st.expander("Lista completa de paises"):
        full = intl["nationality_std"].fillna("NA").value_counts().rename_axis("Pais").reset_index(name="Inscritos")
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

    st.plotly_chart(px.bar(week_counts, x="week", y="Inscricoes", title="Ultimas 12 semanas"), use_container_width=True)
    col1, col2 = st.columns(2)
    col1.plotly_chart(px.bar(day_counts.tail(30), x="day", y="Inscricoes", title="Ultimos 30 dias"), use_container_width=True)
    col2.plotly_chart(px.bar(weekday_counts, x="weekday", y="Inscricoes", title="Media por dia da semana"), use_container_width=True)

    day_counts["acumulado"] = day_counts["Inscricoes"].cumsum()
    st.plotly_chart(px.line(day_counts, x="day", y="acumulado", title="Inscricoes acumuladas"), use_container_width=True)


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
        .sort_values("receita_liquida", ascending=False)
    )
    comp["ticket_medio"] = comp.apply(
        lambda row: row["receita_liquida"] / row["inscritos"] if row["inscritos"] else 0, axis=1
    )
    st.dataframe(comp, hide_index=True, use_container_width=True)

    if not comp.empty:
        fig = px.pie(comp, names="Competition", values="receita_liquida", hole=0.45)
        fig.update_layout(height=360)
        st.plotly_chart(fig, use_container_width=True)


def render_exports(raw_df: pd.DataFrame, kpi_df: pd.DataFrame, decisions: dict[str, str]) -> None:
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
        "module_decisions": decisions,
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
    st.sidebar.title("Configuracoes 2026")
    st.sidebar.caption("Dashboard unico em Streamlit com upload manual BRL/USD.")

    uploaded_files = st.sidebar.file_uploader(
        "Envie os arquivos 2026 (BR_FULL / US_FULL)",
        type=["xlsx"],
        accept_multiple_files=True,
    )

    module_decisions = {}
    st.sidebar.markdown("### Revisao por modulo")
    for module in DEFAULT_MODULES:
        module_decisions[module] = st.sidebar.selectbox(
            f"{module}",
            MODULE_DECISION_OPTIONS,
            index=0,
            key=f"decision_{module}",
        )

    active_modules = [module for module, decision in module_decisions.items() if decision != "Remover"]

    meta_total = st.sidebar.number_input("Meta total de inscritos", min_value=1, value=3500, step=50)
    start_date = st.sidebar.date_input("Inicio da campanha", value=date(2025, 10, 1))
    end_date = st.sidebar.date_input("Fim da campanha", value=date(2026, 9, 20))

    if not uploaded_files:
        st.info("Envie as planilhas para iniciar o dashboard 2026.")
        render_module_review(module_decisions)
        return

    dfs = [preprocess_uploaded_file(file) for file in uploaded_files]
    full_df = pd.concat(dfs, ignore_index=True)
    filtered = get_filtered_base(full_df)

    if filtered.empty:
        st.warning("Nao ha registros com `Registered status = True` no recorte atual.")
        render_module_review(module_decisions)
        return

    data_base = filtered["Registration date"].max()
    data_base = data_base.date() if pd.notna(data_base) else date.today()

    all_competitions = sorted(filtered["Competition"].dropna().unique().tolist())
    selected_competitions = st.sidebar.multiselect(
        "Filtrar competicoes",
        all_competitions,
        default=all_competitions,
    )
    all_currencies = sorted(filtered["edition_currency"].dropna().unique().tolist())
    selected_currencies = st.sidebar.multiselect(
        "Filtrar moeda (Edition ID)",
        all_currencies,
        default=all_currencies,
    )

    scoped = filtered[
        filtered["Competition"].isin(selected_competitions)
        & filtered["edition_currency"].isin(selected_currencies)
    ].copy()
    if scoped.empty:
        st.warning("Os filtros atuais nao retornaram dados.")
        render_module_review(module_decisions)
        return

    ibge_df = load_ibge()
    render_header(scoped, data_base)
    render_module_review(module_decisions)

    if "Projecoes e ritmo de vendas" in active_modules:
        render_progress_projection(scoped, meta_total, start_date, end_date)
    if "Demografia" in active_modules:
        render_demography(scoped)
    if "Geografia Brasil" in active_modules:
        render_geography(scoped, ibge_df)
    if "Internacional" in active_modules:
        render_international(scoped)
    if "Comparativo historico" in active_modules:
        render_historical(scoped)
    if "Padroes de venda" in active_modules:
        render_sales_patterns(scoped)
    if "Financeiro" in active_modules:
        render_financial(scoped)
    if "Exportacoes" in active_modules:
        render_exports(full_df, scoped, module_decisions)

    with st.expander("Diagnostico tecnico dos uploads"):
        diag = (
            full_df.groupby(["source_file", "edition_currency"])
            .size()
            .rename("registros")
            .reset_index()
            .sort_values("registros", ascending=False)
        )
        st.dataframe(diag, hide_index=True, use_container_width=True)


if __name__ == "__main__":
    main()
