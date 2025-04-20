import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import requests
from io import BytesIO
import unicodedata
import re
import difflib
import pdfkit
from matplotlib_venn import venn3
import matplotlib.pyplot as plt
import os
from datetime import date

st.set_page_config(layout="wide")

# Subsection: Estilos de Impressão
def apply_print_css():
    st.markdown("""
    <style>
    @media print {
        body { margin: 1cm; }
        .element-container, .stTable, .plotly-graph-div {
            page-break-inside: avoid !important;
            max-width: 100% !important;
        }
        .stTable table { font-size: 10pt; }
        footer::after {
            content: "Página " counter(page);
            position: fixed;
            bottom: 0;
        }
        .titulo { font-size: 24pt; }
    }
    </style>
    """, unsafe_allow_html=True)

apply_print_css()

# Subsection: Quebra de Página
def page_break():
    st.markdown('<div style="page-break-after: always;"></div>', unsafe_allow_html=True)

# ─── Funções Auxiliares ───
# Subsection: Normalização de Texto
def norm_text(text):
    t = unicodedata.normalize('NFKD', str(text))
    t = ''.join(c for c in t if not unicodedata.combining(c))
    return re.sub(r'[^a-z0-9\s]', '', t.lower().strip())

# Subsection: Formatação
def format_currency(val):
    try:
        return f"R$ {float(val):,.0f}".replace(",", ".")
    except Exception:
        return val

def format_percentage(val):
    try:
        return f"{float(val):.2f}%".replace('.', ',')
    except Exception:
        return val

def format_integer(val):
    try:
        return str(int(round(float(val))))
    except Exception:
        return val

def format_integer_thousands(val):
    try:
        return f"{int(round(float(val))):,}".replace(",", ".")
    except Exception:
        return val

# Subsection: Padronização de Nacionalidade
def standardize_nationality(value):
    if pd.isnull(value):
        return value
    value = value.strip().upper()
    mapping = {"BRASIL": "BR", "BRAZIL": "BR"}
    return mapping.get(value, value)

# Subsection: Padronização de Competição
def standardize_competition(value):
    if pd.isnull(value):
        return value
    competition_mapping = {
        "UTSB 110": "UTSB 100",
        "UTSB 100 (108km)": "UTSB 100",
        "PTR 55 (58km)": "PTR 55",
        "PTR 35 (34km)": "PTR 35",
        "PTR 20 (25km)": "PTR 20",
        "Fun 7km": "FUN 7KM",
        "FUN 7KM": "FUN 7KM",
        "PTR20": "PTR 20",
        "PTR35": "PTR 35",
        "PTR55": "PTR 55",
        "Kids": "KIDS",
        "Kids Race": "KIDS",
        "KIDS_EVENT": "KIDS"
    }
    value = str(value).strip().upper()
    return competition_mapping.get(value, value)

# ─── Carregamento de Dados ───
# Subsection: Municípios IBGE
@st.cache_data(show_spinner=False)
def load_ibge_municipios():
    IBGE_URL = (
        "https://raw.githubusercontent.com/rafaelnmiranda/dash_utmb/"
        "de2e7125c2a3c08c7c41be14c43e528b43c2ea58/municipios_IBGE.xlsx"
    )
    resp = requests.get(IBGE_URL, timeout=10)
    df = pd.read_excel(BytesIO(resp.content), engine='openpyxl')
    df['City_norm'] = df['City'].apply(norm_text)
    # Deduplicate cities to prevent merge inflation
    df = df.drop_duplicates(subset=['City_norm'], keep='first')
    st.write(f"Loaded IBGE municipalities: {len(df)} unique cities after deduplication.")
    return df

# Subsection: Correção de Cidade
def correct_city(city, ibge_df, cutoff=0.8):
    city_choices = ibge_df['City_norm'].tolist()
    norm = norm_text(city)
    matches = difflib.get_close_matches(norm, city_choices, n=1, cutoff=cutoff)
    if matches:
        return ibge_df.loc[ibge_df['City_norm'] == matches[0], 'City'].iat[0]
    return city

# Subsection: Dados Estáticos
@st.cache_data(show_spinner=False)
def load_static_data(ano, url):
    response = requests.get(url)
    df = pd.read_excel(BytesIO(response.content), sheet_name=0)
    df['Registration date'] = pd.to_datetime(df['Registration date'], errors='coerce')
    df['Ano'] = ano
    df['Moeda'] = 'USD'
    return df

# Subsection: Pré-processamento
@st.cache_data(show_spinner=False)
def preprocess_data(dfs, ibge_df, taxa_cambio=5.5):
    processed_dfs = []
    for df in dfs:
        df = df.copy()
        #st.write(f"Initial rows for year {df['Ano'].iloc[0]}: {len(df)}")
        
        df['Registration date'] = pd.to_datetime(df['Registration date'], errors='coerce')
        df['City'] = df['City'].astype(str).apply(lambda x: correct_city(x, ibge_df))
        df['Nationality'] = df['Nationality'].apply(standardize_nationality)
        
        if 'Registration amount' in df.columns:
            df['Registration amount'] = pd.to_numeric(df['Registration amount'], errors='coerce')
            if df['Moeda'].iloc[0] == 'USD':
                df['Registration amount'] *= taxa_cambio
        if 'Discounts amount' in df.columns:
            df['Discounts amount'] = pd.to_numeric(df['Discounts amount'], errors='coerce')
            if df['Moeda'].iloc[0] == 'USD':
                df['Discounts amount'] *= taxa_cambio
        
        # Standardize Competition before exclusion
        df['Competition'] = df['Competition'].apply(standardize_competition)
        
        # Exclude KIDS before any further processing
        df = df[~df['Competition'].str.contains("KIDS", na=False, case=False)]
        #st.write(f"Rows after excluding KIDS for year {df['Ano'].iloc[0]}: {len(df)}")
        
        processed_dfs.append(df)
    return pd.concat(processed_dfs, ignore_index=True)

# ─── Carregamento e Upload de Dados ───
with st.spinner("🔄 Carregando dados..."):
    ibge_df = load_ibge_municipios()
    url_2023 = "https://github.com/rafaelnmiranda/dash_utmb/raw/815dda1e46bf0b731212e12a365ad169dc4d4e23/UTMB%20-%202023%20-%20USD.xlsx"
    url_2024 = "https://github.com/rafaelnmiranda/dash_utmb/raw/815dda1e46bf0b731212e12a365ad169dc4d4e23/UTMB%20-%202024%20-%20USD.xlsx"
    df_2023 = load_static_data(2023, url_2023)
    df_2024 = load_static_data(2024, url_2024)
    uploaded_files = st.file_uploader("Carregue os arquivos de 2025 (USD e BRL)", type=["xlsx"], accept_multiple_files=True)
    dfs_2025 = []
    for file in uploaded_files:
        if "USD" in file.name.upper():
            moeda = "USD"
        elif "R$" in file.name or "BRL" in file.name.upper():
            moeda = "BRL"
        else:
            moeda = "Desconhecido"
        df = pd.read_excel(file, sheet_name=0)
        df['Registration date'] = pd.to_datetime(df['Registration date'], errors='coerce')
        df['Ano'] = 2025
        df['Moeda'] = moeda
        dfs_2025.append(df)

if dfs_2025:
    df_total = preprocess_data([df_2023, df_2024] + dfs_2025, ibge_df, taxa_cambio=5.5)
    df_2025 = df_total[df_total['Ano'] == 2025]
    data_base = df_2025['Registration date'].max().date()
    if not isinstance(data_base, date):
        st.error("Erro: Data base inválida. Verifique os dados de 'Registration date' em 2025.")
        st.stop()
else:
    st.error("Por favor, faça o upload dos arquivos de 2025.")
    st.stop()

# ─── Dashboard ───
# Subsection: Estilos Visuais
st.markdown(
    """
    <style>
    .titulo {
        font-size: 48px;
        color: #002D74;
        font-weight: bold;
        text-align: center;
    }
    .subtitulo {
        font-size: 22px;
        color: #00C4B3;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Subsection: Exportação para PDF
@st.cache_resource(show_spinner=False)
def export_to_pdf(_data_base):
    try:
        public_url = "https://your-streamlit-app-url"
        config = pdfkit.configuration()
        if not os.path.exists(config.wkhtmltopdf.decode('utf-8')):
            config = pdfkit.configuration(wkhtmltopdf="/usr/bin/wkhtmltopdf")
        pdf_bytes = pdfkit.from_url(
            public_url,
            False,
            configuration=config,
            options={
                "page-size": "A4",
                "margin-top": "1cm",
                "margin-bottom": "1cm",
                "margin-left": "1cm",
                "margin-right": "1cm",
                "title": f"Dashboard de Inscrições – Paraty Brazil by UTMB (até {data_base:%d/%m/%Y})",
                "author": "Paraty Brazil by UTMB"
            }
        )
        return pdf_bytes
    except Exception:
        return None

if st.button("Baixar PDF"):
    pdf_bytes = export_to_pdf(data_base)
    if pdf_bytes:
        st.download_button(
            label="Baixar PDF",
            data=pdf_bytes,
            file_name="dashboard.pdf",
            mime="application/pdf"
        )
    else:
        st.error("Erro ao gerar PDF com wkhtmltopdf. Use a opção de impressão do navegador.")
        st.markdown('<button onclick="window.print()">Imprimir</button>', unsafe_allow_html=True)

# Subsection: Cabeçalho e KPIs
st.markdown(f'<p class="titulo">Dashboard de Inscrições – Paraty Brazil by UTMB (até {data_base:%d/%m/%Y})</p>', unsafe_allow_html=True)
total_inscritos_2025 = df_2025.shape[0]
num_paises_2025 = df_2025['Nationality'].nunique()
num_mulheres = df_2025['Gender'].str.strip().str.upper().isin(['F', 'FEMALE']).sum()
perc_mulheres = (num_mulheres / total_inscritos_2025 * 100) if total_inscritos_2025 else 0
meta_total = 3500
meta_progress = (total_inscritos_2025 / meta_total) * 100
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Inscritos", format_integer(total_inscritos_2025))
col2.metric("Países Diferentes", format_integer(num_paises_2025))
col3.metric("% Mulheres", format_percentage(perc_mulheres))
col4.metric("% Meta", format_percentage(meta_progress))
st.divider()
page_break()

# Subsection: Progressos e Projeções 2025
st.header("Progressos e Projeções 2025")
st.subheader("Metas por Percurso")
metas = pd.DataFrame({
    "Percurso": ["FUN 7KM", "PTR 20", "PTR 35", "PTR 55", "UTSB 100", "TOTAL"],
    "Meta 2025": [900, 900, 620, 770, 310, 3500]
})

inscritos_2025 = df_2025['Competition'].value_counts().reset_index()
inscritos_2025.columns = ['Percurso', 'Inscritos']
inscritos_2025['Percurso'] = inscritos_2025['Percurso'].str.upper().str.strip()
total_row = pd.DataFrame([{'Percurso': 'TOTAL', 'Inscritos': total_inscritos_2025}])
inscritos_2025 = pd.concat([inscritos_2025, total_row], ignore_index=True)
metas_df = metas.merge(inscritos_2025, on="Percurso", how="left").fillna(0)
metas_df["Inscritos"] = metas_df["Inscritos"].apply(format_integer)
metas_df["Meta 2025"] = metas_df["Meta 2025"].apply(format_integer)
metas_df["% da Meta"] = ((metas_df["Inscritos"].astype(float) / metas_df["Meta 2025"].astype(float)) * 100).fillna(0).apply(format_percentage)
st.table(metas_df)


st.subheader("Prazo Decorrido vs. Meta")
start_date = pd.Timestamp("2024-10-28")
end_date = pd.Timestamp("2025-08-15")
data_base = min(data_base, end_date.date())
total_period = (end_date - start_date).days
days_elapsed = (data_base - start_date.date()).days
prazo_percent = (days_elapsed / total_period) * 100
col_p, col_m = st.columns(2)
col_p.metric("Prazo Decorrido (%)", format_percentage(prazo_percent))
col_m.metric("Meta Alcançada (%)", format_percentage(meta_progress))

st.subheader("Média Móvel e Projeção")
@st.cache_data(show_spinner=False)
def compute_moving_averages(df_2025):
    df_daily = df_2025.copy()
    df_daily['Date'] = pd.to_datetime(df_daily['Registration date'].dt.date)
    daily_counts = df_daily.groupby('Date').size().reset_index(name='Inscritos')
    daily_counts.set_index('Date', inplace=True)
    daily_counts['MM_7'] = daily_counts['Inscritos'].rolling(window=7).mean()
    daily_counts['MM_15'] = daily_counts['Inscritos'].rolling(window=15).mean()
    daily_counts['MM_30'] = daily_counts['Inscritos'].rolling(window=30).mean()
    return daily_counts

daily_counts = compute_moving_averages(df_2025)
last_date = daily_counts.index.max()
projection_date = pd.Timestamp(year=2025, month=8, day=15)
delta_days = max((projection_date - last_date).days, 0)
mm7 = daily_counts.loc[last_date, 'MM_7'] if pd.notna(daily_counts.loc[last_date, 'MM_7']) else 0
mm15 = daily_counts.loc[last_date, 'MM_15'] if pd.notna(daily_counts.loc[last_date, 'MM_15']) else 0
mm30 = daily_counts.loc[last_date, 'MM_30'] if pd.notna(daily_counts.loc[last_date, 'MM_30']) else 0
proj_7 = total_inscritos_2025 + mm7 * delta_days
proj_15 = total_inscritos_2025 + mm15 * delta_days
proj_30 = total_inscritos_2025 + mm30 * delta_days
projection_df = pd.DataFrame({
    'Média Móvel': ['7 dias', '15 dias', '30 dias'],
    'Valor Médio Diário': [format_integer(mm7), format_integer(mm15), format_integer(mm30)],
    'Projeção Inscritos (15/08/2025)': [format_integer(proj_7), format_integer(proj_15), format_integer(proj_30)]
})
st.table(projection_df)
st.divider()
page_break()

# Subsection: Demografia
st.header("Demografia")
df_2025_age = df_2025.copy()
df_2025_age['Birthdate'] = pd.to_datetime(df_2025_age['Birthdate'], errors='coerce')
df_2025_age['Age'] = 2025 - df_2025_age['Birthdate'].dt.year
df_2025_age.loc[df_2025_age['Age'] < 15, 'Age'] = 40
mean_age = df_2025_age['Age'].mean()
st.metric("Idade Média", format_integer(mean_age))

st.subheader("Idade Mínima e Máxima por Competição")
age_by_competition = df_2025_age.groupby('Competition')['Age'].agg(['min', 'max']).reset_index()
age_by_competition.columns = ['Percurso', 'Idade Mínima', 'Idade Máxima']
st.table(age_by_competition)

with st.expander("Distribuição de Idades"):
    hist_data = [df_2025_age['Age'].dropna().tolist()]
    group_labels = ['Idades']
    fig_age = ff.create_distplot(hist_data, group_labels, show_hist=True, show_rug=False)
    fig_age.update_layout(title_text='Distribuição de Idade dos Atletas (2025)')
    st.plotly_chart(fig_age)
st.divider()
page_break()

# Subsection: Geografia Brasil
st.header("Geografia Brasil")
df_br_2025 = df_2025[df_2025['Country'].str.lower() == 'brazil']
num_cidades = df_br_2025['City'].nunique()
total_atletas = len(df_br_2025)
col1, col2 = st.columns(2)
col1.metric("Total de Municípios Distintos", num_cidades)
col2.metric("Total de Atletas", total_atletas)

with st.expander("Top 10 Cidades"):
    top_cidades = df_br_2025['City'].value_counts().reset_index()
    top_cidades.columns = ['Cidade', 'Inscritos']
    top_cidades = top_cidades.nlargest(10, 'Inscritos', keep='all')
    top_cidades['% do Total'] = (top_cidades['Inscritos'] / total_atletas * 100).round(2).astype(str) + '%'
    st.table(top_cidades)

with st.expander("Inscritos por Estado"):
    df_br_2025_with_uf = df_br_2025.merge(ibge_df[['City', 'UF']], on='City', how='left')
    all_ufs = sorted(ibge_df['UF'].dropna().unique())
    uf_counts = df_br_2025_with_uf['UF'].value_counts().reindex(all_ufs, fill_value=0)
    uf_df = uf_counts.reset_index()
    uf_df.columns = ['UF', 'Inscritos']
    uf_df['% do Total'] = (uf_df['Inscritos'] / total_atletas * 100).round(2).astype(str) + '%'
    uf_df = uf_df.sort_values('Inscritos', ascending=False).reset_index(drop=True)
    uf_df.index = uf_df.index + 1
    uf_df.index.name = 'Posição'
    st.table(uf_df)

with st.expander("Inscritos por Região"):
    df_br_2025_with_reg = df_br_2025.merge(ibge_df[['City', 'Região']], on='City', how='left')
    all_regs = sorted(ibge_df['Região'].dropna().unique())
    reg_counts = df_br_2025_with_reg['Região'].value_counts().reindex(all_regs, fill_value=0)
    reg_df = reg_counts.reset_index()
    reg_df.columns = ['Região', 'Inscritos']
    reg_df['% do Total'] = (reg_df['Inscritos'] / total_atletas * 100).round(2).astype(str) + '%'
    reg_df = reg_df.sort_values('Inscritos', ascending=False)
    st.table(reg_df)
st.divider()
page_break()

# Subsection: Internacional
st.header("Internacional")
with st.expander("Top 10 Países Estrangeiros"):
    df_2025['Nat_std'] = df_2025['Nationality'].apply(standardize_nationality)
    vc = df_2025[df_2025['Nat_std'] != 'BR']['Nat_std'].value_counts()
    total_estrangeiros = vc.sum()
    df_top = vc.head(10).reset_index()
    df_top.columns = ['País', 'Count']
    df_top['Inscritos'] = df_top['Count'].apply(format_integer)
    df_top['%'] = (df_top['Count'] / total_estrangeiros * 100).round(0).astype(int).astype(str) + '%'
    total_row = pd.DataFrame([{
        'País': 'Total Estrangeiros',
        'Inscritos': format_integer(total_estrangeiros),
        '%': '100%'
    }])
    st.table(pd.concat([df_top[['País', 'Inscritos', '%']], total_row], ignore_index=True))
st.divider()
page_break()

# Subsection: Comparativo Histórico
st.header("Comparativo Histórico (2023–2025)")
with st.expander("Comparativos Locais"):
    df_br_2023 = df_total.query("Ano==2023 and Country.str.lower()=='brazil'", engine="python")
    df_br_2024 = df_total.query("Ano==2024 and Country.str.lower()=='brazil'", engine="python")
    tot23, tot24, tot25 = map(len, (df_br_2023, df_br_2024, df_br_2025))
    
    st.subheader("Cidades")
    top_cities25 = df_br_2025['City'].value_counts().nlargest(10, keep='all')
    df_cmp_cidades = pd.DataFrame({'Cidade': top_cities25.index, 'Inscritos': top_cities25.values})
    df_cmp_cidades['%'] = (df_cmp_cidades['Inscritos'] / tot25 * 100).round(2).astype(str) + '%'
    for ano, df_ano, col_c, col_p, tot in [
        (2023, df_br_2023, 'Inscritos 2023', '% 2023', tot23),
        (2024, df_br_2024, 'Inscritos 2024', '% 2024', tot24),
    ]:
        cnt = df_ano['City'].value_counts().reindex(df_cmp_cidades['Cidade'], fill_value=0)
        df_cmp_cidades[col_c] = cnt.values
        df_cmp_cidades[col_p] = (cnt / tot * 100).round(2).astype(str) + '%'
    st.table(df_cmp_cidades)
    
    st.subheader("Estados")
    df_uf23 = df_br_2023.merge(ibge_df[['City', 'UF']], on='City', how='left')
    df_uf24 = df_br_2024.merge(ibge_df[['City', 'UF']], on='City', how='left')
    df_uf25 = df_br_2025.merge(ibge_df[['City', 'UF']], on='City', how='left')
    ufs = df_uf25['UF'].dropna().unique()
    cnt25 = df_uf25['UF'].value_counts().reindex(ufs, fill_value=0)
    df_cmp_uf = pd.DataFrame({'UF': cnt25.index, 'Inscritos': cnt25.values})
    df_cmp_uf['%'] = (df_cmp_uf['Inscritos'] / tot25 * 100).round(2).astype(str) + '%'
    for ano, df_ano, col_c, col_p, tot in [
        (2023, df_uf23, 'Inscritos 2023', '% 2023', tot23),
        (2024, df_uf24, 'Inscritos 2024', '% 2024', tot24),
    ]:
        cnt = df_ano['UF'].value_counts().reindex(df_cmp_uf['UF'], fill_value=0)
        df_cmp_uf[col_c] = cnt.values
        df_cmp_uf[col_p] = (cnt / tot * 100).round(2).astype(str) + '%'
    df_cmp_uf = df_cmp_uf.sort_values('Inscritos', ascending=False)
    st.table(df_cmp_uf)
    
    st.subheader("Regiões")
    df_reg23 = df_br_2023.merge(ibge_df[['City', 'Região']], on='City', how='left')
    df_reg24 = df_br_2024.merge(ibge_df[['City', 'Região']], on='City', how='left')
    df_reg25 = df_br_2025.merge(ibge_df[['City', 'Região']], on='City', how='left')
    regs = df_reg25['Região'].dropna().unique()
    r25 = df_reg25['Região'].value_counts().reindex(regs, fill_value=0)
    df_cmp_reg = pd.DataFrame({'Região': r25.index, 'Inscritos': r25.values})
    df_cmp_reg['%'] = (df_cmp_reg['Inscritos'] / tot25 * 100).round(2).astype(str) + '%'
    for ano, df_ano, col_c, col_p, tot in [
        (2023, df_reg23, 'Inscritos 2023', '% 2023', tot23),
        (2024, df_reg24, 'Inscritos 2024', '% 2024', tot24),
    ]:
        cnt = df_ano['Região'].value_counts().reindex(df_cmp_reg['Região'], fill_value=0)
        df_cmp_reg[col_c] = cnt.values
        df_cmp_reg[col_p] = (cnt / tot * 100).round(2).astype(str) + '%'
    df_cmp_reg = df_cmp_reg.sort_values('Inscritos', ascending=False)
    st.table(df_cmp_reg)

with st.expander("Participação por Ano"):
    st.subheader("Participação dos Atletas por Ano")
    set_2023 = set(df_total[df_total['Ano'] == 2023]['Email'].dropna().unique())
    set_2024 = set(df_total[df_total['Ano'] == 2024]['Email'].dropna().unique())
    set_2025 = set(df_total[df_total['Ano'] == 2025]['Email'].dropna().unique())
    only_2023 = set_2023 - set_2024 - set_2025
    only_2024 = set_2024 - set_2023 - set_2025
    only_2025 = set_2025 - set_2023 - set_2024
    only_2023_2024 = (set_2023 & set_2024) - set_2025
    only_2023_2025 = (set_2023 & set_2025) - set_2024
    only_2024_2025 = (set_2024 & set_2025) - set_2023
    all_three = set_2023 & set_2024 & set_2025
    participation = pd.DataFrame({
        'Participação': ['Apenas 2023', 'Apenas 2024', 'Apenas 2025',
                        'Apenas 2023 e 2024', 'Apenas 2023 e 2025',
                        'Apenas 2024 e 2025', '2023, 2024 e 2025'],
        'Quantidade': [len(only_2023), len(only_2024), len(only_2025),
                       len(only_2023_2024), len(only_2023_2025),
                       len(only_2024_2025), len(all_three)]
    })
    participation['Quantidade'] = participation['Quantidade'].apply(format_integer)
    st.table(participation)
    
    plt.figure(figsize=(6, 6))
    set_2023_count = len(set_2023)
    set_2024_count = len(set_2024)
    set_2025_count = len(set_2025)
    total_unique = len(set_2023.union(set_2024).union(set_2025))
    venn3([set_2023, set_2024, set_2025],
          set_labels=(f"2023 ({set_2023_count})", f"2024 ({set_2024_count})", f"2025 ({set_2025_count})"))
    plt.title(f"Participação dos Atletas por Ano (TOTAL: {total_unique})")
    st.pyplot(plt)
st.divider()
page_break()

# Subsection: Padrões de Venda
st.header("Padrões de Venda")
st.subheader("Inscrições por Semana (Últimas 10 Semanas)")
df_2025_acc = df_2025.copy()
df_2025_acc['Date'] = pd.to_datetime(df_2025_acc['Registration date'].dt.date)
df_2025_acc = df_2025_acc.sort_values('Date')

last_date = df_2025_acc['Date'].max()
if pd.isna(last_date):
    st.error("Nenhuma data de inscrição válida encontrada para 2025. Verifique os dados.")
    st.stop()

intervals = []
for i in range(10):
    end = last_date - pd.Timedelta(days=1) - pd.Timedelta(days=7 * i)
    start = end - pd.Timedelta(days=6)
    intervals.append((start, end))

data = []
for start, end in intervals:
    cnt = df_2025_acc[(df_2025_acc['Date'] >= start) & (df_2025_acc['Date'] <= end)].shape[0]
    label = f"{start.strftime('%d/%m')} – {end.strftime('%d/%m')}"
    data.append({'Semana': label, 'Inscritos': cnt})

weekly_counts = pd.DataFrame(data)[::-1].reset_index(drop=True)

if weekly_counts['Inscritos'].sum() == 0:
    st.warning("Nenhuma inscrição encontrada nas últimas 10 semanas.")

media_inscritos = weekly_counts['Inscritos'].mean()
fig_weekly = px.bar(
    weekly_counts,
    x='Semana',
    y='Inscritos',
    text='Inscritos',
    title="Inscrições Vendidas por Semana (Últimas 10 Semanas) - 2025",
    labels={"Semana": "Período", "Inscritos": "Quantidade de Inscrições"}
)
fig_weekly.update_traces(textposition='outside')
fig_weekly.add_scatter(
    x=weekly_counts['Semana'],
    y=[media_inscritos] * len(weekly_counts),
    mode='lines',
    name=f'Média: {media_inscritos:.1f}',
    line=dict(color='orange')
)
fig_weekly.add_annotation(
    x=weekly_counts['Semana'].iloc[-1],
    y=media_inscritos,
    text=f"{media_inscritos:.1f}",
    showarrow=False,
    yshift=10,
    font=dict(color='orange')
)
st.plotly_chart(fig_weekly)

st.subheader("Média de Inscrições por Dia da Semana")
df_2025_dia = df_2025[df_2025['Registration date'] >= pd.to_datetime('2024-11-01')].copy()
df_2025_dia['Date'] = pd.to_datetime(df_2025_dia['Registration date'].dt.date)
daily_counts = df_2025_dia.groupby('Date').size().reset_index(name='Inscritos')
daily_counts['Weekday'] = daily_counts['Date'].dt.dayofweek
weekday_avg = daily_counts.groupby('Weekday')['Inscritos'].mean().reset_index(name='Media Inscritos')
weekday_avg['Media Inscritos'] = weekday_avg['Media Inscritos'].round(2)
weekday_names = {0: "Segunda", 1: "Terça", 2: "Quarta", 3: "Quinta", 4: "Sexta", 5: "Sábado", 6: "Domingo"}
weekday_avg['Dia da Semana'] = weekday_avg['Weekday'].map(weekday_names)
weekday_avg = weekday_avg.sort_values('Weekday')
fig_weekday = px.bar(
    weekday_avg,
    x='Dia da Semana',
    y='Media Inscritos',
    text='Media Inscritos',
    title="Média de Inscrições por Dia da Semana (nov/24 + 2025)",
    labels={"Media Inscritos": "Média de Inscrições"}
)
fig_weekday.update_traces(texttemplate='%{text:.2f}', textposition='outside')
st.plotly_chart(fig_weekday)

with st.expander("Top 15 Dias de Maior Venda"):
    daily_counts['Media_Semanadia'] = daily_counts['Weekday'].map(weekday_avg.set_index('Weekday')['Media Inscritos'])
    daily_counts['%_Acima_Media'] = ((daily_counts['Inscritos'] - daily_counts['Media_Semanadia']) / daily_counts['Media_Semanadia'] * 100)
    daily_counts['Dia (dd/mm/aa)'] = daily_counts['Date'].dt.strftime('%d/%m/%y') + ' (' + daily_counts['Weekday'].map(weekday_names) + ')'
    daily_counts['% acima da média'] = daily_counts['%_Acima_Media'].round(0).astype(int).astype(str) + '%'
    top15 = daily_counts.sort_values('Inscritos', ascending=False).head(15)[['Dia (dd/mm/aa)', 'Inscritos', '% acima da média']]
    st.table(top15)
st.divider()
page_break()

# Subsection: Métricas Financeiras
st.header("Métricas Financeiras")
anos_fin = sorted(df_total['Ano'].unique())
opcao_ano_fin = st.multiselect("Selecione o(s) ano(s):", anos_fin, default=anos_fin)
df_financial = df_total[df_total['Ano'].isin(opcao_ano_fin)]
receita_bruta = df_financial['Registration amount'].sum() + df_financial['Discounts amount'].sum()
receita_liquida = df_financial['Registration amount'].sum()
total_inscritos_fin = df_financial['Email'].nunique()
col1, col2, col3 = st.columns(3)
col1.metric("Receita Bruta (R$)", format_currency(receita_bruta))
col2.metric("Receita Líquida (R$)", format_currency(receita_liquida))
col3.metric("Total Inscritos Únicos", format_integer(total_inscritos_fin))

with st.expander("Detalhamento Financeiro"):
    st.subheader("Cupom de Desconto")
    coupon_groups = df_financial['Discount code'].fillna('OUTROS').str.upper().str.extract(r'(PTY25|TG25|TP25|GP25)', expand=False)
    df_financial['Coupon Group'] = coupon_groups.fillna('OUTROS')
    coupon_summary = df_financial.groupby('Coupon Group').agg(
        Total_Discounts=('Discounts amount', 'sum'),
        Quantidade_Descontos=('Discounts amount', lambda x: (x > 0).sum())
    ).reset_index()
    coupon_summary['Total_Discounts'] = coupon_summary['Total_Discounts'].apply(lambda x: f"{int(round(x)):,}".replace(',', '.'))
    coupon_summary['Quantidade_Descontos'] = coupon_summary['Quantidade_Descontos'].apply(format_integer)
    total_valores = df_financial['Discounts amount'].sum()
    total_qtd = (df_financial['Discounts amount'] > 0).sum()
    total_row = pd.DataFrame([{
        'Coupon Group': 'Total',
        'Total_Discounts': f"{int(round(total_valores)):,}".replace(',', '.'),
        'Quantidade_Descontos': format_integer(total_qtd)
    }])
    st.table(pd.concat([coupon_summary, total_row], ignore_index=True))
    
    st.subheader("Receitas por Competição")
    grp = df_financial.groupby('Competition').agg({
        'Registration amount': 'sum',
        'Discounts amount': 'sum'
    })
    grp['Total Inscritos'] = df_financial.groupby('Competition').size()
    grp['Receita_Bruta'] = grp['Registration amount'] + grp['Discounts amount']
    grp['Receita_Líquida'] = grp['Registration amount']
    grp['Total_Descontos'] = grp['Discounts amount']
    revenue_by_competition = grp.reset_index().drop(columns=['Registration amount', 'Discounts amount'])
    total_row2 = pd.DataFrame([{
        'Competition': 'Total',
        'Total Inscritos': revenue_by_competition['Total Inscritos'].sum(),
        'Receita_Bruta': revenue_by_competition['Receita_Bruta'].sum(),
        'Receita_Líquida': revenue_by_competition['Receita_Líquida'].sum(),
        'Total_Descontos': revenue_by_competition['Total_Descontos'].sum()
    }])
    revenue_by_competition = pd.concat([revenue_by_competition, total_row2], ignore_index=True)
    revenue_by_competition['Total Inscritos'] = revenue_by_competition['Total Inscritos'].apply(format_integer)
    for col in ['Receita_Bruta', 'Receita_Líquida', 'Total_Descontos']:
        revenue_by_competition[col] = revenue_by_competition[col].apply(lambda x: f"{int(round(x)):,}".replace(',', '.'))
    st.table(revenue_by_competition)
st.divider()

st.markdown("***Fim do Dashboard***")
