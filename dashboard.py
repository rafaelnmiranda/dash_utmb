import streamlit as st
import pandas as pd
import plotly.express as px
import requests
from io import BytesIO

# ---------------------------
# Funções Auxiliares de Formatação
# ---------------------------
def format_currency(val):
    """Formata valor monetário para o formato 'R$ 3.412'."""
    try:
        # Formata com zero casas decimais e substitui vírgula por ponto
        return f"R$ {float(val):,.0f}".replace(",", ".")
    except Exception:
        return val

def format_percentage(val):
    """Formata valores percentuais para exibir como '35%'."""
    try:
        return f"{round(float(val))}%"
    except Exception:
        return val

def format_integer(val):
    """Formata valores numéricos (quantidades ou idades) para inteiros."""
    try:
        return str(int(round(float(val))))
    except Exception:
        return val

# ---------------------------
# Constante de Taxa de Câmbio
# ---------------------------
TAXA_CAMBIO = 5.5  # 1 USD = R$ 5,5

# -----------------------------------------------------
# 1. Carregamento dos Dados (2023, 2024 e 2025)
# -----------------------------------------------------
@st.cache_data(show_spinner=False)
def load_static_data(ano, url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            df = pd.read_excel(BytesIO(response.content), sheet_name="dados")
            df['Registration date'] = pd.to_datetime(df['Registration date'], errors='coerce')
            df['Ano'] = ano
            # Converte os valores monetários de USD para BRL
            if 'Registration amount' in df.columns:
                df['Registration amount'] = pd.to_numeric(df['Registration amount'], errors='coerce') * TAXA_CAMBIO
            if 'Discounts amount' in df.columns:
                df['Discounts amount'] = pd.to_numeric(df['Discounts amount'], errors='coerce') * TAXA_CAMBIO
            df['Moeda'] = 'USD'
            return df
        else:
            st.error(f"Erro ao carregar dados de {ano}. Status code: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Erro ao carregar dados de {ano}: {e}")
        return None

# URLs dos dados estáticos (2023 e 2024) em formato RAW
url_2023 = "https://github.com/rafaelnmiranda/dash_utmb/raw/815dda1e46bf0b731212e12a365ad169dc4d4e23/UTMB%20-%202023%20-%20USD.xlsx"
url_2024 = "https://github.com/rafaelnmiranda/dash_utmb/raw/815dda1e46bf0b731212e12a365ad169dc4d4e23/UTMB%20-%202024%20-%20USD.xlsx"

# Carrega os dados de 2023 e 2024
df_2023 = load_static_data(2023, url_2023)
df_2024 = load_static_data(2024, url_2024)

# Upload dos arquivos de 2025 (permitindo múltiplos arquivos)
st.subheader("Carregue os arquivos de inscrições 2025 (USD e BRL)")
uploaded_files = st.file_uploader("Selecione os arquivos de 2025", type=["xlsx"], accept_multiple_files=True)

dfs_2025 = []
if uploaded_files:
    for file in uploaded_files:
        if "USD" in file.name.upper():
            moeda = "USD"
        elif "R$" in file.name or "BRL" in file.name.upper():
            moeda = "BRL"
        else:
            moeda = "Desconhecido"
        try:
            df = pd.read_excel(file, sheet_name="registrations_sheet_name")
            df['Registration date'] = pd.to_datetime(df['Registration date'], errors='coerce')
            df['Ano'] = 2025  # Inscrições de 2025
            df['Moeda'] = moeda
            if 'Registration amount' in df.columns:
                df['Registration amount'] = pd.to_numeric(df['Registration amount'], errors='coerce')
                if moeda == "USD":
                    df['Registration amount'] = df['Registration amount'] * TAXA_CAMBIO
            if 'Discounts amount' in df.columns:
                df['Discounts amount'] = pd.to_numeric(df['Discounts amount'], errors='coerce')
                if moeda == "USD":
                    df['Discounts amount'] = df['Discounts amount'] * TAXA_CAMBIO
            dfs_2025.append(df)
        except Exception as e:
            st.error(f"Erro ao carregar o arquivo {file.name}: {e}")

if dfs_2025:
    df_2025 = pd.concat(dfs_2025, ignore_index=True)
else:
    df_2025 = None
    st.warning("Aguarde o upload dos arquivos de 2025.")

# Unir os dados de 2023, 2024 e 2025
dataframes = []
if df_2023 is not None:
    dataframes.append(df_2023)
if df_2024 is not None:
    dataframes.append(df_2024)
if df_2025 is not None:
    dataframes.append(df_2025)

if dataframes:
    df_total = pd.concat(dataframes, ignore_index=True)
    # Remover registros da prova KIDS
    if 'Competition' in df_total.columns:
        df_total = df_total[~df_total['Competition'].str.contains("KIDS", na=False, case=False)]
    else:
        st.error("A coluna 'Competition' não foi encontrada no dataset.")
else:
    st.error("Nenhum dado foi carregado para análise.")
    st.stop()

# -----------------------------------------------------
# 2. Customização Visual (CSS)
# -----------------------------------------------------
st.markdown(
    """
    <style>
    .titulo {
        font-size: 28px;
        color: #002D74;
        font-weight: bold;
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
st.markdown('<p class="titulo">Dashboard de Inscrições - Paraty Brazil by UTMB</p>', unsafe_allow_html=True)

# -----------------------------------------------------
# 3. Resumo Geral (Início do Dashboard)
# -----------------------------------------------------
st.header("Resumo Geral")

# 3.1 Tabela de Inscritos com Metas (para 2025)
st.subheader("Metas de Inscritos por Percurso - 2025")
metas = pd.DataFrame({
    "Percurso": ["FUN 7km", "PTR 20", "PTR 35", "PTR 55", "UTSB 100", "Total"],
    "Meta 2025": [900, 900, 620, 770, 310, 3500]
})
df_2025_inscritos = df_total[df_total['Ano'] == 2025]
inscritos_2025 = df_2025_inscritos['Competition'].value_counts().reset_index()
inscritos_2025.columns = ['Percurso', 'Inscritos']
total_inscritos_2025 = inscritos_2025['Inscritos'].sum()
# Adiciona linha total, se não houver
if 'Total' not in inscritos_2025['Percurso'].values:
    inscritos_2025 = inscritos_2025.append({'Percurso': 'Total', 'Inscritos': total_inscritos_2025}, ignore_index=True)
metas_df = metas.merge(inscritos_2025, on="Percurso", how="left").fillna(0)
# Garantindo que quantidades são inteiros e percentuais sem casas decimais:
metas_df['Meta 2025'] = metas_df['Meta 2025'].apply(lambda x: format_integer(x))
metas_df['Inscritos'] = metas_df['Inscritos'].apply(lambda x: format_integer(x))
metas_df['% da Meta'] = ((metas_df['Inscritos'].astype(float) / metas_df['Meta 2025'].astype(float)) * 100).fillna(0).apply(lambda x: format_percentage(x))
st.table(metas_df)

# 3.2 Percentual de Mulheres Inscritas (coluna Gender)
if 'Gender' in df_total.columns:
    total_reg = df_total.shape[0]
    num_mulheres = df_total['Gender'].str.strip().str.upper().isin(['F', 'FEMALE']).sum()
    perc_mulheres = (num_mulheres / total_reg) * 100
    st.metric("% de Mulheres Inscritas", format_percentage(perc_mulheres))
else:
    st.info("Coluna 'Gender' não encontrada.")

# 3.3 Número de Países Diferentes (coluna Nationality)
if 'Nationality' in df_total.columns:
    num_paises = df_total['Nationality'].nunique()
    st.metric("Número de Países Diferentes", format_integer(num_paises))
else:
    st.info("Coluna 'Nationality' não encontrada.")

# 3.4 Tabela de Participação dos Atletas (baseado em Email)
st.subheader("Participação dos Atletas por Ano (baseado em Email)")
set_2023 = set(df_total[df_total['Ano'] == 2023]['Email'].dropna().unique())
set_2024 = set(df_total[df_total['Ano'] == 2024]['Email'].dropna().unique())
set_2025 = set(df_total[df_total['Ano'] == 2025]['Email'].dropna().unique())

only_2023      = set_2023 - set_2024 - set_2025
only_2024      = set_2024 - set_2023 - set_2025
only_2025      = set_2025 - set_2023 - set_2024
only_2023_2024 = (set_2023 & set_2024) - set_2025
only_2023_2025 = (set_2023 & set_2025) - set_2024
only_2024_2025 = (set_2024 & set_2025) - set_2023
all_three      = set_2023 & set_2024 & set_2025

participation = pd.DataFrame({
    'Participação': ['Apenas 2023', 'Apenas 2024', 'Apenas 2025',
                     'Apenas 2023 e 2024', 'Apenas 2023 e 2025',
                     'Apenas 2024 e 2025', '2023, 2024 e 2025'],
    'Quantidade': [len(only_2023), len(only_2024), len(only_2025),
                   len(only_2023_2024), len(only_2023_2025),
                   len(only_2024_2025), len(all_three)]
})
# Adiciona linha total se necessário (soma de todas as quantidades)
participation = participation.append({'Participação': 'Total', 'Quantidade': participation['Quantidade'].sum()}, ignore_index=True)
st.table(participation)

# 3.5 Top 10 Países Inscritos em 2025
if 'Nationality' in df_total.columns:
    df_2025_only = df_total[df_total['Ano'] == 2025]
    top_paises = df_2025_only['Nationality'].value_counts().head(10).reset_index()
    top_paises.columns = ['País', 'Inscritos']
    top_paises['Inscritos'] = top_paises['Inscritos'].apply(format_integer)
    st.subheader("Top 10 Países Inscritos em 2025")
    # Adiciona linha total
    total_top = top_paises['Inscritos'].astype(int).sum()
    top_paises = top_paises.append({'País': 'Total', 'Inscritos': format_integer(total_top)}, ignore_index=True)
    st.table(top_paises)
else:
    st.info("Coluna 'Nationality' não encontrada.")

# 3.6 Comparativo de Inscrições até Data Base
if df_2025 is not None and 'Registration date' in df_2025.columns:
    data_base = df_2025['Registration date'].max()
else:
    data_base = pd.Timestamp.today()
df_comp = df_total[df_total['Registration date'] <= data_base]
inscritos_comp = df_comp.groupby('Ano').size().reset_index(name=f'Inscritos até {data_base.strftime("%d/%m/%Y")}')
inscritos_comp[f'Inscritos até {data_base.strftime("%d/%m/%Y")}'] = inscritos_comp[f'Inscritos até {data_base.strftime("%d/%m/%Y")}'].apply(format_integer)
st.subheader(f"Comparativo de Inscrições até {data_base.strftime('%d/%m/%Y')}")
# Linha total (soma dos inscritos dos anos)
total_comp = inscritos_comp[f'Inscritos até {data_base.strftime("%d/%m/%Y")}'].astype(int).sum()
inscritos_comp = inscritos_comp.append({'Ano': 'Total', f'Inscritos até {data_base.strftime("%d/%m/%Y")}': format_integer(total_comp)}, ignore_index=True)
st.table(inscritos_comp)

# 3.7 Média Móvel e Projeção para 15 de Agosto (para 2025)
if df_2025 is not None and 'Registration date' in df_2025.columns:
    df_2025_daily = df_2025.copy()
    df_2025_daily['Date'] = df_2025_daily['Registration date'].dt.date
    daily_counts = df_2025_daily.groupby('Date').size().reset_index(name='Inscritos')
    daily_counts['Date'] = pd.to_datetime(daily_counts['Date'])
    daily_counts = daily_counts.sort_values('Date')
    daily_counts.set_index('Date', inplace=True)
    daily_counts['MM_7'] = daily_counts['Inscritos'].rolling(window=7).mean()
    daily_counts['MM_15'] = daily_counts['Inscritos'].rolling(window=15).mean()
    daily_counts['MM_30'] = daily_counts['Inscritos'].rolling(window=30).mean()
    last_date = daily_counts.index.max()
    projection_date = pd.Timestamp(year=2025, month=8, day=15)
    delta_days = (projection_date - last_date).days
    if delta_days < 0:
        delta_days = 0
    total_2025_current = df_2025.shape[0]
    mm7   = daily_counts.loc[last_date, 'MM_7'] if pd.notna(daily_counts.loc[last_date, 'MM_7']) else 0
    mm15  = daily_counts.loc[last_date, 'MM_15'] if pd.notna(daily_counts.loc[last_date, 'MM_15']) else 0
    mm30  = daily_counts.loc[last_date, 'MM_30'] if pd.notna(daily_counts.loc[last_date, 'MM_30']) else 0
    # Projeção: soma o total atual com (média * dias faltantes)
    proj_7  = total_2025_current + mm7  * delta_days
    proj_15 = total_2025_current + mm15 * delta_days
    proj_30 = total_2025_current + mm30 * delta_days
    projection_df = pd.DataFrame({
        'Média Móvel': ['7 dias', '15 dias', '30 dias'],
        'Valor Médio Diário': [format_integer(mm7), format_integer(mm15), format_integer(mm30)],
        'Projeção Inscritos (15/08/2025)': [format_integer(proj_7), format_integer(proj_15), format_integer(proj_30)]
    })
    st.subheader("Média Móvel e Projeção de Inscritos (até 15/08/2025)")
    st.table(projection_df)
else:
    st.info("Dados de 2025 não disponíveis para cálculo de média móvel e projeção.")

# -----------------------------------------------------
# 4. Filtro por Ano para Análises Detalhadas
# -----------------------------------------------------
anos_disponiveis = sorted(df_total['Ano'].dropna().unique())
opcao_ano = st.sidebar.multiselect("Selecione o(s) ano(s) para análise detalhada:", anos_disponiveis, default=anos_disponiveis)
df_filtrado = df_total[df_total['Ano'].isin(opcao_ano)]
if 'Registration amount' in df_filtrado.columns and 'Discounts amount' in df_filtrado.columns:
    df_filtrado['Total Amount'] = df_filtrado['Registration amount'] + df_filtrado['Discounts amount']
else:
    st.error("As colunas financeiras não foram encontradas.")
    st.stop()

# Métricas principais em 3 colunas
col1, col2, col3 = st.columns(3)
receita_bruta = df_filtrado['Total Amount'].sum()
receita_liquida = df_filtrado['Registration amount'].sum()
total_inscritos = df_filtrado.shape[0]
col1.metric("Receita Bruta (R$)", format_currency(receita_bruta))
col2.metric("Receita Líquida (R$)", format_currency(receita_liquida))
col3.metric("Total Inscritos", format_integer(total_inscritos))

# -----------------------------------------------------
# 5. Layout com Abas para Análises Detalhadas
# -----------------------------------------------------
tab1, tab2, tab3 = st.tabs(["Inscritos", "Análises Financeiras", "Comparativo Entre Anos"])

with tab1:
    st.subheader("Dados de Inscritos por Competição")
    if 'Competition' in df_filtrado.columns:
        inscritos = df_filtrado['Competition'].value_counts().reset_index()
        inscritos.columns = ['Percurso', 'Inscritos']
        # Adiciona total se necessário
        total_insc = inscritos['Inscritos'].astype(int).sum()
        inscritos = inscritos.append({'Percurso': 'Total', 'Inscritos': format_integer(total_insc)}, ignore_index=True)
        st.table(inscritos)
    else:
        st.error("Coluna 'Competition' ausente.")

with tab2:
    st.subheader("Análises Financeiras")
    if 'Discount code' in df_filtrado.columns:
        coupon_groups = df_filtrado['Discount code'].fillna('OUTROS').str.upper().str.extract(r'(PTY25|TG25|TP25|GP25)', expand=False)
        df_filtrado['Coupon Group'] = coupon_groups.fillna('OUTROS')
        coupon_summary = df_filtrado.groupby('Coupon Group').agg(
            Total_Discounts=pd.NamedAgg(column='Discounts amount', aggfunc='sum'),
            Quantidade_Descontos=pd.NamedAgg(column='Discounts amount', aggfunc=lambda x: (x > 0).sum())
        ).reset_index()
        # Formata os valores monetários e as quantidades
        coupon_summary['Total_Discounts'] = coupon_summary['Total_Discounts'].apply(format_currency)
        coupon_summary['Quantidade_Descontos'] = coupon_summary['Quantidade_Descontos'].apply(format_integer)
        # Adiciona linha total se possível
        total_desc = coupon_summary['Quantidade_Descontos'].astype(int).sum()
        coupon_summary = coupon_summary.append({'Coupon Group': 'Total', 'Total_Discounts': '', 'Quantidade_Descontos': format_integer(total_desc)}, ignore_index=True)
        st.table(coupon_summary)
    else:
        st.info("Não há informações de cupom de desconto na base.")
    
    revenue_by_competition = df_filtrado.groupby('Competition').agg(
        Total_Inscritos=pd.NamedAgg(column='Competition', aggfunc='size'),
        Receita_Bruta=pd.NamedAgg(column='Total Amount', aggfunc='sum'),
        Receita_Líquida=pd.NamedAgg(column='Registration amount', aggfunc='sum'),
        Total_Descontos=pd.NamedAgg(column='Discounts amount', aggfunc='sum')
    ).reset_index()
    # Formatação: valores monetários e inteiros
    revenue_by_competition['Total_Inscritos'] = revenue_by_competition['Total_Inscritos'].apply(format_integer)
    revenue_by_competition['Receita_Bruta'] = revenue_by_competition['Receita_Bruta'].apply(format_currency)
    revenue_by_competition['Receita_Líquida'] = revenue_by_competition['Receita_Líquida'].apply(format_currency)
    revenue_by_competition['Total_Descontos'] = revenue_by_competition['Total_Descontos'].apply(format_currency)
    # Adiciona linha total
    total_recs = revenue_by_competition['Total_Inscritos'].astype(int).sum()
    receita_bruta_total = revenue_by_competition['Receita_Bruta'].apply(lambda x: float(x.replace("R$ ", "").replace(".", "")) if isinstance(x, str) and x.strip() != "" else 0).sum()
    receita_liquida_total = revenue_by_competition['Receita_Líquida'].apply(lambda x: float(x.replace("R$ ", "").replace(".", "")) if isinstance(x, str) and x.strip() != "" else 0).sum()
    total_descontos_total = revenue_by_competition['Total_Descontos'].apply(lambda x: float(x.replace("R$ ", "").replace(".", "")) if isinstance(x, str) and x.strip() != "" else 0).sum()
    total_row = pd.DataFrame({
        'Competition': ['Total'],
        'Total_Inscritos': [format_integer(total_recs)],
        'Receita_Bruta': [format_currency(receita_bruta_total)],
        'Receita_Líquida': [format_currency(receita_liquida_total)],
        'Total_Descontos': [format_currency(total_descontos_total)]
    })
    revenue_by_competition = pd.concat([revenue_by_competition, total_row], ignore_index=True)
    st.table(revenue_by_competition)

with tab3:
    st.subheader("Comparativo de Receita por Competição (por Ano)")
    comparativo = df_filtrado.groupby(['Ano', 'Competition'])['Registration amount'].sum().reset_index()
    # Aplica formatação monetária à coluna de receita
    comparativo['Registration amount'] = comparativo['Registration amount'].apply(format_currency)
    fig_comparativo = px.bar(comparativo, x='Competition', y='Registration amount', color='Ano',
                              barmode='group', title="Comparativo de Receita por Competição (R$)")
    st.plotly_chart(fig_comparativo)

# Gráfico adicional de inscrições mensais por ano (se disponível)
if 'Registration date' in df_filtrado.columns:
    df_filtrado['Mês'] = df_filtrado['Registration date'].dt.month
    inscricoes_mensais = df_filtrado.groupby(['Ano', 'Mês']).size().reset_index(name='Inscrições')
    inscricoes_mensais['Inscrições'] = inscricoes_mensais['Inscrições'].apply(format_integer)
    fig_temporal = px.line(inscricoes_mensais, x='Mês', y='Inscrições', color='Ano',
                           title="Inscrições Mensais por Ano")
    st.plotly_chart(fig_temporal)

st.markdown("***Fim do Dashboard***")
