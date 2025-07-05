import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import requests
from io import BytesIO
import unicodedata
import re
from rapidfuzz import fuzz, process
import os
from datetime import datetime

# Configuração da página
st.set_page_config(
    page_title="Dashboard Marketing - Paraty Brazil by UTMB 2025",
    page_icon="🏃‍♂️",
    layout="wide"
)

# ─── FUNÇÕES AUXILIARES ───

def norm_text(text):
    """Normaliza texto removendo acentos e caracteres especiais"""
    t = unicodedata.normalize('NFKD', str(text))
    t = ''.join(c for c in t if not unicodedata.combining(c))
    return re.sub(r'[^a-z0-9\s]', '', t.lower().strip())

def format_percentage(val):
    """Formata valor como percentual"""
    try:
        return f"{float(val):.1f}%".replace('.', ',')
    except Exception:
        return val

def format_integer(val):
    """Formata valor como inteiro"""
    try:
        return str(int(round(float(val))))
    except Exception:
        return val

def standardize_nationality(value):
    """Padroniza nacionalidade"""
    if pd.isnull(value):
        return value
    value = value.strip().upper()
    mapping = {"BRASIL": "BR", "BRAZIL": "BR"}
    return mapping.get(value, value)

def standardize_competition(value):
    """Padroniza nomes das competições"""
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

def consolidate_teams(teams_series, similarity_threshold=80):
    """
    Consolida nomes de assessorias esportivas por similaridade
    """
    if teams_series.empty:
        return teams_series
    
    # Remove valores nulos e normaliza
    teams_clean = teams_series.dropna().astype(str).str.strip()
    teams_clean = teams_clean[teams_clean != 'nan']
    
    if teams_clean.empty:
        return teams_series
    
    # Cria dicionário de mapeamento
    team_mapping = {}
    processed_teams = set()
    
    for team in teams_clean.unique():
        if team in processed_teams:
            continue
            
        # Encontra times similares
        similar_teams = process.extract(
            team, 
            teams_clean.unique(), 
            limit=len(teams_clean.unique()),
            score_cutoff=similarity_threshold
        )
        
        # Agrupa times similares
        for similar_team, score in similar_teams:
            if similar_team not in processed_teams:
                team_mapping[similar_team] = team
                processed_teams.add(similar_team)
    
    # Aplica o mapeamento
    consolidated = teams_series.map(team_mapping).fillna(teams_series)
    return consolidated

# ─── CARREGAMENTO DE DADOS ───

@st.cache_data(show_spinner=False)
def load_ibge_municipios():
    """Carrega dados de municípios do IBGE"""
    IBGE_URL = (
        "https://raw.githubusercontent.com/rafaelnmiranda/dash_utmb/"
        "de2e7125c2a3c08c7c41be14c43e528b43c2ea58/municipios_IBGE.xlsx"
    )
    try:
        resp = requests.get(IBGE_URL, timeout=10)
        df = pd.read_excel(BytesIO(resp.content), engine='openpyxl')
        df['City_norm'] = df['City'].apply(norm_text)
        df = df.drop_duplicates(subset=['City_norm'], keep='first')
        return df
    except Exception as e:
        st.error(f"Erro ao carregar dados do IBGE: {str(e)}")
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_static_data(ano, url):
    """Carrega dados estáticos dos anos anteriores"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        df = pd.read_excel(BytesIO(response.content), sheet_name=0)
        df['Registration date'] = pd.to_datetime(df['Registration date'], errors='coerce')
        df['Ano'] = ano
        df['Moeda'] = 'USD'
        return df
    except Exception as e:
        st.error(f"Erro ao carregar dados de {ano}: {str(e)}")
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def preprocess_data(dfs, ibge_df, taxa_cambio=5.5):
    """Pré-processa os dados"""
    processed_dfs = []
    for df in dfs:
        if df.empty:
            continue
        df = df.copy()
        
        df['Registration date'] = pd.to_datetime(df['Registration date'], errors='coerce')
        df['City'] = df['City'].astype(str)
        df['Nationality'] = df['Nationality'].apply(standardize_nationality)
        df['Competition'] = df['Competition'].apply(standardize_competition)
        
        # Exclui KIDS
        df = df[~df['Competition'].str.contains("KIDS", na=False, case=False)]
        
        processed_dfs.append(df)
    
    if not processed_dfs:
        return pd.DataFrame()
    
    return pd.concat(processed_dfs, ignore_index=True)

# ─── FUNÇÕES DE MÉTRICAS ───

def calculate_competition_metrics(df_2025):
    """Calcula métricas por competição"""
    if df_2025.empty:
        return pd.DataFrame()
    
    # Metas por percurso
    metas = {
        "FUN 7KM": 900,
        "PTR 20": 900,
        "PTR 35": 620,
        "PTR 55": 770,
        "UTSB 100": 310
    }
    
    # Conta inscritos por competição
    inscritos_por_competicao = df_2025['Competition'].value_counts().reset_index()
    inscritos_por_competicao.columns = ['Percurso', 'Inscritos']
    
    # Adiciona metas e calcula percentual
    inscritos_por_competicao['Meta'] = inscritos_por_competicao['Percurso'].map(metas)
    inscritos_por_competicao['% da Meta'] = (
        (inscritos_por_competicao['Inscritos'] / inscritos_por_competicao['Meta']) * 100
    ).round(1)
    
    return inscritos_por_competicao

def calculate_gender_metrics(df_2025):
    """Calcula métricas de gênero"""
    if df_2025.empty:
        return 0, 0
    
    total_inscritos = len(df_2025)
    mulheres = df_2025['Gender'].str.strip().str.upper().isin(['F', 'FEMALE']).sum()
    percentual_mulheres = (mulheres / total_inscritos * 100) if total_inscritos > 0 else 0
    
    return mulheres, percentual_mulheres

def calculate_nationality_metrics(df_2025):
    """Calcula métricas de nacionalidade"""
    if df_2025.empty:
        return 0, 0, 0, 0, pd.DataFrame()
    
    total_inscritos = len(df_2025)
    df_2025['Nat_std'] = df_2025['Nationality'].apply(standardize_nationality)
    
    # Brasileiros
    brasileiros = df_2025[df_2025['Nat_std'] == 'BR'].shape[0]
    percentual_brasileiros = (brasileiros / total_inscritos * 100) if total_inscritos > 0 else 0
    
    # Estrangeiros
    estrangeiros = total_inscritos - brasileiros
    percentual_estrangeiros = (estrangeiros / total_inscritos * 100) if total_inscritos > 0 else 0
    
    # Países diferentes
    paises_diferentes = df_2025['Nat_std'].nunique()
    
    # Lista de países
    paises_lista = df_2025['Nat_std'].value_counts().reset_index()
    paises_lista.columns = ['País', 'Inscritos']
    
    return brasileiros, percentual_brasileiros, estrangeiros, percentual_estrangeiros, paises_lista, paises_diferentes

def calculate_brazilian_cities_metrics(df_2025, ibge_df):
    """Calcula métricas de cidades brasileiras"""
    if df_2025.empty or ibge_df.empty:
        return 0, pd.DataFrame()
    
    # Filtra apenas brasileiros
    df_br = df_2025[df_2025['Country'].str.lower() == 'brazil'].copy()
    
    if df_br.empty:
        return 0, pd.DataFrame()
    
    # Corrige nomes das cidades
    df_br['City'] = df_br['City'].astype(str)
    
    # Conta municípios únicos
    total_municipios = df_br['City'].nunique()
    
    # Top 5 cidades
    top_cidades = df_br['City'].value_counts().head(5).reset_index()
    top_cidades.columns = ['Cidade', 'Inscritos']
    
    return total_municipios, top_cidades

def calculate_teams_metrics(df_2025):
    """Calcula métricas de assessorias esportivas"""
    if df_2025.empty or 'Team / Club' not in df_2025.columns:
        return pd.DataFrame()
    
    # Consolida nomes das assessorias
    df_2025['Team_Consolidated'] = consolidate_teams(df_2025['Team / Club'])
    
    # Remove valores vazios
    teams_data = df_2025[df_2025['Team_Consolidated'].notna() & 
                        (df_2025['Team_Consolidated'] != '') & 
                        (df_2025['Team_Consolidated'] != 'nan')]
    
    if teams_data.empty:
        return pd.DataFrame()
    
    # Top 5 assessorias
    top_teams = teams_data['Team_Consolidated'].value_counts().head(5).reset_index()
    top_teams.columns = ['Assessoria', 'Atletas']
    
    return top_teams

# ─── FUNÇÕES DE VISUALIZAÇÃO ───

def create_gauge_chart(value, title, target=50):
    """Cria gráfico gauge"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title},
        delta={'reference': target},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, target*0.8], 'color': "lightgray"},
                {'range': [target*0.8, target], 'color': "yellow"},
                {'range': [target, 100], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': target
            }
        }
    ))
    fig.update_layout(height=300)
    return fig

# ─── CARREGAMENTO E PROCESSAMENTO DE DADOS ───

st.title("🏃‍♂️ Dashboard Marketing - Paraty Brazil by UTMB 2025")

# Data da última alteração
arquivo_atual = __file__
timestamp_modificacao = os.path.getmtime(arquivo_atual)
data_alteracao = datetime.fromtimestamp(timestamp_modificacao).strftime('%d/%m/%Y')
st.markdown(
    f'<div style="position: absolute; top: 10px; right: 20px; font-size: 10px; color: #666; font-style: italic;">'
    f'Última atualização: {data_alteracao}</div>',
    unsafe_allow_html=True
)

with st.spinner("🔄 Carregando dados..."):
    # Carrega dados do IBGE
    ibge_df = load_ibge_municipios()
    
    # Carrega dados dos anos anteriores
    url_2023 = "https://github.com/rafaelnmiranda/dash_utmb/raw/815dda1e46bf0b731212e12a365ad169dc4d4e23/UTMB%20-%202023%20-%20USD.xlsx"
    url_2024 = "https://github.com/rafaelnmiranda/dash_utmb/raw/815dda1e46bf0b731212e12a365ad169dc4d4e23/UTMB%20-%202024%20-%20USD.xlsx"
    
    df_2023 = load_static_data(2023, url_2023)
    df_2024 = load_static_data(2024, url_2024)
    
    # Upload de arquivos 2025
    uploaded_files = st.file_uploader(
        "📁 Carregue os arquivos de 2025 (USD e BRL)", 
        type=["xlsx"], 
        accept_multiple_files=True
    )
    
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

# Verifica se há dados de 2025
if not dfs_2025:
    st.error("❌ Por favor, faça o upload dos arquivos de 2025.")
    st.stop()

# Processa dados
df_total = preprocess_data([df_2023, df_2024] + dfs_2025, ibge_df)
df_2025 = df_total[df_total['Ano'] == 2025]

if df_2025.empty:
    st.error("❌ Nenhum dado válido encontrado para 2025.")
    st.stop()

# ─── DASHBOARD ───

st.header("📊 Métricas Principais")

# 1️⃣ Total de inscritos por percurso
st.subheader("1️⃣ Total de Inscritos por Percurso")
competition_metrics = calculate_competition_metrics(df_2025)

if not competition_metrics.empty:
    # Exibe tabela
    st.dataframe(
        competition_metrics.style.format({
            'Inscritos': '{:,.0f}',
            'Meta': '{:,.0f}',
            '% da Meta': '{:.1f}%'
        }),
        use_container_width=True
    )
    
    # Gráfico de barras
    fig_competition = px.bar(
        competition_metrics,
        x='Percurso',
        y='Inscritos',
        text='Inscritos',
        title="Inscritos por Percurso",
        color='% da Meta',
        color_continuous_scale='RdYlGn'
    )
    fig_competition.update_traces(textposition='outside')
    st.plotly_chart(fig_competition, use_container_width=True)

st.divider()

# 2️⃣ Percentual de mulheres
st.subheader("2️⃣ Percentual de Mulheres")
mulheres, percentual_mulheres = calculate_gender_metrics(df_2025)

col1, col2 = st.columns(2)
with col1:
    st.metric("Total de Mulheres", format_integer(mulheres))
    st.metric("Percentual de Mulheres", format_percentage(percentual_mulheres))

with col2:
    fig_gender = create_gauge_chart(percentual_mulheres, "Percentual de Mulheres", 50)
    st.plotly_chart(fig_gender, use_container_width=True)

st.divider()

# 3️⃣ Percentual de estrangeiros
st.subheader("3️⃣ Percentual de Estrangeiros")
brasileiros, perc_brasileiros, estrangeiros, perc_estrangeiros, paises_lista, paises_diferentes = calculate_nationality_metrics(df_2025)

col1, col2 = st.columns(2)
with col1:
    st.metric("Total de Estrangeiros", format_integer(estrangeiros))
    st.metric("Percentual de Estrangeiros", format_percentage(perc_estrangeiros))
    st.metric("Países Diferentes", format_integer(paises_diferentes))

with col2:
    fig_foreign = create_gauge_chart(perc_estrangeiros, "Percentual de Estrangeiros", 30)
    st.plotly_chart(fig_foreign, use_container_width=True)

# Lista de países
with st.expander("🌍 Lista Completa de Países"):
    if not paises_lista.empty:
        st.dataframe(paises_lista, use_container_width=True)

st.divider()

# 4️⃣ Percentual e número de brasileiros
st.subheader("4️⃣ Percentual e Número de Brasileiros")
col1, col2 = st.columns(2)
with col1:
    st.metric("Total de Brasileiros", format_integer(brasileiros))
    st.metric("Percentual de Brasileiros", format_percentage(perc_brasileiros))

with col2:
    fig_brazil = create_gauge_chart(perc_brasileiros, "Percentual de Brasileiros", 70)
    st.plotly_chart(fig_brazil, use_container_width=True)

st.divider()

# 5️⃣ Municípios brasileiros
st.subheader("5️⃣ Municípios Brasileiros")
total_municipios, top_cidades = calculate_brazilian_cities_metrics(df_2025, ibge_df)

col1, col2 = st.columns(2)
with col1:
    st.metric("Total de Municípios", format_integer(total_municipios))

with col2:
    if not top_cidades.empty:
        st.subheader("🏆 Top 5 Municípios")
        st.dataframe(top_cidades, use_container_width=True)

st.divider()

# 6️⃣ Assessorias esportivas
st.subheader("6️⃣ Assessorias Esportivas")
top_teams = calculate_teams_metrics(df_2025)

if not top_teams.empty:
    st.subheader("🏆 Top 5 Assessorias")
    st.dataframe(top_teams, use_container_width=True)
    
    # Gráfico de barras
    fig_teams = px.bar(
        top_teams,
        x='Assessoria',
        y='Atletas',
        text='Atletas',
        title="Top 5 Assessorias Esportivas"
    )
    fig_teams.update_traces(textposition='outside')
    fig_teams.update_xaxes(tickangle=45)
    st.plotly_chart(fig_teams, use_container_width=True)
else:
    st.info("ℹ️ Nenhuma assessoria esportiva encontrada nos dados.")

st.divider()

# ─── RESUMO EXECUTIVO ───
st.header("📋 Resumo Executivo")

total_inscritos = len(df_2025)
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total de Inscritos", format_integer(total_inscritos))

with col2:
    st.metric("Mulheres", f"{format_integer(mulheres)} ({format_percentage(percentual_mulheres)})")

with col3:
    st.metric("Estrangeiros", f"{format_integer(estrangeiros)} ({format_percentage(perc_estrangeiros)})")

with col4:
    st.metric("Países", format_integer(paises_diferentes))

st.markdown("---")
st.markdown("*Dashboard de Marketing - Paraty Brazil by UTMB 2025*")
