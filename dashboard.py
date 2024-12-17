import streamlit as st
import pandas as pd
import plotly.express as px
from rapidfuzz import process, fuzz
import requests
from io import BytesIO

# Título do Dashboard
st.title("Dashboard de Inscrições - Paraty Brazil by UTMB 2025")

# Função para baixar a base de municipios do GitHub
def load_ibge_municipios():
    url = "https://github.com/rafaelnmiranda/dash_utmb/raw/main/municipios_IBGE.xlxs"
    response = requests.get(url)
    if response.status_code == 200:
        return pd.read_excel(BytesIO(response.content))
    else:
        st.error("Erro ao carregar a base de municípios do IBGE. Verifique o link e tente novamente.")
        return None

# Função para realizar o fuzzy matching
def fuzzy_match_cities(city_name, official_cities):
    if pd.isna(city_name):
        return None
    match, score = process.extractOne(city_name, official_cities, scorer=fuzz.token_sort_ratio)
    return match if score > 80 else city_name

# Upload da Planilha
uploaded_file = st.file_uploader("Carregue sua planilha de inscrições", type=["xlsx"])

if uploaded_file:
    # Carregar os dados
    df = pd.read_excel(uploaded_file, sheet_name="registrations_sheet_name")
    df['Registration date'] = pd.to_datetime(df['Registration date'], errors='coerce')

    # ------------------- METAS DE INSCRITOS -------------------
    st.header("\ud83d\udcca Metas de Inscritos por Percurso")
    metas = pd.DataFrame({
        "Percurso": ["FUN 7km", "PTR 20", "PTR 35", "PTR 55", "UTSB 100", "Total"],
        "Meta 2025": [900, 900, 620, 770, 310, 3500]
    })

    # Contar inscritos por percurso
    inscritos = df['Competition'].value_counts().reset_index()
    inscritos.columns = ['Percurso', 'Inscritos']
    total_inscritos = inscritos['Inscritos'].sum()
    inscritos.loc[len(inscritos)] = ["Total", total_inscritos]

    # Merge Metas e Real
    metas_df = metas.merge(inscritos, on="Percurso", how="left").fillna(0)
    metas_df['Inscritos'] = metas_df['Inscritos'].astype(int)
    metas_df['% da Meta'] = (metas_df['Inscritos'] / metas_df['Meta 2025'] * 100).fillna(0).round(2).astype(str) + '%'
    st.table(metas_df)

    # ------------------- META DE MULHERES -------------------
    st.header("\ud83d\udc69 Percentual de Mulheres (Meta: 40%)")
    total_mulheres = df[df['T-shirt size (woman)'].notnull()].shape[0]
    total_homens = len(df) - total_mulheres

    # Calcular percentuais
    perc_mulheres = (total_mulheres / len(df)) * 100
    perc_homens = 100 - perc_mulheres

    # Criar gráfico de barras horizontais com percentuais
    fig_gender = px.bar(
        x=[perc_mulheres, perc_homens],
        y=["Mulheres", "Homens"],
        orientation='h',
        text=[f"{perc_mulheres:.2f}%", f"{perc_homens:.2f}%"],
        title="Distribuição de Género (%)",
    )
    st.plotly_chart(fig_gender)

    # Mostrar indicador de progresso
    st.metric("Percentual Atual de Mulheres", f"{perc_mulheres:.2f}%", delta=f"{40 - perc_mulheres:.2f}% para meta")

    # ------------------- CORREÇÃO DE CIDADES -------------------
    st.header("\ud83d\udccd Correção Automática de Cidades")
    municipios_ibge = load_ibge_municipios()

    if municipios_ibge is not None:
        # Lista oficial de cidades do IBGE
        official_cities = municipios_ibge['City'].dropna().unique().tolist()
        
        # Aplicar o fuzzy matching
        st.text("Realizando a correção dos nomes das cidades, isso pode levar alguns segundos...")
        df['Corrected City'] = df['City'].apply(lambda x: fuzzy_match_cities(x, official_cities))

        # Filtrar apenas cidades do Brasil
        df_brazil = df[df['Country'].str.upper().isin(['BRAZIL', 'BR'])]

        # Top 10 cidades
        top_cities = df_brazil['Corrected City'].value_counts().head(10).reset_index()
        top_cities.columns = ['City', 'Total Athletes']
        st.table(top_cities)

    # ------------------- NACIONALIDADE DOS ATLETAS -------------------
    st.header("\ud83c\udf0e Nacionalidade dos Atletas")
    total_countries = df['Nationality'].nunique()
    st.write(f"**Número total de países inscritos:** {total_countries}")

    # Top 5 países
    top_countries = df['Nationality'].value_counts().head(5).reset_index()
    top_countries.columns = ['Country', 'Total Athletes']
    st.table(top_countries)

    # Percentual de brasileiros e estrangeiros
    brazilian_count = df['Country'].str.upper().isin(['BRAZIL', 'BR']).sum()
    foreign_count = len(df) - brazilian_count
    st.write(f"**Brasileiros:** {brazilian_count} ({brazilian_count/len(df):.1%})")
    st.write(f"**Estrangeiros:** {foreign_count} ({foreign_count/len(df):.1%})")
