import streamlit as st
import pandas as pd
import plotly.express as px

# Título do Dashboard
st.title("Dashboard de Inscrições - Paraty Brazil by UTMB 2025")

# Upload da Planilha
uploaded_file = st.file_uploader("Carregue sua planilha", type=["xlsx"])

if uploaded_file:
    # Carregar os dados
    df = pd.read_excel(uploaded_file, sheet_name="registrations_sheet_name")
    df['Registration date'] = pd.to_datetime(df['Registration date'], errors='coerce')

    # ------------------- CORREÇÃO DAS CIDADES -------------------
    st.header("📍 Correção de Cidades")
    # URL do GitHub com a base de cidades corrigidas (substitua pelo seu link)
    correction_url = "https://github.com/rafaelnmiranda/dash_utmb/blob/main/cidades.csv"
    
    # Carregar a tabela de correção de cidades
    city_corrections = pd.read_csv(correction_url)
    
    # Criar um dicionário para correção
    correction_dict = dict(zip(city_corrections['Cidade_Original'], city_corrections['Cidade_Corrigida']))

    # Aplicar a correção
    df['Corrected City'] = df['City'].replace(correction_dict)

    # Filtrar apenas cidades do Brasil
    df_brazil = df[df['Country'].str.upper().isin(['BRAZIL', 'BR'])]

    # Top 10 cidades
    st.header("🏆 Top 10 Cidades (Brasil)")
    top_cities = df_brazil['Corrected City'].value_counts().head(10).reset_index()
    top_cities.columns = ['City', 'Total Athletes']
    st.table(top_cities)

    # ------------------- NACIONALIDADE DOS ATLETAS -------------------
    st.header("🌎 Nacionalidade dos Atletas")
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
