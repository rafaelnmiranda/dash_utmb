import streamlit as st
import pandas as pd
import plotly.express as px

# Título do Dashboard
st.title("Dashboard de Inscrições - Paraty Brazil by UTMB")

# Upload da Planilha
uploaded_file = st.file_uploader("Carregue sua planilha", type=["xlsx"])

if uploaded_file:
    # Carrega os dados
    df = pd.read_excel(uploaded_file, sheet_name="registrations_sheet_name")

    # Filtros Básicos
    st.sidebar.header("Filtros")
    selected_country = st.sidebar.selectbox("País", df['Country'].unique())
    filtered_df = df[df['Country'] == selected_country]

    # Gráfico de Atletas por Cidade
    st.header("Atletas por Cidade")
    city_counts = filtered_df['City'].value_counts().reset_index()
    city_counts.columns = ['City', 'Total Athletes']
    fig = px.bar(city_counts, x='City', y='Total Athletes', title="Distribuição de Atletas por Cidade")
    st.plotly_chart(fig)

    # Porcentagem de Brasileiros x Estrangeiros
    st.header("Percentual de Brasileiros x Estrangeiros")
    brazilian_count = (df['Country'].str.upper().isin(['BRAZIL', 'BR'])).sum()
    foreign_count = len(df) - brazilian_count
    percentages = pd.DataFrame({
        "Tipo": ["Brasileiros", "Estrangeiros"],
        "Percentual": [brazilian_count / len(df) * 100, foreign_count / len(df) * 100]
    })
    fig_pie = px.pie(percentages, names='Tipo', values='Percentual', title="Distribuição de Brasileiros e Estrangeiros")
    st.plotly_chart(fig_pie)
