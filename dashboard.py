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

    # ------------------- METAS DE INSCRITOS -------------------
    st.header("📊 Metas de Inscritos por Percurso")
    metas = pd.DataFrame({
        "Percurso": ["FUN 7km", "PTR 20", "PTR 35", "PTR 55", "UTSB 100", "Total"],
        "Meta 2025": [900, 900, 620, 770, 310, 3500]
    })

    inscritos = df['Competition'].value_counts().reset_index()
    inscritos.columns = ['Percurso', 'Inscritos']
    total_inscritos = inscritos['Inscritos'].sum()
    inscritos.loc[len(inscritos)] = ["Total", total_inscritos]

    # Merge Metas e Real
    metas_df = metas.merge(inscritos, on="Percurso", how="left").fillna(0)
    metas_df['Inscritos'] = metas_df['Inscritos'].astype(int)
    metas_df['% da Meta'] = (metas_df['Inscritos'] / metas_df['Meta 2025'] * 100).fillna(0).round(2).astype(str) + '%'

    # Exibir tabela com % da meta
    st.table(metas_df)

    # ------------------- META DE MULHERES -------------------
    st.header("👩 Percentual de Mulheres (Meta: 40%)")
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
        title="Distribuição de Gênero (%)",
    )
    st.plotly_chart(fig_gender)

    # Mostrar indicador de progresso
    st.metric("Percentual Atual de Mulheres", f"{perc_mulheres:.2f}%", delta=f"{40 - perc_mulheres:.2f}% para meta")

    # ------------------- CORREÇÃO DE CIDADES -------------------
    st.header("📍 Top 10 Cidades - Brasil")
    # TODO: Implementar correção com base externa via Git
    # Código a ser adicionado futuramente
