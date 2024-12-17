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

    # Tratamento da Data
    df['Registration date'] = pd.to_datetime(df['Registration date'])
    df['Day of Week'] = df['Registration date'].dt.day_name()
    df['Hour'] = df['Registration date'].dt.hour

    # KPIs Principais
    st.header("KPIs Principais")
    total_athletes = len(df)
    brazilian_count = df['Country'].str.upper().isin(['BRAZIL', 'BR']).sum()
    foreign_count = total_athletes - brazilian_count

    col1, col2, col3 = st.columns(3)
    col1.metric("Total de Atletas", total_athletes)
    col2.metric("Brasileiros", f"{brazilian_count} ({brazilian_count/total_athletes:.1%})")
    col3.metric("Estrangeiros", f"{foreign_count} ({foreign_count/total_athletes:.1%})")

    # Filtros
    st.sidebar.header("Filtros")
    selected_country = st.sidebar.multiselect("País", df['Country'].unique(), default=df['Country'].unique())
    selected_competition = st.sidebar.multiselect("Percurso", df['Competition'].unique(), default=df['Competition'].unique())
    date_range = st.sidebar.date_input("Período de Inscrições", 
                                       [df['Registration date'].min(), df['Registration date'].max()])

    # Aplicar Filtros
    filtered_df = df[
        (df['Country'].isin(selected_country)) &
        (df['Competition'].isin(selected_competition)) &
        (df['Registration date'].between(date_range[0], date_range[1]))
    ]

    # Gráfico de Inscrições por Cidade
    st.header("Inscrições por Cidade")
    city_counts = filtered_df['City'].value_counts().reset_index()
    city_counts.columns = ['City', 'Total Athletes']
    fig = px.bar(city_counts, x='City', y='Total Athletes', title="Distribuição de Atletas por Cidade")
    st.plotly_chart(fig)

    # Gráfico de Inscrições por Hora
    st.header("Distribuição de Inscrições por Hora")
    hour_counts = filtered_df['Hour'].value_counts().reset_index()
    hour_counts.columns = ['Hour', 'Total Registrations']
    hour_counts = hour_counts.sort_values('Hour')
    fig_hour = px.bar(hour_counts, x='Hour', y='Total Registrations', title="Distribuição de Inscrições por Hora")
    st.plotly_chart(fig_hour)

    # Gráfico de Inscrições ao Longo do Tempo
    st.header("Evolução das Inscrições ao Longo do Tempo")
    registrations_over_time = filtered_df.resample('D', on='Registration date').size().reset_index(name='Total Registrations')
    fig_time = px.line(registrations_over_time, x='Registration date', y='Total Registrations', title="Inscrições Diárias")
    st.plotly_chart(fig_time)

    # Gráfico de Inscrições por Percurso
    st.header("Inscrições por Percurso")
    competition_counts = filtered_df['Competition'].value_counts().reset_index()
    competition_counts.columns = ['Competition', 'Total Athletes']
    fig_comp = px.pie(competition_counts, names='Competition', values='Total Athletes', title="Distribuição de Inscrições por Percurso")
    st.plotly_chart(fig_comp)

    # Tabela de Dados Filtrados
    st.header("Dados Filtrados")
    st.dataframe(filtered_df)
