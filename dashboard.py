import streamlit as st
import pandas as pd
import plotly.express as px
#imports removidos temporariamente: from rapidfuzz import process, fuzz
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

# Upload da Planilha
uploaded_file = st.file_uploader("Carregue sua planilha de inscrições", type=["xlsx"])

if uploaded_file:
    # Carregar os dados
    df = pd.read_excel(uploaded_file, sheet_name="registrations_sheet_name")
    df['Registration date'] = pd.to_datetime(df['Registration date'], errors='coerce')

    # Ajustando o formato do campo 'Discounts amount'
    if 'Discounts amount' in df.columns:
        df['Discounts amount'] = pd.to_numeric(df['Discounts amount'], errors='coerce').fillna(0)

    # Classificando descontos
    if 'Discount code' in df.columns:
        df['Discount Type'] = df['Discount code'].apply(lambda x: 'Link' if str(x).endswith('25') else 'Real')

        # Tabela de análise de descontos
        discount_summary = df.groupby(['Competition', 'Discount Type']).agg(
            Total_Descontos=pd.NamedAgg(column='Discounts amount', aggfunc='sum'),
            Quantidade_Descontos=pd.NamedAgg(column='Discounts amount', aggfunc=lambda x: (x > 0).sum())
        ).unstack(fill_value=0)

        # Ajustar formatação das colunas
        discount_summary.columns = ['Quantidade_Descontos_Link', 'Quantidade_Descontos_Real', 'Total_Descontos_Link', 'Total_Descontos_Real']
        discount_summary = discount_summary.reset_index()

        # Exibir a tabela
        st.subheader("Análise de Descontos por Tipo e Percurso")
        st.table(discount_summary)

    # ------------------- METAS DE INSCRITOS -------------------
    st.header("Metas de Inscritos por Percurso")
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
    st.header("Percentual de Mulheres (Meta: 40%)")
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
    st.header("Correção Automática de Cidades (Função temporariamente desativada)")
    municipios_ibge = load_ibge_municipios()

    if municipios_ibge is not None:
        # Lista oficial de cidades do IBGE
        official_cities = municipios_ibge['City'].dropna().unique().tolist()
        
        # Mensagem Temporária para Usuário
        st.text("Correção de cidades desativada temporariamente. Utilize outras funcionalidades do Dashboard.")

    # ------------------- NACIONALIDADE DOS ATLETAS -------------------
    st.header("Nacionalidade dos Atletas")
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

    # ------------------- ANÁLISES FINANCEIRAS -------------------
    st.header("Análises Financeiras")

    # Receita Bruta e Receita Líquida
    df['Total Amount'] = df['Discounts amount'] + df['Registration amount']
    total_revenue = df['Total Amount'].sum()
    net_revenue = df['Registration amount'].sum()
    st.metric("Receita Bruta", f"R$ {total_revenue:,.2f}")
    st.metric("Receita Líquida", f"R$ {net_revenue:,.2f}")

    # Análise de Cupons de Desconto
    st.subheader("Cupons de Desconto")
    if 'Discount code' in df.columns:
        coupon_groups = df['Discount code'].fillna('OUTROS').str.upper().str.extract(r'(PTY25|TG25|TP25|GP25)', expand=False).fillna('OUTROS')
        df['Coupon Group'] = coupon_groups[0]

        coupon_summary = df.groupby('Coupon Group').agg(
            Total_Discounts=pd.NamedAgg(column='Discounts amount', aggfunc='sum'),
            Count=pd.NamedAgg(column='Coupon Group', aggfunc='size')
        ).reset_index()

        st.table(coupon_summary)
    else:
        st.warning("A coluna 'Discount code' não foi encontrada na base de dados. Verifique sua planilha.")

    # Receita por Percurso
    st.subheader("Receita por Percurso")
    revenue_by_competition = df.groupby('Competition').agg(
        Total_Inscritos=pd.NamedAgg(column='Competition', aggfunc='size'),
        Receita_Bruta=pd.NamedAgg(column='Total Amount', aggfunc='sum'),
        Receita_Líquida=pd.NamedAgg(column='Registration amount', aggfunc='sum'),
        Total_Descontos=pd.NamedAgg(column='Discounts amount', aggfunc='sum'),
        Quantidade_Descontos=pd.NamedAgg(column='Discounts amount', aggfunc=lambda x: x[x > 0].count()),
        Percentual_Mulheres=pd.NamedAgg(column='T-shirt size (woman)', aggfunc=lambda x: x.notnull().mean() * 100)
    ).reset_index()

    st.table(revenue_by_competition)
