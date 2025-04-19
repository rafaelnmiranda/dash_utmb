import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import requests
from io import BytesIO
import unicodedata
import re
import difflib
import os
from config import CONFIG
from dateutil.relativedelta import relativedelta
from matplotlib_venn import venn3_unweighted
import matplotlib.pyplot as plt

# Configuration
st.set_page_config(layout="wide")
TAXA_CAMBIO = CONFIG["TAXA_CAMBIO"]

# Formatter class
class Formatter:
    @staticmethod
    def currency(val):
        try:
            return f"R$ {float(val):,.0f}".replace(",", ".")
        except Exception:
            return val

    @staticmethod
    def percentage(val):
        try:
            return f"{float(val):.2f}%".replace(".", ",")
        except Exception:
            return val

    @staticmethod
    def integer(val):
        try:
            return str(int(round(float(val))))
        except Exception:
            return val

    @staticmethod
    def integer_thousands(val):
        try:
            return f"{int(round(float(val))):,}".replace(",", ".")
        except Exception:
            return val

formatter = Formatter()

# CSS Styling
def apply_print_css(theme="light"):
    css = """
    <style>
    @media print {
        body * { visibility: hidden !important; }
        #print-content, #print-content * { visibility: visible !important; }
        #print-content { position: absolute; top: 0; left: 0; width: 100%; }
        .element-container, .stTable, .plotly-graph-div { page-break-inside: avoid !important; }
        .page-break { page-break-after: always; }
    }
    .titulo { font-size: 48px; color: #002D74; font-weight: bold; text-align: center; }
    .subtitulo { font-size: 22px; color: #00C4B3; font-weight: bold; }
    """
    if theme == "dark":
        css += """
        body { background-color: #1E1E1E; color: #FFFFFF; }
        .stTable, .stMetric { background-color: #2C2C2C; }
        """
    css += "</style>"
    st.markdown(css, unsafe_allow_html=True)

def page_break():
    st.markdown('<div class="page-break"></div>', unsafe_allow_html=True)

# Theme toggle
def toggle_theme():
    if st.session_state.get("theme", "light") == "light":
        st.session_state.theme = "dark"
    else:
        st.session_state.theme = "light"
    apply_print_css(st.session_state.theme)

# Plotly theme
def apply_plotly_theme(fig):
    fig.update_layout(
        template="plotly_white",
        font=dict(family="Arial", size=12, color="#002D74"),
        title_font=dict(size=20, color="#002D74"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=40, t=60, b=40),
        showlegend=True
    )
    fig.update_traces(marker=dict(color="#00C4B3"))
    return fig

# Text normalization
def norm_text(text):
    t = unicodedata.normalize('NFKD', str(text))
    t = ''.join(c for c in t if not unicodedata.combining(c))
    return re.sub(r'[^a-z0-9\s]', '', t.lower().strip())

# IBGE data loading
@st.cache_data(show_spinner=False)
def load_ibge_municipios():
    try:
        if os.path.exists("municipios_IBGE.parquet"):
            df = pd.read_parquet("municipios_IBGE.parquet")
        else:
            df = pd.read_excel(CONFIG["IBGE_URL"], engine='openpyxl')
            df.to_parquet("municipios_IBGE.parquet")
        df['City_norm'] = df['City'].apply(norm_text)
        return df
    except Exception as e:
        st.error(f"❌ Erro ao carregar IBGE: {e}")
        return None

# City correction
def correct_city(city, ibge_df, cutoff=0.8):
    norm = norm_text(city)
    city_choices = ibge_df['City_norm'].tolist()
    matches = difflib.get_close_matches(norm, city_choices, n=1, cutoff=cutoff)
    if matches:
        return ibge_df.loc[ibge_df['City_norm'] == matches[0], 'City'].iat[0]
    return city

# Data loading
def load_data(year, url=None, file=None, moeda=None, ibge_df=None):
    try:
        if url:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            df = pd.read_excel(BytesIO(response.content), sheet_name=0)
        elif file:
            df = pd.read_excel(file, sheet_name=0)
        else:
            return None

        df['Registration date'] = pd.to_datetime(df['Registration date'], errors='coerce')
        df['Ano'] = year
        df['Moeda'] = moeda
        if 'Registration amount' in df.columns:
            df['Registration amount'] = pd.to_numeric(df['Registration amount'], errors='coerce')
            if moeda == "USD":
                df['Registration amount'] *= TAXA_CAMBIO
        if 'Discounts amount' in df.columns:
            df['Discounts amount'] = pd.to_numeric(df['Discounts amount'], errors='coerce')
            if moeda == "USD":
                df['Discounts amount'] *= TAXA_CAMBIO
        if 'City' in df.columns and ibge_df is not None:
            df['City'] = df['City'].astype(str).apply(lambda x: correct_city(x, ibge_df))
        return df
    except Exception as e:
        st.error(f"Erro ao carregar dados de {year}: {e}")
        return None

# Nationality standardization
def standardize_nationality(value):
    if pd.isnull(value):
        return value
    value = value.strip().upper()
    mapping = {"BRASIL": "BR", "BRAZIL": "BR"}
    return mapping.get(value, value)

# Age calculation
def calculate_age(birthdate, reference_date=pd.Timestamp("2025-08-15")):
    try:
        return relativedelta(reference_date, birthdate).years
    except Exception:
        return None

# Comparison table
def create_comparison_table(df_total, group_col, years=[2023, 2024, 2025], title="Comparativo", ibge_df=None):
    st.subheader(title)
    dfs = {year: df_total.query(f"Ano=={year} and Country.str.lower()=='brazil'", engine="python") for year in years}
    if group_col in ['UF', 'Região'] and ibge_df is not None:
        dfs = {year: df.merge(ibge_df[['City', group_col]], on='City', how='left') for year, df in dfs.items()}
    totals = {year: len(df) for year, df in dfs.items()}
    
    main_df = dfs[years[-1]][group_col].value_counts().reset_index()
    main_df.columns = [group_col, 'Inscritos']
    main_df['%'] = (main_df['Inscritos'] / totals[years[-1]] * 100).round(2).astype(str) + '%'
    
    for year in years[:-1]:
        cnt = dfs[year][group_col].value_counts().reindex(main_df[group_col], fill_value=0)
        main_df[f'Inscritos {year}'] = cnt.values
        main_df[f'% {year}'] = (cnt / totals[year] * 100).round(2).astype(str) + '%'
    
    main_df = main_df.sort_values('Inscritos', ascending=False)
    st.table(main_df)
    return main_df

# Download CSV
def download_csv(df, filename):
    csv = df.to_csv(index=False, sep=';')
    st.download_button(
        label="📥 Baixar como CSV",
        data=csv,
        file_name=filename,
        mime="text/csv"
    )

# Main application
def main():
    apply_print_css(st.session_state.get("theme", "light"))
    
    # Sidebar
    st.sidebar.title("Navegação")
    section = st.sidebar.radio("Seção", ["Resumo Geral", "Cidades Brasileiras", "Análises Financeiras", "Comparativos"])
    st.sidebar.button("🌙 Alternar Tema", on_click=toggle_theme)
    st.sidebar.header("Filtros Gerais")
    
    # Load IBGE data
    with st.spinner("🔄 Carregando municípios do IBGE..."):
        ibge_df = load_ibge_municipios()
        if ibge_df is None:
            st.stop()
        st.success("✅ IBGE carregado")
    
    # Load data
    with st.spinner("🔄 Carregando dados..."):
        df_2023 = load_data(2023, url=CONFIG["URL_2023"], moeda="USD", ibge_df=ibge_df)
        df_2024 = load_data(2024, url=CONFIG["URL_2024"], moeda="USD", ibge_df=ibge_df)
        uploaded_files = st.file_uploader("Carregue os arquivos de 2025 (USD e BRL)", type=["xlsx"], accept_multiple_files=True)
        dfs_2025 = []
        if uploaded_files:
            progress_bar = st.progress(0)
            for i, file in enumerate(uploaded_files):
                moeda = "USD" if "USD" in file.name.upper() else "BRL" if ("R$" in file.name or "BRL" in file.name.upper()) else "Desconhecido"
                df = load_data(2025, file=file, moeda=moeda, ibge_df=ibge_df)
                if df is not None:
                    dfs_2025.append(df)
                progress_bar.progress((i + 1) / len(uploaded_files))
            df_2025 = pd.concat(dfs_2025, ignore_index=True) if dfs_2025 else None
        else:
            df_2025 = None
            st.warning("Aguarde o upload dos arquivos de 2025.")
    
    # Combine data
    dataframes = [df for df in [df_2023, df_2024, df_2025] if df is not None]
    if not dataframes:
        st.error("Nenhum dado foi carregado.")
        st.stop()
    
    df_total = pd.concat(dataframes, ignore_index=True)
    if 'Competition' in df_total.columns:
        df_total = df_total[~df_total['Competition'].str.contains("KIDS", na=False, case=False)]
    
    # Dynamic data base
    data_base = df_2025['Registration date'].max() if df_2025 is not None else pd.Timestamp.today()
    data_base_str = data_base.strftime("%d/%m/%y")
    
    # Filters
    competitions = sorted(df_total['Competition'].dropna().unique()) if 'Competition' in df_total.columns else []
    selected_competitions = st.sidebar.multiselect("Competições", competitions, default=competitions)
    df_filtered = df_total[df_total['Competition'].isin(selected_competitions)] if selected_competitions else df_total
    
    # Main content
    st.markdown('<div id="print-content">', unsafe_allow_html=True)
    st.markdown(f'<p class="titulo">Dashboard de Inscrições - Paraty Brazil by UTMB ({data_base_str})</p>', unsafe_allow_html=True)
    
    if section == "Resumo Geral":
        st.header("Resumo Geral")
        
        # Metas de Inscritos
        st.subheader("Metas de Inscritos por Percurso - 2025")
        df_2025_inscritos = df_filtered[df_filtered['Ano'] == 2025].copy()
        if 'Competition' in df_2025_inscritos.columns:
            df_2025_inscritos['Competition'] = df_2025_inscritos['Competition'].str.upper().str.strip()
            metas = pd.DataFrame({
                "Percurso": list(CONFIG["METAS_2025"].keys()),
                "Meta 2025": list(CONFIG["METAS_2025"].values())
            })
            inscritos_2025 = df_2025_inscritos['Competition'].value_counts().reset_index()
            inscritos_2025.columns = ['Percurso', 'Inscritos']
            inscritos_2025['Percurso'] = inscritos_2025['Percurso'].str.upper().str.strip()
            total_inscritos_2025 = inscritos_2025['Inscritos'].sum()
            total_row = pd.DataFrame([{'Percurso': 'TOTAL', 'Inscritos': total_inscritos_2025}])
            inscritos_2025 = pd.concat([inscritos_2025, total_row], ignore_index=True)
            metas_df = metas.merge(inscritos_2025, on="Percurso", how="left").fillna(0)
            metas_df["Inscritos"] = metas_df["Inscritos"].apply(formatter.integer)
            metas_df["Meta 2025"] = metas_df["Meta 2025"].apply(formatter.integer)
            metas_df["% da Meta"] = ((metas_df["Inscritos"].astype(float) / metas_df["Meta 2025"].astype(float)) * 100).fillna(0).apply(formatter.percentage)
            st.table(metas_df)
            download_csv(metas_df, "metas_2025.csv")
        
        # Gender percentage
        if 'Gender' in df_filtered.columns:
            df_2025_gender = df_filtered[df_filtered['Ano'] == 2025]
            total_reg = df_2025_gender.shape[0]
            num_mulheres = df_2025_gender['Gender'].str.strip().str.upper().isin(['F', 'FEMALE']).sum()
            perc_mulheres = (num_mulheres / total_reg) * 100 if total_reg else 0
            st.metric("% de Mulheres Inscritas (2025)", formatter.percentage(perc_mulheres))
        
        # Nationality metrics
        if 'Nationality' in df_filtered.columns:
            df_2025_nat = df_filtered[df_filtered['Ano'] == 2025].copy()
            df_2025_nat['Nationality_std'] = df_2025_nat['Nationality'].dropna().apply(standardize_nationality)
            num_paises_2025 = df_2025_nat['Nationality_std'].nunique()
            total_2025 = len(df_2025_nat)
            num_estrangeiros = df_2025_nat[df_2025_nat['Nationality_std'] != 'BR'].shape[0]
            perc_estrangeiros = (num_estrangeiros / total_2025) * 100 if total_2025 else 0
            col1, col2 = st.columns(2)
            col1.metric("Número de Países Diferentes (2025)", formatter.integer(num_paises_2025))
            col2.metric("% de Estrangeiros (2025)", formatter.percentage(perc_estrangeiros))
        
        # Prazo and Meta
        start_date = pd.Timestamp(CONFIG["START_DATE"])
        end_date = pd.Timestamp(CONFIG["END_DATE"])
        if data_base > end_date:
            data_base = end_date
        total_period = (end_date - start_date).days
        days_elapsed = (data_base - start_date).days
        prazo_percent = (days_elapsed / total_period) * 100
        meta_total = CONFIG["METAS_2025"]["TOTAL"]
        meta_progress = (total_inscritos_2025 / meta_total) * 100
        col_p, col_m = st.columns(2)
        col_p.metric("Prazo Decorrido (%)", formatter.percentage(prazo_percent))
        col_m.metric("Meta Alcançada (%)", formatter.percentage(meta_progress))
        
        # Age distribution
        with st.expander("Distribuição de Idade dos Atletas (2025)"):
            if 'Birthdate' in df_filtered.columns:
                df_2025_age = df_filtered[df_filtered['Ano'] == 2025].copy()
                df_2025_age['Birthdate'] = pd.to_datetime(df_2025_age['Birthdate'], errors='coerce')
                df_2025_age['Age'] = df_2025_age['Birthdate'].apply(calculate_age)
                df_2025_age.loc[df_2025_age['Age'] < 15, 'Age'] = 40
                mean_age = df_2025_age['Age'].mean()
                st.metric("Idade Média dos Atletas (2025)", formatter.integer(mean_age))
                fig_age = px.histogram(
                    df_2025_age,
                    x='Age',
                    nbins=30,
                    title="Distribuição de Idade dos Atletas (2025)",
                    labels={"Age": "Idade", "count": "Número de Atletas"}
                )
                fig_age = apply_plotly_theme(fig_age)
                st.plotly_chart(fig_age, use_container_width=True)
            else:
                st.info("Coluna 'Birthdate' não encontrada.")
    
    elif section == "Cidades Brasileiras":
        st.header("Cidades Brasileiras - Apenas Atletas Brasileiros")
        df_br = df_filtered[(df_filtered['Country'].str.lower() == 'brazil') & (df_filtered['Ano'] == 2025)].copy()
        num_cidades = df_br['City'].nunique()
        total_atletas = len(df_br)
        col1, col2 = st.columns(2)
        col1.metric("Total de Municípios Distintos (2025)", num_cidades)
        col2.metric("Total de Atletas (2025)", total_atletas)
        
        top_cidades = df_br['City'].value_counts().head(10).rename_axis('Cidade').reset_index(name='Inscritos')
        top_cidades['% do Total'] = (top_cidades['Inscritos'] / total_atletas * 100).round(2).astype(str) + '%'
        st.table(top_cidades)
        download_csv(top_cidades, "top_cidades_2025.csv")
        page_break()
        
        st.subheader("Inscritos por Estado (2025)")
        df_uf = df_br.merge(ibge_df[['City', 'UF']], on='City', how='left')
        all_ufs = sorted(ibge_df['UF'].dropna().unique())
        uf_counts = df_uf['UF'].value_counts().reindex(all_ufs, fill_value=0)
        uf_df = uf_counts.reset_index()
        uf_df.columns = ['UF', 'Inscritos']
        uf_df['% do Total'] = (uf_df['Inscritos'] / total_atletas * 100).round(2).astype(str) + '%'
        uf_df = uf_df.sort_values('Inscritos', ascending=False).reset_index(drop=True)
        uf_df.index = uf_df.index + 1
        uf_df.index.name = 'Posição'
        st.table(uf_df)
        download_csv(uf_df, "inscritos_estado_2025.csv")
        
        st.subheader("Inscritos por Região (2025)")
        df_reg = df_br.merge(ibge_df[['City', 'Região']], on='City', how='left')
        all_regs = sorted(ibge_df['Região'].dropna().unique())
        reg_counts = df_reg['Região'].value_counts().reindex(all_regs, fill_value=0)
        reg_df = reg_counts.reset_index()
        reg_df.columns = ['Região', 'Inscritos']
        reg_df['% do Total'] = (reg_df['Inscritos'] / total_atletas * 100).round(2).astype(str) + '%'
        reg_df = reg_df.sort_values('Inscritos', ascending=False)
        st.table(reg_df)
        download_csv(reg_df, "inscritos_regiao_2025.csv")
        page_break()
    
    elif section == "Análises Financeiras":
        st.header("Análises Financeiras")
        st.sidebar.header("Filtro para Métricas Financeiras")
        anos_fin = sorted(df_filtered['Ano'].dropna().unique())
        opcao_ano_fin = st.sidebar.multiselect("Selecione o(s) ano(s):", anos_fin, default=anos_fin)
        df_financial = df_filtered[df_filtered['Ano'].isin(opcao_ano_fin)].copy()
        
        if 'Registration amount' in df_financial.columns and 'Discounts amount' in df_financial.columns:
            receita_bruta = df_financial['Registration amount'].sum() + df_financial['Discounts amount'].sum()
            receita_liquida = df_financial['Registration amount'].sum()
            total_inscritos_fin = df_financial['Email'].nunique()
            col1, col2, col3 = st.columns(3)
            col1.metric("Receita Bruta (R$)", formatter.currency(receita_bruta))
            col2.metric("Receita Líquida (R$)", formatter.currency(receita_liquida))
            col3.metric("Total Inscritos Únicos", formatter.integer(total_inscritos_fin))
        
        tab1, tab2, tab3, tab4 = st.tabs(["Inscritos", "Análises Financeiras", "Comparativo Entre Anos", "Dias da Semana"])
        
        with tab1:
            st.subheader("Dados de Inscritos por Competição")
            if 'Competition' in df_financial.columns:
                inscritos = df_financial['Competition'].value_counts().reset_index()
                inscritos.columns = ['Percurso', 'Inscritos']
                total_insc = inscritos['Inscritos'].astype(int).sum()
                total_row = pd.DataFrame([{'Percurso': 'Total', 'Inscritos': formatter.integer(total_insc)}])
                inscritos = pd.concat([inscritos, total_row], ignore_index=True)
                st.table(inscritos)
                download_csv(inscritos, "inscritos_competicao.csv")
        
        with tab2:
            st.subheader("Análises Financeiras (R$)")
            if 'Discount code' in df_financial.columns:
                coupon_groups = df_financial['Discount code'].fillna('OUTROS').str.upper().str.extract(r'(PTY25|TG25|TP25|GP25)', expand=False)
                df_financial['Coupon Group'] = coupon_groups.fillna('OUTROS')
                coupon_summary = df_financial.groupby('Coupon Group').agg(
                    Total_Discounts=('Discounts amount', 'sum'),
                    Quantidade_Descontos=('Discounts amount', lambda x: (x > 0).sum())
                ).reset_index()
                coupon_summary['Total_Discounts'] = coupon_summary['Total_Discounts'].apply(formatter.integer_thousands)
                coupon_summary['Quantidade_Descontos'] = coupon_summary['Quantidade_Descontos'].apply(formatter.integer)
                total_valores = df_financial['Discounts amount'].sum()
                total_qtd = (df_financial['Discounts amount'] > 0).sum()
                total_row = pd.DataFrame([{
                    'Coupon Group': 'Total',
                    'Total_Discounts': formatter.integer_thousands(total_valores),
                    'Quantidade_Descontos': formatter.integer(total_qtd)
                }])
                coupon_summary = pd.concat([coupon_summary, total_row], ignore_index=True)
                st.table(coupon_summary)
                download_csv(coupon_summary, "descontos.csv")
            
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
            revenue_by_competition['Total Inscritos'] = revenue_by_competition['Total Inscritos'].apply(formatter.integer)
            for col in ['Receita_Bruta', 'Receita_Líquida', 'Total_Descontos']:
                revenue_by_competition[col] = revenue_by_competition[col].apply(formatter.integer_thousands)
            st.table(revenue_by_competition)
            download_csv(revenue_by_competition, "receitas_competicao.csv")
        
        with tab3:
            st.subheader("Participação por Competição (Inscrições Vendidas)")
            comp_sum = df_financial.groupby('Competition')['Registration amount'].sum().reset_index()
            fig_pie = px.pie(
                comp_sum,
                names='Competition',
                values='Registration amount',
                title="Inscrições Vendidas por Competição (R$)"
            )
            fig_pie.update_traces(
                textposition='inside',
                textinfo='label+percent+value',
                texttemplate='%{label}<br>%{percent} – R$%{value:,.0f}',
                hovertemplate='%{label}: R$%{value:,.0f} (%{percent})'
            )
            fig_pie = apply_plotly_theme(fig_pie)
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with tab4:
            st.subheader("Média de Inscrições por Dia da Semana (desde 01/11/2024)")
            df_total['Date'] = pd.to_datetime(df_total['Registration date'].dt.date)
            df_period = df_total[df_total['Date'] >= pd.to_datetime('2024-11-01')].copy()
            daily_counts = df_period.groupby('Date').size().reset_index(name='Inscritos')
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
            fig_weekday = apply_plotly_theme(fig_weekday)
            st.plotly_chart(fig_weekday, use_container_width=True)
    
    elif section == "Comparativos":
        st.header("Comparativos")
        
        # City, State, Region comparisons
        create_comparison_table(df_filtered, 'City', title="Cidades – 2023 vs 2024 vs 2025", ibge_df=ibge_df)
        create_comparison_table(df_filtered, 'UF', title="Estados – 2023 vs 2024 vs 2025", ibge_df=ibge_df)
        create_comparison_table(df_filtered, 'Região', title="Regiões – 2023 vs 2024 vs 2025", ibge_df=ibge_df)
        
        # Participation by year
        st.subheader("Participação dos Atletas por Ano (baseado em Email)")
        set_2023 = set(df_filtered[df_filtered['Ano'] == 2023]['Email'].dropna().unique())
        set_2024 = set(df_filtered[df_filtered['Ano'] == 2024]['Email'].dropna().unique())
        set_2025 = set(df_filtered[df_filtered['Ano'] == 2025]['Email'].dropna().unique())
        only_2023 = set_2023 - set_2024 - set_2025
        only_2024 = set_2024 - set_2023 - set_2025
        only_2025 = set_2025 - set_2023 - set_2024
        only_2023_2024 = (set_2023 & set_2024) - set_2025
        only_2023_2025 = (set_2023 & set_2025) - set_2024
        only_2024_2025 = (set_2024 & set_2025) - set_2023
        all_three = set_2023 & set_2024 & set_2025
        participation = pd.DataFrame({
            'Participação': ['Apenas 2023', 'Apenas 2024', 'Apenas 2025', 'Apenas 2023 e 2024', 'Apenas 2023 e 2025', 'Apenas 2024 e 2025', '2023, 2024 e 2025'],
            'Quantidade': [len(only_2023), len(only_2024), len(only_2025), len(only_2023_2024), len(only_2023_2025), len(only_2024_2025), len(all_three)]
        })
        participation['Quantidade'] = participation['Quantidade'].apply(formatter.integer)
        st.table(participation)
        download_csv(participation, "participacao_ano.csv")
        
        # Venn diagram
        plt.figure(figsize=(8, 8))
        venn = venn3_unweighted(
            [set_2023, set_2024, set_2025],
            set_labels=(f"2023 ({len(set_2023)})", f"2024 ({len(set_2024)})", f"2025 ({len(set_2025)})"),
            set_colors=("#002D74", "#00C4B3", "#FF5733")
        )
        plt.title(f"Participação dos Atletas por Ano (TOTAL: {len(set_2023.union(set_2024).union(set_2025))})")
        buf = BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        plt.close()
        st.image(buf)
        
        # Top 10 foreign countries
        if 'Nationality' in df_filtered.columns:
            st.subheader("Top 10 Países Estrangeiros Inscritos em 2025")
            df_2025 = df_filtered[df_filtered['Ano'] == 2025].copy()
            df_2025['Nat_std'] = df_2025['Nationality'].dropna().apply(standardize_nationality)
            vc = df_2025[df_2025['Nat_std'] != 'BR']['Nat_std'].value_counts()
            total_estrangeiros = vc.sum()
            df_top = vc.head(10).reset_index()
            df_top.columns = ['País', 'Count']
            df_top['Inscritos'] = df_top['Count'].apply(formatter.integer)
            df_top['%'] = (df_top['Count'] / total_estrangeiros * 100).round(0).astype(int).astype(str) + '%'
            total_row = pd.DataFrame([{
                'País': 'Total Estrangeiros',
                'Inscritos': formatter.integer(total_estrangeiros),
                '%': '100%'
            }])
            df_top_countries = pd.concat([df_top[['País', 'Inscritos', '%']], total_row], ignore_index=True)
            st.table(df_top_countries)
            download_csv(df_top_countries, "top_paises_2025.csv")
        
        # Comparative inscriptions
        st.subheader(f"Comparativo de Inscrições até ({data_base.strftime('%d/%m')})")
        cutoff_dates = {
            2023: pd.Timestamp(year=2023, month=data_base.month, day=data_base.day),
            2024: pd.Timestamp(year=2024, month=data_base.month, day=data_base.day),
            2025: data_base
        }
        comp_rows = []
        for ano, cutoff in cutoff_dates.items():
            qtd = df_filtered[(df_filtered['Ano'] == ano) & (df_filtered['Registration date'] <= cutoff)].shape[0]
            comp_rows.append({'Ano': ano, data_base.strftime('%d/%m'): qtd})
        comp_cutoff = pd.DataFrame(comp_rows)
        qtd_2025 = int(comp_cutoff.loc[comp_cutoff['Ano'] == 2025, data_base.strftime('%d/%m')])
        comp_cutoff['Variação (%)'] = comp_cutoff.apply(
            lambda row: None if row['Ano'] == 2025 else (qtd_2025 - row[data_base.strftime('%d/%m')]) / qtd_2025 * 100,
            axis=1
        )
        comp_cutoff_styled = comp_cutoff.style \
            .format({
                data_base.strftime('%d/%m'): formatter.integer_thousands,
                'Variação (%)': lambda x: f"{x:+.2f}%".replace('.', ',') if pd.notnull(x) else ''
            }) \
            .applymap(
                lambda v: "color: green" if isinstance(v, (int, float)) and v > 0 else "color: red" if isinstance(v, (int, float)) and v < 0 else "",
                subset=['Variação (%)']
            )
        st.dataframe(comp_cutoff_styled, use_container_width=True)
        download_csv(comp_cutoff, "comparativo_inscricoes.csv")
        
        # Weekly inscriptions
        st.subheader("Inscrições Vendidas por Semana (Últimas 10 Semanas) - 2025")
        df_2025_acc = df_filtered[df_filtered['Ano'] == 2025].copy()
        df_2025_acc['Date'] = pd.to_datetime(df_2025_acc['Registration date'].dt.date)
        df_2025_acc = df_2025_acc.sort_values('Date')
        last_date = df_2025_acc['Date'].max()
        intervals = [(last_date - pd.Timedelta(days=1) - pd.Timedelta(days=7 * i) - pd.Timedelta(days=6), last_date - pd.Timedelta(days=1) - pd.Timedelta(days=7 * i)) for i in range(10)]
        data = []
        for start, end in intervals:
            cnt = df_2025_acc[(df_2025_acc['Date'] >= start) & (df_2025_acc['Date'] <= end)].shape[0]
            label = f"{start.strftime('%d/%m')} – {end.strftime('%d/%m')}"
            data.append({'Semana': label, 'Inscritos': cnt})
        weekly_counts = pd.DataFrame(data)[::-1].reset_index(drop=True)
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
        fig_weekly = apply_plotly_theme(fig_weekly)
        st.plotly_chart(fig_weekly, use_container_width=True)
        
        # Weekday averages
        st.subheader("Média de Inscrições por Dia da Semana - 2025")
        df_2025_dia = df_filtered[df_filtered['Ano'] == 2025].copy()
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
            title="Média de Inscrições por Dia da Semana - 2025",
            labels={"Media Inscritos": "Média de Inscrições"}
        )
        fig_weekday.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        fig_weekday = apply_plotly_theme(fig_weekday)
        st.plotly_chart(fig_weekday, use_container_width=True)
        
        # Top 15 sales days
        st.subheader("Top 15 Dias de Maior Venda (2025)")
        media_por_dia = weekday_avg.set_index('Weekday')['Media Inscritos']
        daily_counts['Media_Semanadia'] = daily_counts['Weekday'].map(media_por_dia)
        daily_counts['%_Acima_Media'] = ((daily_counts['Inscritos'] - daily_counts['Media_Semanadia']) / daily_counts['Media_Semanadia'] * 100)
        daily_counts['Dia (dd/mm/aa)'] = daily_counts['Date'].dt.strftime('%d/%m/%y') + ' (' + daily_counts['Weekday'].map(weekday_names) + ')'
        daily_counts['% acima da média'] = daily_counts['%_Acima_Media'].round(0).astype(int).astype(str) + '%'
        top15 = daily_counts.sort_values('Inscritos', ascending=False).head(15)[['Dia (dd/mm/aa)', 'Inscritos', '% acima da média']]
        st.table(top15)
        download_csv(top15, "top15_dias_2025.csv")
        
        # Accumulated inscriptions
        st.subheader("Inscrições Acumuladas - 2025")
        df_2025_acc = df_filtered[df_filtered['Ano'] == 2025].copy()
        df_2025_acc['Date'] = pd.to_datetime(df_2025_acc['Registration date'].dt.date)
        df_2025_acc = df_2025_acc.sort_values('Date')
        acc_daily = df_2025_acc.groupby('Date').size().reset_index(name='Inscritos')
        acc_daily['Acumulado'] = acc_daily['Inscritos'].cumsum()
        fig_acc = px.line(acc_daily, x='Date', y='Acumulado', title="Inscrições Acumuladas em 2025")
        fig_acc = apply_plotly_theme(fig_acc)
        st.plotly_chart(fig_acc, use_container_width=True)
        
        # Moving average and projection
        if df_2025 is not None and 'Registration date' in df_2025.columns:
            st.subheader("Média Móvel e Projeção de Inscritos (até 15/08/2025)")
            df_2025_daily = df_2025.copy()
            df_2025_daily['Date'] = pd.to_datetime(df_2025_daily['Registration date'].dt.date)
            daily_counts = df_2025_daily.groupby('Date').size().reset_index(name='Inscritos')
            daily_counts = daily_counts.sort_values('Date')
            daily_counts.set_index('Date', inplace=True)
            daily_counts['MM_7'] = daily_counts['Inscritos'].rolling(window=7).mean()
            daily_counts['MM_15'] = daily_counts['Inscritos'].rolling(window=15).mean()
            daily_counts['MM_30'] = daily_counts['Inscritos'].rolling(window=30).mean()
            last_date = daily_counts.index.max()
            projection_date = pd.Timestamp(year=2025, month=8, day=15)
            delta_days = max((projection_date - last_date).days, 0)
            total_2025_current = df_2025.shape[0]
            mm7 = daily_counts.loc[last_date, 'MM_7'] if pd.notna(daily_counts.loc[last_date, 'MM_7']) else 0
            mm15 = daily_counts.loc[last_date, 'MM_15'] if pd.notna(daily_counts.loc[last_date, 'MM_15']) else 0
            mm30 = daily_counts.loc[last_date, 'MM_30'] if pd.notna(daily_counts.loc[last_date, 'MM_30']) else 0
            proj_7 = total_2025_current + mm7 * delta_days
            proj_15 = total_2025_current + mm15 * delta_days
            proj_30 = total_2025_current + mm30 * delta_days
            projection_df = pd.DataFrame({
                'Média Móvel': ['7 dias', '15 dias', '30 dias'],
                'Valor Médio Diário': [formatter.integer(mm7), formatter.integer(mm15), formatter.integer(mm30)],
                'Projeção Inscritos (15/08/2025)': [formatter.integer(proj_7), formatter.integer(proj_15), formatter.integer(proj_30)]
            })
            st.table(projection_df)
            download_csv(projection_df, "projecao_inscritos_2025.csv")
    
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("***Fim do Dashboard***")

if __name__ == "__main__":
    main()
