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
        return f"R$ {float(val):,.0f}".replace(",", ".")
    except Exception:
        return val

def format_percentage(val):
    """Formata valores percentuais para exibir como '38,94%'."""
    try:
        # Exibe duas casas decimais e troca ponto por vírgula
        return f"{float(val):.2f}%".replace('.', ',')
    except Exception:
        return val

def format_integer(val):
    """Formata valores numéricos para inteiros (sem separador)."""
    try:
        return str(int(round(float(val))))
    except Exception:
        return val

def format_integer_thousands(val):
    """Formata valores inteiros com separador de milhar, ex.: 2444 -> '2.444'."""
    try:
        return f"{int(round(float(val))):,}".replace(",", ".")
    except Exception:
        return val

def standardize_nationality(value):
    """Padroniza os nomes das nacionalidades para contagem única."""
    if pd.isnull(value):
        return value
    value = value.strip().upper()
    mapping = {
        "BRASIL": "BR",
        "BRAZIL": "BR"
        # Adicione outros mapeamentos se necessário.
    }
    return mapping.get(value, value)

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
            # Usa a primeira planilha
            df = pd.read_excel(BytesIO(response.content), sheet_name=0)
            df['Registration date'] = pd.to_datetime(df['Registration date'], errors='coerce')
            df['Ano'] = ano
            # Converter valores monetários de USD para BRL
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

# URLs dos dados estáticos para 2023 e 2024
url_2023 = "https://github.com/rafaelnmiranda/dash_utmb/raw/815dda1e46bf0b731212e12a365ad169dc4d4e23/UTMB%20-%202023%20-%20USD.xlsx"
url_2024 = "https://github.com/rafaelnmiranda/dash_utmb/raw/815dda1e46bf0b731212e12a365ad169dc4d4e23/UTMB%20-%202024%20-%20USD.xlsx"

df_2023 = load_static_data(2023, url_2023)
df_2024 = load_static_data(2024, url_2024)

# Upload dos arquivos de 2025 (USD e BRL)
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
            df = pd.read_excel(file, sheet_name=0)
            df['Registration date'] = pd.to_datetime(df['Registration date'], errors='coerce')
            df['Ano'] = 2025
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

# Unir os dados dos três anos
dataframes = [df for df in [df_2023, df_2024, df_2025] if df is not None]
if dataframes:
    df_total = pd.concat(dataframes, ignore_index=True)
    # Remover registros da prova KIDS
    if 'Competition' in df_total.columns:
        df_total = df_total[~df_total['Competition'].str.contains("KIDS", na=False, case=False)]
    else:
        st.error("A coluna 'Competition' não foi encontrada.")
else:
    st.error("Nenhum dado foi carregado.")
    st.stop()


# -----------------------------------------------------
# 2. Customização Visual (CSS) com Data Base Dinâmica
# -----------------------------------------------------
if df_2025 is not None:
    data_base = df_2025['Registration date'].max()
    data_base_str = data_base.strftime("%d/%m/%y")
else:
    data_base_str = "N/D"

st.markdown(
    f"""
    <style>
    .titulo {{
        font-size: 48px;  /* Tamanho grande para o título */
        color: #002D74;
        font-weight: bold;
        text-align: center;
    }}
    .subtitulo {{
        font-size: 22px;
        color: #00C4B3;
        font-weight: bold;
    }}
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown(f'<p class="titulo">Dashboard de Inscrições - Paraty Brazil by UTMB ({data_base_str})</p>', unsafe_allow_html=True)


# -----------------------------------------------------
# 3. Resumo Geral
# -----------------------------------------------------
st.header("Resumo Geral")

### 3.1 Metas de Inscritos por Percurso - 2025
st.subheader("Metas de Inscritos por Percurso - 2025")
df_2025_inscritos = df_total[df_total['Ano'] == 2025].copy()
if 'Competition' in df_2025_inscritos.columns:
    df_2025_inscritos['Competition'] = df_2025_inscritos['Competition'].str.upper().str.strip()
else:
    st.error("Coluna 'Competition' não encontrada em 2025.")
metas = pd.DataFrame({
    "Percurso": ["FUN 7KM", "PTR 20", "PTR 35", "PTR 55", "UTSB 100", "TOTAL"]
})
metas["Meta 2025"] = [900, 900, 620, 770, 310, 3500]
inscritos_2025 = df_2025_inscritos['Competition'].value_counts().reset_index()
inscritos_2025.columns = ['Percurso', 'Inscritos']
inscritos_2025['Percurso'] = inscritos_2025['Percurso'].str.upper().str.strip()
total_inscritos_2025 = inscritos_2025['Inscritos'].sum()
if 'TOTAL' not in inscritos_2025['Percurso'].values:
    total_row = pd.DataFrame([{'Percurso': 'TOTAL', 'Inscritos': total_inscritos_2025}])
    inscritos_2025 = pd.concat([inscritos_2025, total_row], ignore_index=True)
metas_df = metas.merge(inscritos_2025, on="Percurso", how="left").fillna(0)
metas_df["Meta 2025"] = metas_df["Meta 2025"].apply(format_integer)
metas_df["Inscritos"] = metas_df["Inscritos"].apply(format_integer)
metas_df["% da Meta"] = ((metas_df["Inscritos"].astype(float) / metas_df["Meta 2025"].astype(float)) * 100).fillna(0).apply(format_percentage)
st.table(metas_df)

### 3.2 Percentual de Mulheres Inscritas (coluna Gender) - somente 2025
if 'Gender' in df_total.columns:
    df_2025_gender = df_total[df_total['Ano'] == 2025]
    total_reg = df_2025_gender.shape[0]
    num_mulheres = df_2025_gender['Gender'].str.strip().str.upper().isin(['F', 'FEMALE']).sum()
    perc_mulheres = (num_mulheres / total_reg) * 100
    st.metric("% de Mulheres Inscritas (2025)", format_percentage(perc_mulheres))
else:
    st.info("Coluna 'Gender' não encontrada.")

### 3.3 Número de Países Diferentes (coluna Nationality) - somente 2025
if 'Nationality' in df_total.columns:
    df_2025_nat = df_total[df_total['Ano'] == 2025].copy()
    df_2025_nat['Nationality_std'] = df_2025_nat['Nationality'].dropna().apply(standardize_nationality)
    num_paises_2025 = df_2025_nat['Nationality_std'].nunique()
    st.metric("Número de Países Diferentes (2025)", format_integer(num_paises_2025))
else:
    st.info("Coluna 'Nationality' não encontrada.")

### NOVO: Idade Média e Distribuição das Idades dos Atletas (2025)
if 'Birthdate' in df_total.columns:
    df_2025_age = df_total[df_total['Ano'] == 2025].copy()
    # Converte a coluna "Birthdate" para datetime
    df_2025_age['Birthdate'] = pd.to_datetime(df_2025_age['Birthdate'], errors='coerce')
    df_2025_age['Age'] = 2025 - df_2025_age['Birthdate'].dt.year
    # Se a idade calculada for menor que 15, define como 40
    df_2025_age.loc[df_2025_age['Age'] < 15, 'Age'] = 40

    # Calcula a idade média e exibe em um metric
    mean_age = df_2025_age['Age'].mean()
    st.metric("Idade Média dos Atletas (2025)", format_integer(mean_age))
    
    # Cria o gráfico de distribuição de idade
    import plotly.figure_factory as ff
    hist_data = [df_2025_age['Age'].dropna().tolist()]
    group_labels = ['Idades']
    fig_age = ff.create_distplot(hist_data, group_labels, show_hist=True, show_rug=False)
    fig_age.update_layout(title_text='Distribuição de Idade dos Atletas (2025)')
    st.plotly_chart(fig_age)
else:
    st.info("Coluna 'Birthdate' não encontrada.")


### Novo: Comparação entre Prazo Decorrido e Meta Alcançada
# Definir datas do prazo de vendas:
start_date = pd.Timestamp("2024-10-28")
end_date   = pd.Timestamp("2025-08-15")
# Definir a data base como a última data de inscrição de 2025 (ou hoje, se não houver dados)
if df_2025 is not None:
    data_base = df_2025['Registration date'].max()
else:
    data_base = pd.Timestamp.today()
# Caso a data base ultrapasse o prazo final, considera o prazo final
if data_base > end_date:
    data_base = end_date

total_period = (end_date - start_date).days  # Total de dias do prazo
days_elapsed = (data_base - start_date).days  # Dias decorrido até a data base
prazo_percent = (days_elapsed / total_period) * 100

# Meta: considerando o total de inscritos de 2025 e meta total de 3500
meta_total = 3500
# A variável total_inscritos_2025 já foi calculada na seção de Metas (para 2025)
meta_progress = (total_inscritos_2025 / meta_total) * 100

# Exibe os dois valores lado a lado
col_p, col_m = st.columns(2)
col_p.metric("Prazo Decorrido (%)", format_percentage(prazo_percent))
col_m.metric("Meta Alcançada (%)", format_percentage(meta_progress))

### 3.4 Tabela de Participação dos Atletas (baseado em Email)
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
participation['Quantidade'] = participation['Quantidade'].apply(format_integer)
st.table(participation)


# --- Diagrama de Venn para Participação dos Atletas ---
from matplotlib_venn import venn3
import matplotlib.pyplot as plt

# Supondo que você já tenha os conjuntos:
# set_2023, set_2024 e set_2025 (obtidos a partir dos emails dos atletas para cada ano)
set_2023_count = len(set_2023)
set_2024_count = len(set_2024)
set_2025_count = len(set_2025)

# Cria as labels com as contagens de cada ano
labels = (
    f"2023 ({set_2023_count})",
    f"2024 ({set_2024_count})",
    f"2025 ({set_2025_count})"
)

# Calcula o total de emails únicos (a união dos três conjuntos)
total_unique = len(set_2023.union(set_2024).union(set_2025))

plt.figure(figsize=(6, 6))
venn3([set_2023, set_2024, set_2025], set_labels=labels)
plt.title(f"Participação dos Atletas por Ano (TOTAL: {total_unique})")
st.pyplot(plt)




### 3.5 Top 10 Países Inscritos em 2025
if 'Nationality' in df_total.columns:
    df_2025_only = df_total[df_total['Ano'] == 2025]
    top_paises = df_2025_only['Nationality'].dropna().apply(standardize_nationality).value_counts().head(10).reset_index()
    top_paises.columns = ['País', 'Inscritos']
    top_paises['Inscritos'] = top_paises['Inscritos'].apply(format_integer)
    st.subheader("Top 10 Países Inscritos em 2025")
    total_top = top_paises['Inscritos'].astype(int).sum()
    total_row = pd.DataFrame([{'País': 'Total', 'Inscritos': format_integer(total_top)}])
    top_paises = pd.concat([top_paises, total_row], ignore_index=True)
    st.table(top_paises)
else:
    st.info("Coluna 'Nationality' não encontrada.")

### 3.6 Comparativo de Inscrições até Data Base
# Calcula a data base dinâmica (última inscrição de 2025)
if df_2025 is not None:
    data_base = df_2025['Registration date'].max()
    data_base_str = data_base.strftime("%d/%m")
else:
    data_base = pd.Timestamp("2025-04-12")
    data_base_str = data_base.strftime("%d/%m")

# Subtítulo com dia e mês
st.subheader(f"Comparativo de Inscrições até ({data_base_str})")

# Define os cutoffs para cada ano
cutoff_dates = {
    2023: pd.Timestamp(year=2023, month=data_base.month, day=data_base.day),
    2024: pd.Timestamp(year=2024, month=data_base.month, day=data_base.day),
    2025: data_base
}

# Monta as linhas com Ano e quantidade de inscritos até o cutoff
comp_rows = []
for ano, cutoff in cutoff_dates.items():
    qtd = df_total[(df_total['Ano'] == ano) & (df_total['Registration date'] <= cutoff)].shape[0]
    comp_rows.append({'Ano': ano, data_base_str: qtd})

comp_cutoff = pd.DataFrame(comp_rows)

# Pega a quantidade de 2025 para comparação
qtd_2025 = int(comp_cutoff.loc[comp_cutoff['Ano'] == 2025, data_base_str])

# Calcula a variação invertida em % em relação a 2025:
# (qtd_2025 - qtd_ano) / qtd_2025 * 100
comp_cutoff['Variação (%)'] = comp_cutoff.apply(
    lambda row: None if row['Ano'] == 2025
                else (qtd_2025 - row[data_base_str]) / qtd_2025 * 100,
    axis=1
)

# Estiliza o DataFrame para exibir no Streamlit
comp_cutoff_styled = comp_cutoff.style \
    .format({
        data_base_str: lambda x: format_integer_thousands(x),
        'Variação (%)': lambda x: f"{x:+.2f}%".replace('.', ',')
    }) \
    .applymap(
        lambda v: "color: green" if isinstance(v, (int, float)) and v > 0
                  else "color: red"   if isinstance(v, (int, float)) and v < 0
                  else "",
        subset=['Variação (%)']
    )

st.dataframe(comp_cutoff_styled, use_container_width=True)

### Gráfico de Barras: Inscrições por Semana (Últimas 10 Semanas) - 2025

# --- seu código existente para filtrar 2025 e converter data ---
df_2025_acc = df_total[df_total['Ano'] == 2025].copy()
df_2025_acc['Date'] = pd.to_datetime(df_2025_acc['Registration date'].dt.date)
df_2025_acc = df_2025_acc.sort_values('Date')

# Data base
last_date = df_2025_acc['Date'].max()

# --- cria os 10 intervalos de 7 dias personalizados ---
intervals = []
for i in range(10):
    end = last_date - pd.Timedelta(days=1) - pd.Timedelta(days=7 * i)
    start = end - pd.Timedelta(days=6)
    intervals.append((start, end))

# --- conta inscrições em cada intervalo e formata label ---
data = []
for start, end in intervals:
    cnt = df_2025_acc[(df_2025_acc['Date'] >= start) & (df_2025_acc['Date'] <= end)].shape[0]
    label = f"{start.strftime('%d/%m')} – {end.strftime('%d/%m')}"
    data.append({'Semana': label, 'Inscritos': cnt})

# inverter para que a semana mais antiga venha primeiro no eixo X
weekly_counts = pd.DataFrame(data)[::-1].reset_index(drop=True)

# --- cálculo da média ---
media_inscritos = weekly_counts['Inscritos'].mean()

# --- plot com Plotly ---
fig_weekly = px.bar(
    weekly_counts,
    x='Semana',
    y='Inscritos',
    text='Inscritos',
    title="Inscrições Vendidas por Semana (Últimas 10 Semanas) - 2025",
    labels={"Semana": "Período", "Inscritos": "Quantidade de Inscrições"}
)
fig_weekly.update_traces(textposition='outside')

# linha de média contínua em laranja com legenda
fig_weekly.add_scatter(
    x=weekly_counts['Semana'],
    y=[media_inscritos] * len(weekly_counts),
    mode='lines',
    name=f'Média: {media_inscritos:.1f}',
    line=dict(color='orange')  # cor laranja, estilo contínuo (padrão)
)

# anotação do valor da média em laranja
fig_weekly.add_annotation(
    x=weekly_counts['Semana'].iloc[-1],
    y=media_inscritos,
    text=f"{media_inscritos:.1f}",
    showarrow=False,
    yshift=10,
    font=dict(color='orange')  # rótulo em laranja
)

st.plotly_chart(fig_weekly)

### Gráfico de Barras: Média de Inscrições por Dia da Semana (2025)

# Filtra somente os registros de 2025
df_2025_dia = df_total[df_total['Ano'] == 2025].copy()
# Cria uma coluna 'Date' sem o componente de hora
df_2025_dia['Date'] = pd.to_datetime(df_2025_dia['Registration date'].dt.date)

# Agrupa as inscrições por dia (calculando o número total de inscrições para cada data)
daily_counts = df_2025_dia.groupby('Date').size().reset_index(name='Inscritos')

# Extrai o dia da semana para cada data (0 = Segunda, 6 = Domingo)
daily_counts['Weekday'] = daily_counts['Date'].dt.dayofweek

# Agrupa por dia da semana e calcula a média de inscrições
weekday_avg = daily_counts.groupby('Weekday')['Inscritos'].mean().reset_index(name='Media Inscritos')
# Arredonda a média para 2 casas decimais
weekday_avg['Media Inscritos'] = weekday_avg['Media Inscritos'].round(2)

# Mapeia os números dos dias para nomes (em português), iniciando na Segunda
weekday_names = {0: "Segunda", 1: "Terça", 2: "Quarta", 3: "Quinta", 4: "Sexta", 5: "Sábado", 6: "Domingo"}
weekday_avg['Dia da Semana'] = weekday_avg['Weekday'].map(weekday_names)

# Organiza os dados pelo número do dia da semana para garantir a ordem correta (segunda a domingo)
weekday_avg = weekday_avg.sort_values('Weekday')

# Cria o gráfico de barras com Plotly Express
fig_weekday = px.bar(
    weekday_avg,
    x='Dia da Semana',
    y='Media Inscritos',
    text='Media Inscritos',  # Adiciona os valores de média acima das barras
    title="Média de Inscrições por Dia da Semana - 2025",
    labels={"Media Inscritos": "Média de Inscrições"}
)

# Atualiza os traços para exibir o texto com duas casas decimais e posiciona-o acima das barras
fig_weekday.update_traces(texttemplate='%{text:.2f}', textposition='outside')

st.plotly_chart(fig_weekday)


### 3.7 Gráfico de Inscrições Acumuladas - 2025
st.subheader("Inscrições Acumuladas - 2025")
df_2025_acc = df_total[df_total['Ano'] == 2025].copy()
df_2025_acc['Date'] = pd.to_datetime(df_2025_acc['Registration date'].dt.date)
df_2025_acc = df_2025_acc.sort_values('Date')
acc_daily = df_2025_acc.groupby('Date').size().reset_index(name='Inscritos')
acc_daily['Acumulado'] = acc_daily['Inscritos'].cumsum()
fig_acc = px.line(acc_daily, x='Date', y='Acumulado', title="Inscrições Acumuladas em 2025")
st.plotly_chart(fig_acc)

### 3.8 Média Móvel e Projeção para 15 de Agosto (2025)
if df_2025 is not None and 'Registration date' in df_2025.columns:
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
    delta_days = (projection_date - last_date).days
    if delta_days < 0:
        delta_days = 0
    total_2025_current = df_2025.shape[0]
    mm7   = daily_counts.loc[last_date, 'MM_7'] if pd.notna(daily_counts.loc[last_date, 'MM_7']) else 0
    mm15  = daily_counts.loc[last_date, 'MM_15'] if pd.notna(daily_counts.loc[last_date, 'MM_15']) else 0
    mm30  = daily_counts.loc[last_date, 'MM_30'] if pd.notna(daily_counts.loc[last_date, 'MM_30']) else 0
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
# 4. Filtro de Ano para Métricas Financeiras
# -----------------------------------------------------
st.sidebar.header("Filtro para Métricas Financeiras")
anos_fin = sorted(df_total['Ano'].dropna().unique())
opcao_ano_fin = st.sidebar.multiselect("Selecione o(s) ano(s):", anos_fin, default=anos_fin)
df_financial = df_total[df_total['Ano'].isin(opcao_ano_fin)].copy()

# Cálculo das métricas financeiras:
# Receita Bruta = soma de Registration amount + Discounts amount
# Receita Líquida = soma de Registration amount
# Total Inscritos = atletas únicos (por Email)
if 'Registration amount' in df_financial.columns and 'Discounts amount' in df_financial.columns:
    receita_bruta = df_financial['Registration amount'].sum() + df_financial['Discounts amount'].sum()
    receita_liquida = df_financial['Registration amount'].sum()
    total_inscritos_fin = df_financial['Email'].nunique()
else:
    st.error("As colunas financeiras não foram encontradas.")
    st.stop()

col1, col2, col3 = st.columns(3)
col1.metric("Receita Bruta (R$)", format_currency(receita_bruta))
col2.metric("Receita Líquida (R$)", format_currency(receita_liquida))
col3.metric("Total Inscritos Únicos", format_integer(total_inscritos_fin))

# -----------------------------------------------------
# 5. Layout com Abas para Análises Detalhadas (Métricas Financeiras)
# -----------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["Inscritos", "Análises Financeiras", "Comparativo Entre Anos", "Dias da Semana"])

with tab1:
    st.subheader("Dados de Inscritos por Competição")
    if 'Competition' in df_financial.columns:
        inscritos = df_financial['Competition'].value_counts().reset_index()
        inscritos.columns = ['Percurso', 'Inscritos']
        total_insc = inscritos['Inscritos'].astype(int).sum()
        total_row = pd.DataFrame([{'Percurso': 'Total', 'Inscritos': format_integer(total_insc)}])
        inscritos = pd.concat([inscritos, total_row], ignore_index=True)
        st.table(inscritos)
    else:
        st.error("Coluna 'Competition' ausente.")

with tab2:
    st.subheader("Análises Financeiras (R$)")

    # --- Tabela 1: cupom de desconto com totalizador de valores ---
    if 'Discount code' in df_financial.columns:
        coupon_groups = (
            df_financial['Discount code']
            .fillna('OUTROS')
            .str.upper()
            .str.extract(r'(PTY25|TG25|TP25|GP25)', expand=False)
        )
        df_financial['Coupon Group'] = coupon_groups.fillna('OUTROS')

        coupon_summary = df_financial.groupby('Coupon Group').agg(
            Total_Discounts=('Discounts amount', 'sum'),
            Quantidade_Descontos=('Discounts amount', lambda x: (x > 0).sum())
        ).reset_index()

        coupon_summary['Total_Discounts'] = coupon_summary['Total_Discounts']\
            .apply(lambda x: f"{int(round(x)):,}".replace(',', '.'))
        coupon_summary['Quantidade_Descontos'] = coupon_summary['Quantidade_Descontos']\
            .apply(format_integer)

        total_valores = df_financial['Discounts amount'].sum()
        total_qtd    = (df_financial['Discounts amount'] > 0).sum()
        total_row = pd.DataFrame([{
            'Coupon Group': 'Total',
            'Total_Discounts': f"{int(round(total_valores)):,}".replace(',', '.'),
            'Quantidade_Descontos': format_integer(total_qtd)
        }])

        coupon_summary = pd.concat([coupon_summary, total_row], ignore_index=True)
        st.table(coupon_summary)
    else:
        st.info("Não há informações de cupom de desconto na base.")

    # --- Tabela 2: receitas por competição com totalizador ---
    grp = df_financial.groupby('Competition').agg({
        'Registration amount': 'sum',
        'Discounts amount': 'sum'
    })
    grp['Total Inscritos'] = df_financial.groupby('Competition').size()
    grp['Receita_Bruta']   = grp['Registration amount'] + grp['Discounts amount']
    grp['Receita_Líquida'] = grp['Registration amount']
    grp['Total_Descontos'] = grp['Discounts amount']

    revenue_by_competition = grp.reset_index().drop(
        columns=['Registration amount', 'Discounts amount']
    )

    # Totalizador da segunda tabela
    total_row2 = pd.DataFrame([{
        'Competition': 'Total',
        'Total Inscritos': revenue_by_competition['Total Inscritos'].sum(),
        'Receita_Bruta':   revenue_by_competition['Receita_Bruta'].sum(),
        'Receita_Líquida': revenue_by_competition['Receita_Líquida'].sum(),
        'Total_Descontos': revenue_by_competition['Total_Descontos'].sum()
    }])
    revenue_by_competition = pd.concat(
        [revenue_by_competition, total_row2],
        ignore_index=True
    )

    # Formatação final
    revenue_by_competition['Total Inscritos'] = revenue_by_competition['Total Inscritos']\
        .apply(format_integer)
    for col in ['Receita_Bruta', 'Receita_Líquida', 'Total_Descontos']:
        revenue_by_competition[col] = revenue_by_competition[col]\
            .apply(lambda x: f"{int(round(x)):,}".replace(',', '.'))

    st.table(revenue_by_competition)



with tab3:
    st.subheader("Participação por Competição (Inscrições Vendidas)")

    # Soma total de inscrições vendidas por competição
    comp_sum = (
        df_financial
        .groupby('Competition')['Registration amount']
        .sum()
        .reset_index()
    )

    # Gráfico de pizza
    fig_pie = px.pie(
        comp_sum,
        names='Competition',
        values='Registration amount',
        title="Inscrições Vendidas por Competição (R$)"
    )

    # Texto dentro das fatias: label + percentual + valor formatado
    fig_pie.update_traces(
        textposition='inside',
        textinfo='label+percent+value',
        texttemplate='%{label}<br>%{percent} – R$%{value:,.0f}',
        hovertemplate='%{label}: R$%{value:,.0f} (%{percent})'
    )

    st.plotly_chart(fig_pie)

with tab4:
    st.subheader("Média de Inscrições por Dia da Semana (desde 01/11/2024)")

    # 1) Garante coluna Date sem hora
    df_total['Date'] = pd.to_datetime(df_total['Registration date'].dt.date)

    # 2) Filtra todas as inscrições a partir de 01/11/2024
    df_period = df_total[df_total['Date'] >= pd.to_datetime('2024-11-01')].copy()

    # 3) Conta inscrições por dia
    daily_counts = (
        df_period
        .groupby('Date')
        .size()
        .reset_index(name='Inscritos')
    )

    # 4) Extrai dia da semana (0=Segunda, …,6=Domingo)
    daily_counts['Weekday'] = daily_counts['Date'].dt.dayofweek

    # 5) Calcula média de inscrições por dia da semana
    weekday_avg = (
        daily_counts
        .groupby('Weekday')['Inscritos']
        .mean()
        .reset_index(name='Media Inscritos')
    )
    weekday_avg['Media Inscritos'] = weekday_avg['Media Inscritos'].round(2)

    # 6) Mapeia para nomes em português e ordena
    weekday_names = {
        0: "Segunda", 1: "Terça", 2: "Quarta",
        3: "Quinta", 4: "Sexta", 5: "Sábado", 6: "Domingo"
    }
    weekday_avg['Dia da Semana'] = weekday_avg['Weekday'].map(weekday_names)
    weekday_avg = weekday_avg.sort_values('Weekday')

    # 7) Desenha o gráfico
    fig_weekday = px.bar(
        weekday_avg,
        x='Dia da Semana',
        y='Media Inscritos',
        text='Media Inscritos',
        title="Média de Inscrições por Dia da Semana (nov/24 + 2025)",
        labels={"Media Inscritos": "Média de Inscrições"}
    )
    fig_weekday.update_traces(
        texttemplate='%{text:.2f}',
        textposition='outside'
    )

    st.plotly_chart(fig_weekday)

st.markdown("***Fim do Dashboard***")
