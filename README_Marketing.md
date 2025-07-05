# 🏃‍♂️ Dashboard Marketing - Paraty Brazil by UTMB 2025

Dashboard simplificado focado em métricas de Marketing para a equipe da Paraty Brazil by UTMB 2025.

## 📊 Métricas Incluídas

### 1️⃣ Total de Inscritos por Percurso
- Número absoluto de inscritos em cada percurso
- Percentual em relação à meta de cada percurso
- Gráfico de barras com cores baseadas no percentual da meta

### 2️⃣ Percentual de Mulheres
- Cálculo do % de mulheres sobre o total de inscritos
- Gráfico gauge com meta de 50%
- Métricas absolutas e percentuais

### 3️⃣ Percentual de Estrangeiros
- % de atletas estrangeiros no total do evento
- Número de países diferentes representados
- Lista completa de países presentes
- Gráfico gauge com meta de 30%

### 4️⃣ Percentual e Número de Brasileiros
- % e número absoluto de atletas brasileiros
- Gráfico gauge com meta de 70%

### 5️⃣ Municípios Brasileiros
- Quantidade total de municípios representados
- Top 5 municípios com mais inscritos

### 6️⃣ Assessorias Esportivas
- Consolidação de assessorias usando fuzzy matching
- Top 5 assessorias com mais atletas inscritos
- Gráfico de barras das principais assessorias

## 🛠️ Instalação e Uso

### Pré-requisitos
- Python 3.9+
- pip

### Instalação das Dependências
```bash
pip install -r requirements_marketing.txt
```

### Execução
```bash
streamlit run dashboard_marketing.py
```

### Upload de Dados
1. Acesse o dashboard no navegador
2. Faça upload dos arquivos Excel de 2025 (USD e BRL)
3. O dashboard carregará automaticamente os dados dos anos anteriores

## 📋 Funcionalidades

- **Interface Minimalista**: Foco apenas nas métricas essenciais de Marketing
- **Gráficos Interativos**: Gauges e gráficos de barras com Plotly
- **Fuzzy Matching**: Consolidação automática de nomes de assessorias similares
- **Responsivo**: Interface adaptável para diferentes tamanhos de tela
- **Cache Inteligente**: Carregamento otimizado de dados

## 🔧 Tecnologias Utilizadas

- **Streamlit**: Interface web
- **Pandas**: Manipulação de dados
- **Plotly**: Gráficos interativos
- **RapidFuzz**: Fuzzy matching para consolidação de nomes
- **Requests**: Carregamento de dados externos

## 📈 Metas de Referência

- **Mulheres**: 50% do total de inscritos
- **Estrangeiros**: 30% do total de inscritos  
- **Brasileiros**: 70% do total de inscritos
- **Metas por Percurso**:
  - FUN 7KM: 900 inscritos
  - PTR 20: 900 inscritos
  - PTR 35: 620 inscritos
  - PTR 55: 770 inscritos
  - UTSB 100: 310 inscritos

## 🎯 Uso Recomendado

Este dashboard é ideal para:
- Reuniões de Marketing
- Relatórios executivos
- Acompanhamento de metas
- Análise de público-alvo
- Estratégias de divulgação

## 📝 Notas

- Os dados dos anos anteriores (2023-2024) são carregados automaticamente
- O fuzzy matching para assessorias usa um threshold de 80% de similaridade
- Todos os cálculos são feitos em tempo real
- O dashboard é compatível com Python 3.9+ e pandas 1.5+ 