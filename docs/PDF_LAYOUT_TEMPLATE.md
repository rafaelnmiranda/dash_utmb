# Template Visual Alvo para PDF

## Diretrizes de layout

- Formato: **A4 retrato**
- Margens: top 12mm, laterais 10mm, bottom 14mm
- Hierarquia tipografica clara para `titulo > secao > tabela/grafico`
- Quebras de pagina por bloco tematico

## Estrutura recomendada

1. **Cabecalho do relatorio**
   - Nome do dashboard e data base
   - Contexto do recorte (periodo/filtros)
2. **KPIs executivos**
   - Metricas principais no topo
3. **Blocos analiticos**
   - Graficos principais
   - Tabelas de apoio
4. **Exportacoes e anexos**
   - Resumo de dados e links para CSV/JSON/XLSX
5. **Rodape**
   - Timestamp da geracao

## Regras de pagina

- Evitar quebra dentro de metricas, tabelas e graficos.
- Evitar iniciar secao no fim da pagina.
- Expanders devem ser exibidos no modo impressao para nao ocultar conteudo.
- Conteudo de tabs deve ser incluido na exportacao quando possivel.

## Checklist de qualidade do PDF

- Leitura confortavel em zoom 100%
- Sem truncamento de tabelas criticas
- Sem sobreposicao de titulos e graficos
- Cores e contrastes legiveis em impressao e tela
