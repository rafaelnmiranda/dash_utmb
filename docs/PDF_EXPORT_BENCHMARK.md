# Benchmark de Exportacao PDF - Dashboard Streamlit

## Objetivo

Comparar as opcoes `A` (browser print + CSS) e `B` (HTML executivo + conversao server-side) em cenarios reais do dashboard.

## Cenarios usados

1. **Curto:** relatorio `Financeiro` com periodo pequeno e poucas tabelas.
2. **Medio:** relatorio `Geral` com graficos e tabelas intermediarias.
3. **Longo:** relatorio `Marketing` com varias secoes e tabelas extensas.

## Criterios

- Fidelidade visual (layout final bonito e consistente)
- Paginacao (quebras corretas, sem cortes ruins)
- Esforco de implementacao
- Custo de manutencao
- Dependencia de infraestrutura extra

## Resultado resumido

| Criterio | Opcao A: Print + CSS | Opcao B: HTML + conversor |
|---|---|---|
| Fidelidade visual entre maquinas | Media | Alta |
| Paginacao previsivel | Media | Alta |
| Tempo de entrega | Alto (rapido) | Medio |
| Manutencao | Media | Media/Alta |
| Infra extra | Nao | Sim (pipeline de PDF) |

## Analise por cenario

- **Curto:** A atende bem, com baixo esforco.
- **Medio:** A melhora bastante com CSS de impressao dedicado e controle de quebras.
- **Longo:** B tende a ter resultado final mais consistente, mas com custo maior de implementacao.

## Decisao aplicada agora

Implementar **Opcao A otimizada** no app atual para ganho imediato:

- modo de impressao A4 no sidebar;
- CSS de impressao com paginacao melhor;
- impressao de tabs e expanders com foco em completude do PDF.

Mantemos **Opcao B** como proximo passo se o padrao executivo exigir consistencia total entre ambientes.
