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

## Atualizacao (jun/2026) — Opcao B no piloto WeasyPrint

A **Opcao B** foi implementada como piloto no relatorio **Marketing Diario — Executivo**:

| Componente | Arquivo | Papel |
|---|---|---|
| Geracao HTML + PDF | `pdf_export.py` | Template Jinja2, CSS A4, `WeasyPrint`, graficos Plotly via `kaleido` |
| Builders de dados | `dashboard_2026.py` | `build_demography_bundle`, `build_geography_bundle`, `build_international_bundle`, `build_coupon_bundle`, `build_team_medical_company_bundle`, `build_progress_projection_bundle`, `build_registration_cadence_figures` |
| Download real | `dashboard_2026.py` | `_render_pdf_download` → `st.download_button` com `application/pdf` |
| Infra Streamlit Cloud | `packages.txt` + `requirements.txt` | `weasyprint`, `kaleido==0.2.1`, `pillow<12`, libs Pango/Cairo/GLib via apt |
| Verificacao | `scripts/verify_pdf_deps.py` | Smoke test kaleido + WeasyPrint antes do deploy |

### packages.txt (runtime only, sem `-dev`)

`libcairo2`, `libpango-1.0-0`, `libpangocairo-1.0-0`, `libpangoft2-1.0-0`, `libgdk-pixbuf-2.0-0`, `shared-mime-info`, `fonts-dejavu-core` (sem `libglib2.0-0` — conflita com Trixie e vem transitivamente via gdk-pixbuf)

### Fluxo

1. Usuario clica em **Baixar PDF do Diario (Executivo)**.
2. Backend monta HTML com os mesmos numeros da tela (builders compartilhados).
3. Graficos Plotly viram PNG (`kaleido`) embutidos no HTML.
4. WeasyPrint gera bytes PDF.
5. `st.download_button` entrega arquivo `.pdf` real.

### Expansao planejada

Repetir o padrao `build_<tipo>_html` + `_render_pdf_download` para:

- Marketing Diario — Flash
- Marketing Semanal
- Geral
- Financeiro
- Mercado Pago

Os demais relatorios continuam com **Opcao A** (`window.print`) ate migracao individual.
