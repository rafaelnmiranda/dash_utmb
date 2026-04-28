# Dashboard UTMB — Claude Context

## Design System
Este projeto segue o design system Apple-inspired definido em `docs/DESIGN-apple.md`.
Aplique as diretrizes desse arquivo em todos os componentes visuais do dashboard.

### Resumo rápido das regras principais
- Fonte: Inter (substituto de SF Pro) — weights 300/400/600/700 apenas
- Cores: `#1d1d1f` texto, `#f5f5f7` parchment, `#0066cc` azul de ação, `#272729` dark tile
- Cards: border-radius 18px, borda `1px solid rgba(0,0,0,0.08)`, sem sombras em containers
- Gráficos Plotly: fundo transparente, sem gridlines desnecessários, paleta unificada
- Sem gradientes decorativos, sem sombras em texto ou botões

## Arquitetura
- Entry point: `streamlit_app.py` → delega para `dashboard_2026.py`
- Arquivo principal: `dashboard_2026.py` (~4200 linhas)
- Relatórios: Geral, Marketing, Marketing Diário, Marketing Semanal, Financeiro, Mercado Pago

## Regras de negócio
- Relatórios de Marketing NÃO exibem valores financeiros
- Câmbio USD→BRL: constante `EXCHANGE_RATE_USD_TO_BRL` em `config.py`
- Separadores decimais: usar `_safe_to_numeric()` para lidar com vírgula e ponto
