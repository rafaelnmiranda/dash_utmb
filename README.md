# Dashboard Paraty by UTMB 2026

Dashboard Streamlit para análise de inscrições e métricas da Paraty Brazil by UTMB (edição 2026).

## Rodar localmente

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Ou com o launcher (recomendado no Mac M1):

```bash
python run_streamlit.py
```

## Deploy no Streamlit Community Cloud

1. Envie o projeto para o GitHub (branch `main` ou `master`).
2. Acesse [share.streamlit.io](https://share.streamlit.io), faça login com GitHub.
3. Clique em **"New app"**.
4. Preencha:
   - **Repository**: `seu-usuario/dashboard_utmb` (ou o nome do repositório).
   - **Branch**: `main`.
   - **Main file path**: `streamlit_app.py`.
5. Clique em **"Deploy!"**.

O Streamlit Cloud usa o `requirements.txt` da raiz do repositório. Não é necessário configurar nada além disso para este app.

## Estrutura

- `streamlit_app.py` — entrada do app (ponto de deploy no Streamlit Cloud).
- `dashboard_2026.py` — lógica e UI do dashboard.
- `requirements.txt` — dependências Python.
- `.streamlit/config.toml` — tema e configuração do servidor.

## Documentação

- [Solução de problemas (Mac M1, crash OpenBLAS)](docs/TROUBLESHOOTING.md)
