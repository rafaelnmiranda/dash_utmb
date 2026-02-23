# Solução de problemas

## Rodar direto no Streamlit (Cursor / VS Code)

Para subir o dashboard **pelo IDE** (Run/Debug), use a configuração de execução do projeto:

1. Abra o **Run and Debug** (ícone de play com inseto na barra lateral, ou `Ctrl+Shift+D` / `Cmd+Shift+D`).
2. No dropdown, escolha **"Streamlit: dashboard (localhost)"** ou **"Streamlit: via run_streamlit.py"**.
3. Clique em **Run** (play verde) ou **Debug**.

A configuração **"Streamlit: dashboard (localhost)"** já define as variáveis de ambiente (OPENBLAS_NUM_THREADS etc.) para reduzir crash no Mac M1. Se ainda der segmentation fault, use a solução com Conda abaixo.

---

## Crash do Python no Mac (Segmentation fault: 11)

**Sintoma:** O app Streamlit fecha com "EXC_BAD_ACCESS (SIGSEGV)" ou "Segmentation fault: 11", e o relatório de crash aponta para `libopenblas64_.0.dylib` / `gemm_thread_n`.

**Causa:** Bug conhecido do OpenBLAS (usado pelo NumPy) no macOS em chips ARM (M1/M2/M3).

### 1. Usar o launcher do projeto (recomendado)

O jeito certo de subir o dashboard no Mac (e em qualquer OS) é pelo launcher, que define as variáveis de ambiente antes do Streamlit e evita crash no M1:

```bash
python run_streamlit.py
```

A porta e outras opções ficam em **`.streamlit/config.toml`** (por exemplo `server.port = 8501`).

Se preferir o script em shell (mesmo efeito):

```bash
./run_dashboard.sh
```

### 2. Se ainda crashar: solução definitiva (trocar o NumPy)

O NumPy instalado via `pip` no Mac ARM às vezes vem com um OpenBLAS que ignora as variáveis de ambiente. A solução estável é usar um ambiente com NumPy do **Conda/Miniconda** (conda-forge costuma usar um build que não crasha).

**Opção A – Miniconda (recomendado)**

1. Instale o [Miniconda para Apple Silicon](https://docs.conda.io/en/latest/miniconda.html) (arm64).
2. No terminal:

```bash
cd /caminho/para/dashboard_utmb
conda create -n dashboard python=3.12 -y
conda activate dashboard
conda install -c conda-forge numpy pandas plotly requests streamlit openpyxl rapidfuzz matplotlib -y
pip install matplotlib-venn
pip install -r requirements.txt
streamlit run streamlit_app.py --server.port 8501
```

3. Sempre que for rodar o dashboard, ative o ambiente e use o launcher:

```bash
conda activate dashboard
python run_streamlit.py
```

**Opção B – Manter só pip (tentar outra versão do NumPy)**

Às vezes uma versão diferente do NumPy evita o crash:

```bash
pip install "numpy>=1.24,<2" --force-reinstall
# ou
pip install "numpy>=2.0" --force-reinstall
```

Depois teste de novo com `./run_dashboard.sh`.
