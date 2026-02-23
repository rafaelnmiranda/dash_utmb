#!/usr/bin/env python3
"""
Ponto de entrada recomendado para o dashboard.
Define variáveis de ambiente (evita crash do OpenBLAS no macOS M1) e inicia o Streamlit.
Use: python run_streamlit.py
"""
import os
import sys

# Deve ser a primeira coisa: evita crash do OpenBLAS no macOS ARM (M1)
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_MAX_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

# Só depois importar Streamlit (e o subprocess que rodar o app herdará este ambiente)
from streamlit.web import cli

if __name__ == "__main__":
    # Porta e opções podem ser sobrescritas em .streamlit/config.toml
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "streamlit_app.py")
    sys.argv = ["streamlit", "run", script_path] + sys.argv[1:]
    cli.main()
