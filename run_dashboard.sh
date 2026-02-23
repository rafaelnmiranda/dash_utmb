#!/usr/bin/env bash
# Evita crash do NumPy/OpenBLAS no macOS ARM (M1): segfault em gemm_thread_n.
# Ver: https://github.com/numpy/numpy/issues/17856
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export OMP_NUM_THREADS=1

cd "$(dirname "$0")"
# Garante que sitecustomize.py seja carregado no início do Python (antes do NumPy)
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
streamlit run streamlit_app.py --server.port 8501 "$@"
