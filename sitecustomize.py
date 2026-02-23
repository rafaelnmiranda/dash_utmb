# Carregado automaticamente no início do Python (antes do script principal).
# Evita crash do OpenBLAS no macOS ARM (M1) ao limitar threads antes de qualquer import.
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_MAX_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
