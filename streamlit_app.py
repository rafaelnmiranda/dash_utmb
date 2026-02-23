# Evita crash do OpenBLAS no macOS ARM (M1) — deve vir antes de qualquer import que use NumPy
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_MAX_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from dashboard_2026 import main


if __name__ == "__main__":
    main()
