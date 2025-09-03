# tests/conftest.py
import os, sys
# adiciona a raiz do repo ao sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# opcional: evitar inicialização CUDA etc.
os.environ.setdefault("NUMBA_DISABLE_CUDA", "1")
os.environ.setdefault("MPLBACKEND", "Agg")
