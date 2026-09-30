"""Project runner and Streamlit dashboard launcher for IndoToxic 2024."""

import os
import sys
from pathlib import Path

# Set root directory
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# If executed via 'streamlit run app.py', run dashboard main()
try:
    import streamlit as st
    # Check if running in streamlit context
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    if get_script_run_ctx() is not None:
        from dashboard.app import main
        main()
        sys.exit(0)
except Exception:
    pass

if __name__ == "__main__":
    print("=" * 60)
    print("IndoToxic 2024 — Sistem Cerdas Moderasi & Analisis Toksisitas")
    print("=" * 60)
    print("Untuk menjalankan Dashboard Interaktif Streamlit, ketik:")
    print("  streamlit run dashboard/app.py")
    print("atau:")
    print("  streamlit run app.py")
    print("=" * 60)
    
    # Auto-run if requested with --run flag
    if "--run" in sys.argv:
        os.system(f"streamlit run {ROOT_DIR / 'dashboard' / 'app.py'}")
