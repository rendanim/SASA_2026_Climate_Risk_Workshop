"""AI for Climate Risk workshop app: AIFS → IMERG calibration with IDR.  Run from the repository root:

    streamlit run aifs_app/app.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

st.set_page_config(page_title="AI for Climate Risk · IDR lab", page_icon=":material/thunderstorm:", layout="wide")

from aifs_app.ui import sidebar  # noqa: E402

nav = st.navigation({
    "": [st.Page("labs/home.py", title="Start here", icon=":material/home:", default=True)],
    "Foundations": [
        st.Page("labs/data.py", title="Data preliminaries", icon=":material/database:"),
        st.Page("labs/lab1_data.py", title="Lab 1 · Data for a data-scarce continent", icon=":material/satellite_alt:"),
        st.Page("labs/lab2_idr.py", title="Lab 2 · From one number to a distribution", icon=":material/stairs:"),
        st.Page("labs/lab3_scores.py", title="Lab 3 · Scoring probabilistic forecasts", icon=":material/rule:"),
    ],
    "Uncertainty and risk": [
        st.Page("labs/lab4_scarcity.py", title="Lab 4 · Uncertainty with limited data", icon=":material/data_usage:"),
        st.Page("labs/lab5_warnings.py", title="Lab 5 · Early warnings and risk indices", icon=":material/warning:"),
    ],
    "Applications": [
        st.Page("labs/lab6_maps.py", title="Lab 6 · Where does AIFS add value?", icon=":material/public:"),
        st.Page("labs/lab7_discussion.py", title="Lab 7 · Open challenges", icon=":material/forum:"),
        st.Page("labs/glossary.py", title="Glossary", icon=":material/menu_book:"),
    ],
})
sidebar()
nav.run()
