"""Smoke-test every page headlessly:  python -m pytest aifs_app/tests -q"""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

PAGES = {"labs/home.py": ("AI for Climate Risk", 0),
         "labs/data.py": ("Data preliminaries", 4),
         "labs/lab1_data.py": ("Lab 1 · Data for a data-scarce continent", 3),
         "labs/lab2_idr.py": ("Lab 2 · From one number to a distribution", 4),
         "labs/lab3_scores.py": ("Lab 3 · Scoring probabilistic forecasts", 5),
         "labs/lab4_scarcity.py": ("Lab 4 · Uncertainty with limited data", 4),
         "labs/lab5_warnings.py": ("Lab 5 · Early warnings and risk indices", 6),
         "labs/lab6_maps.py": ("Lab 6 · Where does AIFS add value?", 4),
         "labs/lab7_discussion.py": ("Lab 7 · Open challenges", 0),
         "labs/glossary.py": ("Glossary", 0)}


def _run(page, setup=None):
    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=300)
    at.run()
    if page != "labs/home.py":
        at.switch_page(page)
        at.run()
    if setup:
        setup(at)
        at.run()
    assert not at.exception, [e.value for e in at.exception]
    title, n_charts = PAGES[page]
    assert [t.value for t in at.title] == [title], "page did not render"
    assert len(at.get("plotly_chart")) >= n_charts
    text = " ".join(m.value for m in at.markdown)
    assert "missing text" not in text
    return at


@pytest.mark.parametrize("page", PAGES)
def test_page_runs(page):
    _run(page)


@pytest.mark.parametrize("lead", [3, 5])
def test_other_lead_and_zone(lead):
    def setup(at):
        at.selectbox(key="lead").set_value(lead)
        at.run()
        zones = at.selectbox(key="zone").options
        at.selectbox(key="zone").set_value(at.selectbox(key="zone").options[-1] if zones else None)
    for page in ("labs/lab2_idr.py", "labs/lab3_scores.py", "labs/lab5_warnings.py"):
        _run(page, setup)


@pytest.mark.parametrize("mode", ["time", "space", "pairs"])
def test_lab4_modes(mode):
    _run("labs/lab4_scarcity.py", lambda at: at.radio(key="lab4_mode").set_value(mode))


def test_lab5_controls():
    def setup(at):
        at.select_slider(key="lab5_T").set_value(50)
        at.radio(key="lab5_scope").set_value("This zone")
        at.selectbox(key="lab5_s1").set_value(50)       # not increasing -> repaired
        at.radio(key="lab5_view").set_value("Raw AIFS vs calibrated probability")
        at.selectbox(key="lab5_region").set_value("East Africa (ICPAC region)")
    _run("labs/lab5_warnings.py", setup)


def test_data_controls():
    def setup(at):
        at.radio(key="data_scope").set_value(at.radio(key="data_scope").options[1])
        at.selectbox(key="data_case").set_value(1)
        at.selectbox(key="data_region").set_value("Southern Africa (SADC CSC)")
    _run("labs/data.py", setup)


def test_lab2_extremes():
    def setup(at):
        at.select_slider(key="lab2_t").set_value(50)
        at.slider(key="lab2_x").set_value(0.0)
    _run("labs/lab2_idr.py", setup)
