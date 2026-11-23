# AI for Climate Risk: Statistical Foundations for Forecasting in Data-Scarce Environments

**Workshop · Annual Conference of the South African Statistical Association (SASA) 2026**
Prof Rendani Mbuvha, University of the Witwatersrand

[![Open in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rendanim/SASA_2026_Climate_Risk_Workshop/blob/main/aifs_imerg_idr_SASA2026.ipynb)

Climate change is generating new demands on statistical modelling, particularly in regions like Africa, where historical climate records are sparse, irregular, or entirely absent at the spatial and temporal resolution needed for robust risk assessment. This workshop introduces the statistical and machine learning methods being developed to address this challenge, with a focus on probabilistic forecasting, uncertainty quantification and climate risk index construction.

The session covers:

- statistical foundations of probabilistic weather and climate forecasting;
- methods for uncertainty quantification and decision-making with limited or noisy data;
- techniques for constructing climate risk indices and early-warning indicators;
- case studies applying these methods across African National Meteorological Services;
- opportunities and open challenges at the intersection of statistics, machine learning and climate science.

## The case study

One worked problem runs through the whole workshop. **AIFS-Single**, ECMWF's machine-learning weather model, gives one daily rainfall number per grid box. We turn that number into a **calibrated predictive distribution** with **isotonic distributional regression (IDR)**, fitted separately for each Köppen–Geiger climate zone. We score it against **GPM IMERG** satellite rainfall with the CRPS, compare it with a 21-year IMERG climatology, and turn the probabilities into warnings and economic value for users with different costs.

The question throughout: **does AIFS add trustworthy information about tomorrow's rain beyond climatology, once its errors are corrected?** With under a year of forecasts to learn from and a satellite estimate as the "truth", it is a data-scarce problem in miniature.

## Workshop materials

| Material | What it is | Where |
|---|---|---|
| **Notebook** | The full analysis in Python, from loading the data to value scores. Runs in Google Colab. | [`aifs_imerg_idr_SASA2026.ipynb`](aifs_imerg_idr_SASA2026.ipynb) |
| **Interactive app** | A data-preliminaries page and seven hands-on labs, used live in the session. | [`aifs_app/`](aifs_app/) |
| **Companion note** | Methods, results, a lab guide, discussion questions and references. | [link to the note] |
| **Slides** | The session deck. | [link to the slides] |

How the objectives map to the materials:

| Workshop objective | App | Notebook |
|---|---|---|
| Statistical foundations of probabilistic forecasting | Labs 2–3 | Sections 4–6 |
| Uncertainty and decisions with limited or noisy data | Data preliminaries, Labs 1 and 4 | Sections 1–3 and 7 |
| Climate risk indices and early-warning indicators | Lab 5 | Section 10 |
| Case studies across African National Meteorological Services | Lab 6 | Sections 8–9 |
| Opportunities and open challenges | Lab 7 | Section 11 |

## Run the notebook

Click **Open in Google Colab** above, then choose *Runtime › Run all*. The first cell installs the three packages Colab lacks (`zarr`, `gcsfs`, `cartopy`). The data are read anonymously from the public bucket `gs://acmad-tutorial`, so no account is needed. The full African domain takes about 10–15 minutes on the free CPU runtime; set `REGION` in the configuration cell to Southern Africa (or another region) to run faster.

## Run the app

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r aifs_app/requirements.txt
python aifs_app/prepare_data.py    # once; needs internet, about 10–15 minutes, writes data_aifs/
streamlit run aifs_app/app.py      # opens http://localhost:8501
```

Without internet, `python aifs_app/prepare_data.py --synthetic` builds fake data for trying the interface; the app then shows a "synthetic data" banner on every page. `python -m pytest aifs_app/tests -q` runs the app's checks. See [`aifs_app/README.md`](aifs_app/README.md) for details, including hosting the app for participants.

## Data

| Role | Source | Period |
|---|---|---|
| Forecast | AIFS-Single, 00 UTC runs, 24 h rainfall, lead days 2–5 (0.25°, regridded to 0.1°) | training runs Mar 2024 – Jan 2025; test runs Jan – Oct 2026 |
| Observations | GPM IMERG V07, daily, 0.1° | training: Final run; 2026 test: Late run |
| Reference forecast | IMERG climatology: all days within ±15 days of the date in 2000–2020 | the 2026 test days |
| Pooling groups | Köppen–Geiger climate zones (Beck et al. 2023), 0.1° | – |

All files are in the public Google Cloud Storage bucket `gs://acmad-tutorial`.

## Repository contents

```
aifs_imerg_idr_SASA2026.ipynb   workshop notebook (Colab)
aifs_app/                       interactive app: prepare_data.py, app.py, labs/, content/, tests/
data_aifs/                      the app's data bundle, created by aifs_app/prepare_data.py
```

## Key references

- Henzi, A., Ziegel, J. F. and Gneiting, T. (2021). Isotonic distributional regression. *Journal of the Royal Statistical Society Series B*, 83(5), 963–993.
- Gneiting, T. and Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and estimation. *Journal of the American Statistical Association*, 102, 359–378.
- Wilks, D. S. (2001). A skill score based on economic value for probability forecasts. *Meteorological Applications*, 8, 209–219.
- Beck, H. E. et al. (2023). High-resolution (1 km) Köppen–Geiger maps for 1901–2099 based on constrained CMIP6 projections. *Scientific Data*, 10, 724.
