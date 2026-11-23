# AI for Climate Risk: AIFS → IMERG calibration with IDR

An interactive app for the SASA 2026 workshop **AI for Climate Risk: Statistical Foundations for Forecasting in Data-Scarce Environments** (Prof Rendani Mbuvha, University of the Witwatersrand).
It turns the notebook `aifs_imerg_calibration_tutorial.ipynb` into a data-preliminaries page and seven hands-on labs. In the notebook, ECMWF's AIFS-Single rainfall forecasts are calibrated towards GPM IMERG with isotonic distributional regression (IDR), using one model per Köppen–Geiger climate zone.

| Lab | Workshop objective | What participants do |
|---|---|---|
| Data preliminaries | Limited and noisy data | The four datasets, training and test periods, lead-day matching, what daily rain looks like, one case day side by side, units and caveats |
| 1 · Data for a data-scarce continent | Uncertainty with limited or noisy data | Compare one grid box's data budget with a pooled climate zone; see AIFS drizzle and heavy-rain biases |
| 2 · From one number to a distribution | Statistical foundations | Threshold curves, step-through PAVA, predictive distributions and quantile fans |
| 3 · Scoring probabilistic forecasts | Statistical foundations | CRPS by picture, the notebook's headline scores, the CRPS = MCB − DSC + UNC waterfall |
| 4 · Uncertainty with limited data | Uncertainty quantification | Skill vs weeks / grid boxes / pairs of training data, pooling choices, day-block bootstrap, shuffle control |
| 5 · Early warnings and risk indices | Risk indices and early warning | Reliability, ROC and warning cutoffs, cost–loss value, an impact-based warning index on case-study days |
| 6 · Where does AIFS add value? | Case studies across NMHSs | Skill maps zoomable to ICPAC, ACMAD/AGRHYMET, SADC CSC, ECCAS and North Africa; region and zone tables |
| 7 · Open challenges | Opportunities and open challenges | The notebook's four Mentimeter questions with live answers, and seven open challenges each linked to a lab |

All explanatory text lives in `content/text.md` and can be edited without touching code.

## 1. Build the data bundle (once, needs internet)

The app never reads the bucket directly. Run the preparation script once on a machine that can reach `gs://acmad-tutorial`; Google Colab works well:

```bash
git clone https://github.com/rendanim/SASA_2026_Climate_Risk_Workshop
cd SASA_2026_Climate_Risk_Workshop
pip install -r aifs_app/requirements.txt
python aifs_app/prepare_data.py          # about 10–15 minutes, writes data_aifs/ (about 40–60 MB)
```

It follows the notebook step by step: the same files, region, lead days, settings, random seed and order of random draws. The zone models and headline scores are therefore the notebook's. It saves:

- the notebook's full verification results: scores by lead day, zone and day, reliability counts, and skill maps (coarsened to 0.2°);
- the exact training pairs behind every zone model, so the app can refit and re-score live;
- 20,000 random held-out 2026 test pairs per zone and lead day, with the IMERG climatology's CRPS and P(> 10 mm/day);
- raw AIFS, IMERG and climatological probabilities for four heavy-rain case-study days.

Keep `data_aifs/` next to `aifs_app/`: commit it, or copy it to wherever the app runs. To use a bundle stored elsewhere, set `AIFS_DATA=/path/to/bundle`.

## 2. Run the app

```bash
streamlit run aifs_app/app.py        # from the repository root; opens http://localhost:8501
```

The sidebar sets the **lead day** and **climate zone** used on every page.

**Before the session:** click through every lab once. Lab 4's experiments refit IDR many times, which takes roughly 10–20 seconds on the real data the first time. Results are then cached for every participant on that server.

**Hosting for participants:** the app runs on Streamlit Community Cloud or any server with Python. Point it at `aifs_app/app.py` and `aifs_app/requirements.txt`, and make sure `data_aifs/` is deployed with it.

## 3. Testing without the real data

```bash
python aifs_app/prepare_data.py --synthetic   # small fake bundle in data_aifs_synthetic/, no internet
streamlit run aifs_app/app.py                 # shows a "synthetic data" banner on every page
python -m pytest aifs_app/tests -q            # 24 checks, about 30 s
```

The app uses `data_aifs/` when it exists and falls back to the synthetic bundle otherwise. The tests always use the synthetic bundle (building it if needed) unless `AIFS_DATA` is set. They check:

- the pairwise CRPS against the notebook's lookup-table method and a brute-force integral;
- that the app's refitted zone models are identical to the pipeline's;
- that every page and control runs without errors;
- that the real-data loader runs end to end on fake datasets shaped like the bucket's files.

## How the numbers relate to the notebook

- **Exactly the notebook's:** headline CRPS and CRPSS by lead day, the decomposition waterfall, the daily curves, maps, zone tables and region tables. They are computed on every land grid box and test day when the bundle is built.
- **Live on stored samples:** anything a control changes (thresholds, warning rules, data subsets, pooling choices, bootstraps). When pooled over Africa, each zone's sample is weighted by its number of valid test cells.
- **Reference forecast:** the IMERG climatology (±15 days, 2000–2020). Its probabilities exist only for 10 mm/day, so at other thresholds Lab 5 falls back to each zone's training-period climatology and says so on the page.

## Files

```
aifs_app/
  app.py              navigation
  ui.py               caching, sidebar, text, chart and map helpers
  prepare_data.py     builds the data bundle (real or --synthetic)
  content/text.md     all explanatory text
  core/idr.py         IDR engine ported from the notebook (numba PAVA)
  core/bundle.py      bundle loader
  core/scores.py      CRPS decomposition, Brier, reliability, ROC, economic value, bootstraps
  core/experiments.py learning curves, pooling comparison, shuffle control
  core/pava.py        step-by-step PAVA for display
  labs/               one file per page
  tests/
```

## References to check

The explanatory text cites from memory: Henzi, Ziegel & Gneiting (2021) on IDR; Richardson (2000) on economic value; Arnold et al. and Dimitriadis et al. on CRPS and Brier decompositions; Beck et al. (2023) for the Köppen–Geiger map. Please verify them before the workshop.
