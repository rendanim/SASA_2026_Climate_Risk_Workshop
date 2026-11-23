import streamlit as st

from aifs_app.ui import bundle, lead, synthetic_banner, text, zone

b = bundle()
st.title("AI for Climate Risk")
st.caption("Statistical foundations for forecasting in data-scarce environments · "
           "Prof Rendani Mbuvha, University of the Witwatersrand")
synthetic_banner()

st.markdown(text("home.intro"))
st.info(text("home.pipeline"), icon=":material/route:")

t = b.totals(lead())
c1, c2, c3, c4 = st.columns(4)
c1.metric("Land grid boxes (0.1°)", f"{b.n_land:,}")
c2.metric("Climate zones, one model each", len(b.zones_with_model(lead())))
c3.metric("Training days (AIFS runs)", len(b.train_dates(lead())),
          help="Lead-day-dependent: forecasts are matched to the IMERG day they are valid for.")
c4.metric(f"Held-out test days, lead day {lead()}", len(b.test_dates(lead())))

st.markdown("#### The workshop in seven labs")
objectives = [
    ("Statistical foundations of probabilistic forecasting",
     [("lab2_idr", "Lab 2 · From one number to a distribution"), ("lab3_scores", "Lab 3 · Scoring probabilistic forecasts")]),
    ("Uncertainty quantification and decisions with limited or noisy data",
     [("data", "Data preliminaries"), ("lab1_data", "Lab 1 · Data for a data-scarce continent"),
      ("lab4_scarcity", "Lab 4 · Uncertainty with limited data")]),
    ("Climate risk indices and early-warning indicators",
     [("lab5_warnings", "Lab 5 · Early warnings and risk indices")]),
    ("Case studies across African National Meteorological Services",
     [("lab6_maps", "Lab 6 · Where does AIFS add value?")]),
    ("Opportunities and open challenges",
     [("lab7_discussion", "Lab 7 · Open challenges")]),
]
cols = st.columns(2)
for i, (obj, pages) in enumerate(objectives):
    with cols[i % 2].container(border=True, height="stretch"):
        st.markdown(f"**{obj}**")
        for f, label in pages:
            st.page_link(f"labs/{f}.py", label=label, icon=":material/arrow_forward:")

st.markdown("#### The setup")
st.markdown(text("home.setup", train=b.periods.get("train", ""), test=b.periods.get("test", ""),
                 clim=b.periods.get("clim_label", ""), n_zone=f"{int(b.meta['zone_sample_locs']):,}",
                 n_pairs=f"{int(b.meta['num_data_pts']):,}", n_test=f"{int(b.meta['test_sample_per_zone']):,}"))
st.caption(f"The sidebar sets the **lead day** (now day {lead()}) and **climate zone** (now {b.zone_label(zone())}) "
           "used on every page. The Glossary explains every term.")
with st.expander("Where the numbers come from"):
    st.markdown(text("home.provenance"))
