import streamlit as st

from aifs_app.ui import answer_box, bundle, lead, synthetic_banner, text

b, L = bundle(), lead()
t = b.totals(L)
st.title("Lab 7 · Open challenges")
st.caption("Objective: opportunities and open challenges at the intersection of statistics, ML and climate science")
synthetic_banner()
st.markdown(text("lab7.intro"))

st.markdown("#### The four workshop questions")
st.caption("Answer each one before revealing ours. The answers draw on the labs, with today's numbers where relevant.")
vals = dict(lead=L, mcb=f"{(t['raw'] - t['idr']) * 24:.2f}", dsc=f"{(t['clim'] - t['idr']) * 24:.2f}",
            unc=f"{t['clim'] * 24:.2f}")
links = {"q1": "labs/lab3_scores.py", "q2": "labs/lab3_scores.py", "q3": "labs/lab5_warnings.py",
         "q4": "labs/lab4_scarcity.py"}
for i, q in enumerate(("q1", "q2", "q3", "q4"), start=1):
    with st.container(border=True):
        st.markdown(f"**Question {i}.** " + text(f"lab7.{q}.q"))
        answer_box(f"answer_lab7_{q}", height=80)
        with st.expander("Reveal an answer"):
            st.markdown(text(f"lab7.{q}.a", **vals))
        st.page_link(links[q], label="See it in the lab", icon=":material/arrow_forward:")

st.divider()
st.markdown("#### Open challenges")
st.markdown(text("lab7.challenges"))
st.divider()
st.markdown("#### Your turn")
st.markdown("Which of these challenges matters most for your service or research, and what data would you need to tackle it?")
answer_box("answer_lab7_own", height=120)
