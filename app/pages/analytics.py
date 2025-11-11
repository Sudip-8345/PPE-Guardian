import streamlit as st, pandas as pd, os

st.title("📈 Analytics — Violations Log")

logf = "logs/violations.csv"
if os.path.exists(logf):
    df = pd.read_csv(logf)
    st.dataframe(df.tail(500), use_container_width=True)
    st.metric("Total Violations", len(df))
    st.bar_chart(df["missing"].value_counts())
else:
    st.info("No violations logged yet.")
