import streamlit as st, yaml

st.title("⚙️ Live Settings")
st.write("Edit `configs/classes.yaml` to persist changes. Below is read-only preview.")

with open("configs/classes.yaml") as f:
    cfg = yaml.safe_load(f)
st.json(cfg)
