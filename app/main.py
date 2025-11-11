import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import streamlit as st, cv2, os, pandas as pd
from ultralytics import YOLO
from pathlib import Path
import tempfile

from src.utils import load_yaml, ensure_dir, ts
from src.visual import draw_box, draw_label, draw_violation, box_color
from src.violation import build_indices, split_by_class, person_ppe_association, violations_for_frame
from src.tracker import detect_and_track

st.set_page_config(page_title="PPE Guardian", layout="wide")
st.title("🛡️ PPE Guardian — Safety Violation Detection")

cfg = load_yaml("configs/project.yaml")
cls_cfg = load_yaml("configs/classes.yaml")
required = cls_cfg["required_ppe"]

with st.sidebar:
    st.header("Settings")
    conf = st.slider("Confidence", 0.1, 0.9, float(cfg["conf"]), 0.05)
    req_helmet = st.checkbox("Require Helmet", value=required["helmet"])
    req_vest   = st.checkbox("Require Vest",   value=required["vest"])
    req_gloves = st.checkbox("Require Gloves", value=required["gloves"])
    req_boots  = st.checkbox("Require Boots",  value=required["boots"])
    req_mask   = st.checkbox("Require Mask",   value=required["mask"])
    source = st.radio("Source", ["Upload Video", "Webcam"])
    run_btn = st.button("Run")

required = {
    "helmet": req_helmet,
    "vest": req_vest,
    "gloves": req_gloves,
    "boots": req_boots,
    "mask": req_mask
}

@st.cache_resource
def load_model(path):
    return YOLO(path)

model = load_model(cfg["model_path_pt"])
names = model.model.names
id2canon = build_indices(names, cls_cfg["raw_to_canonical"])

st.write("**Model classes (raw → canonical):**")
st.code("\n".join([f"{i}: {names[i]} -> {id2canon[i]}" for i in range(len(names))]))

stframe = st.empty()

def process_stream(cap):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    ensure_dir(cfg["save_dir"])
    out_path = os.path.join(cfg["save_dir"], "streamlit_output.mp4")
    writer = cv2.VideoWriter(out_path, fourcc, cap.get(cv2.CAP_PROP_FPS) or 25,
                             (int(cap.get(3)), int(cap.get(4))))

    logf = "logs/violations.csv"
    os.makedirs("logs", exist_ok=True)
    cols = ["time","frame","track_id","missing","x1","y1","x2","y2"]
    if not os.path.exists(logf):
        pd.DataFrame(columns=cols).to_csv(logf, index=False)

    fno=0
    while True:
        ok, frame = cap.read()
        if not ok: break
        fno += 1
        res = detect_and_track(model, frame, conf=conf, tracker=cfg["tracker"])
        annotated = res.plot()

        boxes = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else []
        clss  = res.boxes.cls.cpu().numpy().astype(int) if res.boxes is not None else []
        ids   = res.boxes.id.cpu().numpy().astype(int) if (res.boxes is not None and res.boxes.id is not None) else [-1]*len(clss)

        by = split_by_class(boxes, clss, id2canon)
        persons = by.get("person", [])
        have_map = person_ppe_association(persons, by, cls_cfg["association"])
        vio = violations_for_frame(persons, have_map, required)

        for b, c in zip(boxes, clss):
            nm = id2canon.get(int(c), str(c))
            draw_box(annotated, b, box_color(nm))
            draw_label(annotated, nm, int(b[0]), int(b[1]))

        rows=[]
        for pi, missing in vio:
            pb = persons[pi]
            draw_violation(annotated, pb, "Missing: " + ",".join(missing))
            tid = -1
            for b, c, tid_ in zip(boxes, clss, ids):
                if id2canon.get(int(c))=="person":
                    if (abs(b - pb) < 1e-3).all():
                        tid = int(tid_)
                        break
            rows.append([ts(), fno, tid, "|".join(missing), *map(int,pb)])

        if rows:
            pd.DataFrame(rows, columns=cols).to_csv(logf, mode="a", header=False, index=False)

        stframe.image(annotated, channels="BGR", use_column_width=True)
        writer.write(annotated)

    cap.release(); writer.release()
    st.success(f"Saved annotated: {out_path}")

# persistent upload widget
vf = None
if source == "Upload Video":
    vf = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

run_btn = st.button("Run")   # <-- place the Run button AFTER upload widget

if run_btn:
    if source == "Upload Video":
        if vf is None:
            st.error("⚠️ Please upload a video first.")
        else:
            tpath = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tpath.write(vf.read())
            tpath.flush()

            cap = cv2.VideoCapture(tpath.name)
            process_stream(cap)
    
    elif source == "Webcam":
        cap = cv2.VideoCapture(0)
        process_stream(cap)
