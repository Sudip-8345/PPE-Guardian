import cv2, os, argparse, pandas as pd
from .utils import load_yaml, ensure_dir, ts
from .visual import draw_box, draw_label, draw_violation, box_color
from .tracker import load_detector, detect_and_track
from .violation import build_indices, split_by_class, person_ppe_association, violations_for_frame

def run_video(args):
    cfg = load_yaml(args.project_cfg)
    cls_cfg = load_yaml(args.classes_cfg)

    model = load_detector(cfg["model_path_pt"])
    names = model.model.names  # raw class names
    id2canon = build_indices(names, cls_cfg["raw_to_canonical"])
    required = cls_cfg["required_ppe"]
    assoc_cfg = cls_cfg["association"]

    cap = cv2.VideoCapture(args.input)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path = None
    writer = None
    if cfg.get("save_annotated_video", True):
        ensure_dir(cfg["save_dir"])
        out_path = os.path.join(cfg["save_dir"], f"annot_{os.path.basename(args.input)}")
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
        fno+=1

        res = detect_and_track(model, frame, conf=cfg["conf"], tracker=cfg["tracker"])
        annotated = res.plot()

        boxes = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else []
        clss  = res.boxes.cls.cpu().numpy().astype(int) if res.boxes is not None else []
        ids   = res.boxes.id.cpu().numpy().astype(int) if (res.boxes is not None and res.boxes.id is not None) else [-1]*len(clss)

        by = split_by_class(boxes, clss, id2canon)
        persons = by.get("person", [])
        have_map = person_ppe_association(persons, by, assoc_cfg)
        vio = violations_for_frame(persons, have_map, required)

        # draw overlays
        for b, c in zip(boxes, clss):
            nm = id2canon.get(int(c), str(c))
            draw_box(annotated, b, box_color(nm))
            draw_label(annotated, nm, int(b[0]), int(b[1]))

        # violations in red with label
        log_rows=[]
        for pi, missing in vio:
            pb = persons[pi]
            draw_violation(annotated, pb, "Missing: " + ",".join(missing))
            tid = -1
            # try to fetch track id by matching exact person box (simple approach)
            for b, c, tid_ in zip(boxes, clss, ids):
                if id2canon.get(int(c))=="person":
                    if (abs(b - pb) < 1e-3).all():
                        tid = int(tid_)
                        break
            log_rows.append([ts(), fno, tid, "|".join(missing), *map(int,pb)])

        if log_rows:
            df = pd.DataFrame(log_rows, columns=cols)
            df.to_csv(logf, mode="a", header=False, index=False)

        if writer: writer.write(annotated)
        if args.display:
            cv2.imshow("PPE Guardian", annotated)
            if cv2.waitKey(1) & 0xFF == 27: break

    cap.release()
    if writer: writer.release()
    cv2.destroyAllWindows()
    print("Saved:", out_path if out_path else "no video saved")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="path to video file or 0 for webcam")
    ap.add_argument("--project_cfg", default="configs/project.yaml")
    ap.add_argument("--classes_cfg", default="configs/classes.yaml")
    ap.add_argument("--display", action="store_true")
    args = ap.parse_args()
    run_video(args)
