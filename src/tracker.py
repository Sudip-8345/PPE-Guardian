from ultralytics import YOLO

def load_detector(model_path):
    return YOLO(model_path)

def detect_and_track(model, frame, conf=0.5, tracker="bytetrack.yaml"):
    # ultralytics track API in single-image mode via stream=False
    results = model.track(source=frame, conf=conf, persist=True, tracker=tracker, stream=False, verbose=False)
    return results[0]  # first (and only) result
