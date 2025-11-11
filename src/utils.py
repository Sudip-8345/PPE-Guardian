import os, yaml, time
from datetime import datetime

def load_yaml(fp):
    with open(fp, "r") as f:
        return yaml.safe_load(f)

def ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

class Timer:
    def __enter__(self):
        self.t0 = time.time()
        return self
    def __exit__(self, *args):
        self.elapsed = time.time() - self.t0
