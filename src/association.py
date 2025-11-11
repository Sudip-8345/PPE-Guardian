import numpy as np
from scipy.optimize import linear_sum_assignment

def iou_xyxy(a, b):
    ax1,ay1,ax2,ay2 = a; bx1,by1,bx2,by2 = b
    ix1,iy1 = max(ax1,bx1), max(ay1,by1)
    ix2,iy2 = min(ax2,bx2), min(ay2,by2)
    iw, ih = max(0, ix2-ix1), max(0, iy2-iy1)
    inter = iw*ih
    ua = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter + 1e-9
    return inter / ua

def center_dist(a, b):
    ax = (a[0]+a[2])/2; ay = (a[1]+a[3])/2
    bx = (b[0]+b[2])/2; by = (b[1]+b[3])/2
    return np.hypot(ax-bx, ay-by)

def assign_hungarian(person_boxes, item_boxes, cost_fn="center"):
    if not len(person_boxes) or not len(item_boxes):
        return {}
    C = np.zeros((len(person_boxes), len(item_boxes)), dtype=float)
    for i,p in enumerate(person_boxes):
        for j,q in enumerate(item_boxes):
            if cost_fn=="center":
                C[i,j] = center_dist(p,q)
            else:
                C[i,j] = 1 - iou_xyxy(p,q)  # minimize (1 - IoU)
    r, c = linear_sum_assignment(C)
    return {int(rk): int(ck) for rk,ck in zip(r,c)}
