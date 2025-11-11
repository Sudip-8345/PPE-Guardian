import cv2

def draw_label(img, text, x, y):
    cv2.putText(img, text, (x, y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (x, y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)

def box_color(cls_name):
    good = {"helmet","vest","gloves","boots","mask"}
    return (0,255,0) if cls_name in good else (255,255,255)

def draw_box(img, box, color, thick=2):
    x1,y1,x2,y2 = map(int, box)
    cv2.rectangle(img, (x1,y1), (x2,y2), color, thick)

def draw_violation(img, box, message="VIOLATION"):
    x1,y1,x2,y2 = map(int, box)
    cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,255), 2)
    draw_label(img, message, x1, y1)
