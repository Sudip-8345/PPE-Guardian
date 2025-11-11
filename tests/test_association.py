from src.association import iou_xyxy, center_dist
def test_iou():
    a=[0,0,10,10]; b=[5,5,15,15]
    assert 0 < iou_xyxy(a,b) < 1
def test_center():
    a=[0,0,10,10]; b=[10,0,20,10]
    assert center_dist(a,b) > 0
