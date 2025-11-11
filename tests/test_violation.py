from src.violation import violations_for_frame
def test_vio():
    persons=[[0,0,10,10],[20,20,40,40]]
    have={0:{"helmet","vest"},1:set()}
    req={"helmet":True,"vest":True,"gloves":False,"boots":False,"mask":False}
    v = violations_for_frame(persons, have, req)
    assert len(v)==1 and v[0][0]==1
