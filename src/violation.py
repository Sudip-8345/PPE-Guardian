from .association import iou_xyxy, assign_hungarian

def build_indices(names, raw_to_canonical):
    # map raw class id -> canonical name
    id2canon = {}
    for i, nm in enumerate(names):
        nm = str(nm)
        key = next((k for k in raw_to_canonical.keys() if k.lower() == nm.lower()), nm)
        canon = raw_to_canonical.get(key, nm).lower()

        id2canon[i] = canon
    return id2canon

def split_by_class(boxes, clss, id2canon):
    by = {}
    for b, c in zip(boxes, clss):
        nm = id2canon.get(int(c), str(c))
        by.setdefault(nm, []).append(b)
    return by

def person_ppe_association(persons, ppe_dict, method_cfg):
    # For each PPE class, create assignments person->item index
    # Returns: {person_idx: set([ppe_names...])}
    have = {i: set() for i in range(len(persons))}
    for ppe_name, ppe_boxes in ppe_dict.items():
        if ppe_name == "person": 
            continue
        match = assign_hungarian(persons, ppe_boxes, cost_fn=method_cfg.get("method","center"))
        # naive acceptance; you can filter by dist/IoU thresholds:
        for pi, pj in match.items():
            have[pi].add(ppe_name)
    return have

def violations_for_frame(persons, have_map, required):
    # return list of (person_idx, missing_list)
    vio = []
    for i in range(len(persons)):
        missing = [k for k, need in required.items() if need and k not in have_map.get(i,set())]
        if missing:
            vio.append((i, missing))
    return vio
