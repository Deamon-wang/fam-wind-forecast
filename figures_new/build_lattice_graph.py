# Build topology graph from lattice_parsed.json; verify edge->node mapping.
import json, math
d = json.load(open('lattice_parsed.json'))
t, e = d['texts'], d['edges']

A_CLIP, B_CLIP = 'p298d0b1085', 'pf339f445eb'

def labels_for(clip_prefix):
    # panel-a numeric labels y-range vs panel-b q-labels are in different
    # x regions: A panel x < 300, B panel x > 380 (from clip rects)
    out = []
    for x, y, lab in t:
        if lab.isdigit() and x < 300:
            out.append((x, y, int(lab)))
    return out

# panel-b q labels
blabels = [(x, y, lab) for x, y, lab in t if lab.startswith('q') and lab[1:].isdigit()]
print('b labels:', len(blabels))
alabels = labels_for(A_CLIP)
print('a labels:', len(alabels))

def nearest(labels, x, y, tol=15.0):
    best, bd = None, 1e9
    for lx, ly, lab in labels:
        dd = math.hypot(lx - x, ly - y)
        if dd < bd:
            bd, best = dd, lab
    return (best, bd) if bd <= tol else (None, bd)

def build(clip, labels, name):
    edges, fails = [], []
    for c, x1, y1, x2, y2, stroke in e:
        if c != clip:
            continue
        n1, d1 = nearest(labels, x1, y1)
        n2, d2 = nearest(labels, x2, y2)
        if n1 is None or n2 is None:
            fails.append(((x1, y1, x2, y2), d1, d2))
        else:
            edges.append((n1, n2, stroke))
    print(name, 'edges mapped:', len(edges), 'failures:', len(fails))
    if fails:
        print(' sample failure:', fails[0])
    return edges

a_edges = build(A_CLIP, alabels, 'panel-a')
b_edges = build(B_CLIP, blabels, 'panel-b')

# color census
from collections import Counter
print('panel-a stroke colors:', Counter(s for *_, s in a_edges))
print('panel-b stroke colors:', Counter(s for *_, s in b_edges))

# adjacency of partition within full lattice
part = {39,40,48,49,50,57,58,59,66,67,68,75,76,77,84,85,86}
pe = [(u,v) for u,v,s in a_edges if u in part and v in part]
print('partition edges inside panel-a:', len(pe))
print('sorted partition edges:', sorted(tuple(sorted(x)) for x in pe))
json.dump({'a_nodes': alabels, 'a_edges': a_edges,
           'b_nodes': blabels, 'b_edges': b_edges},
          open('lattice_graph.json', 'w'))
