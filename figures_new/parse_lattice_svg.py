# Parse Coupling_lattice SVG v2: texts via comments + inner translate;
# edges via line2d paths. Node coords derived from qubit-number labels.
import re, json, sys
from xml.etree import ElementTree as ET

SVG = 'Coupling_lattice_WK_C180_2_final.svg'
tree = ET.parse(SVG)
root = tree.getroot()
S = '{http://www.w3.org/2000/svg}'

texts = []  # (x, y, label)
for g in root.iter(S + 'g'):
    if not (g.get('id') or '').startswith('text_'):
        continue
    comment = g.find('{http://www.w3.org/2000/svg}comment') if False else None
    # ElementTree drops comments unless custom parser; re-read raw for comment
import io
raw = open(SVG, encoding='utf-8').read()

# regex over raw: <g id="text_N">\s*<!-- LABEL -->\s*<g id="svg_N" transform="translate(x y) ...">
pat = re.compile(
    r'<!-- ([^>]+?) -->\s*<g [^>]*?transform="translate\(([\d.\-]+) ([\d.\-]+)\)'
    r'(?: scale\(([\d.\-]+))?', re.S)
texts_raw = []
for m in pat.finditer(raw):
    label = m.group(1).strip().replace('\n', ' ')
    if not label or label.startswith('Layer') or label.startswith('Matplotlib'):
        continue
    texts_raw.append((float(m.group(2)), float(m.group(3)), label))
print('texts:', len(texts_raw), file=sys.stderr)

edges = []
for m in re.finditer(
    r'<g id="line2d_\d+">\s*<path clip-path="url\(#(p[0-9a-f]+)\)" d="'
    r'm([\d.\-]+),([\d.\-]+)l([\d.\-]+),([\d.\-]+)"[^>]*stroke="([^"]+)"', raw):
    clip = m.group(1)
    x1, y1, dx, dy = (float(v) for v in m.groups()[1:5])
    edges.append((clip, x1, y1, x1 + dx, y1 + dy, m.group(6)))
print('edges:', len(edges), file=sys.stderr)

json.dump({'texts': texts_raw, 'edges': edges}, open('lattice_parsed.json', 'w'))
sys.stderr.write('clips: %s\n' % sorted({e[0] for e in edges}))
