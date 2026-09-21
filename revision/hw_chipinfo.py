# -*- coding: utf-8 -*-
"""hw_chipinfo.py — fetch full WK_C180_2 calibration snapshot and
topology (properly calling pybind methods); save JSON."""
import json
import sys

KEY, OUT = sys.argv[1], sys.argv[2]

from pyqpanda3.qcloud import QCloudService

svc = QCloudService(api_key=KEY)
bk = svc.backend('WK_C180_2')


def call(obj, name):
    try:
        a = getattr(obj, name)
        return a() if callable(a) else a
    except Exception as e:
        return f'<err {e}>'


info = bk.chip_info()
print('chip_id:', call(info, 'chip_id'), 'qubits_num:',
      call(info, 'qubits_num'))

snap = {'selected_archived': [23, 33, 41, 42, 51, 32],
        'single': {}, 'double': [], 'available': None,
        'topology': None, 'high_frequency': None,
        'compensate_angle_keys': None, 'basic_gates': None}

snap['available'] = [int(q) for q in info.available_qubits()]
print('available:', snap['available'])
try:
    snap['high_frequency'] = [int(q) for q in info.high_frequency_qubits()]
except Exception as e:
    print('hf err:', e)

for qi in info.single_qubit_info():
    qid = int(qi.get_qubit_id())
    rec = {}
    for k, fn in (('t1', 'get_t1'), ('t2', 'get_t2'),
                  ('readout_fidelity', 'get_readout_fidelity'),
                  ('frequency', 'get_frequency')):
        try:
            rec[k] = float(call(qi, fn))
        except Exception:
            rec[k] = str(call(qi, fn))
    try:
        rec['single_gate_fidelity'] = float(qi.get_single_gate_fidelity())
    except Exception:
        rec['single_gate_fidelity'] = str(call(qi, 'get_single_gate_fidelity'))
    snap['single'][str(qid)] = rec

for pair in info.double_qubits_info():
    qs = [int(q) for q in pair.get_qubits()]
    fid = float(pair.get_fidelity())
    snap['double'].append({'qubits': qs, 'cz_fidelity': fid})

topo = info.get_chip_topology()
try:
    topo = json.loads(str(topo)) if isinstance(topo, str) else list(topo)
    snap['topology'] = topo
    print('topology edges:', len(topo))
except Exception as e:
    snap['topology'] = str(topo)
    print('topology raw (500):', str(topo)[:500])

try:
    cam = call(info, 'get_compensate_angle_map')
    s = str(cam)
    snap['compensate_angle_keys'] = s[:2000]
    print('compensate map len:', len(s))
except Exception as e:
    print('compensate err:', e)
try:
    snap['basic_gates'] = str(call(info, 'get_basic_gates'))
except Exception as e:
    print('basic gates err:', e)

with open(OUT, 'w', encoding='utf-8') as f:
    json.dump(snap, f, indent=1, ensure_ascii=False)
print('wrote', OUT)

print('\n=== ALL AVAILABLE-QUBIT CALIBRATION ===')
for q in snap['available']:
    r = snap['single'][str(q)]
    print(f"q{q}: T1={r.get('t1')} T2={r.get('t2')} "
          f"RO={r.get('readout_fidelity')} "
          f"1q={r.get('single_gate_fidelity')}")
print('\n=== CZ FIDELITIES (sorted) ===')
for d in sorted(snap['double'], key=lambda d: -d['cz_fidelity'])[:15]:
    print(d['qubits'], d['cz_fidelity'])
