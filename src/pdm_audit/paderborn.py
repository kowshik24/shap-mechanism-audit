"""Source-derived bearing metadata and an explicit new-bearing feasibility gate."""
from pathlib import Path
import re
import subprocess
from pdm_audit.industrial import SourceTable, rank_capacity


def paper_page(pdf, page):
    return subprocess.check_output(['pdftotext', '-f', str(page), '-l', str(page),
                                    '-raw', str(pdf), '-'], text=True)


def bearing_registry(pdf):
    """Parse original Tables 4/5/7, preserving secondary and combined components."""
    result = []
    artificial = paper_page(pdf, 5)
    for code, location, level in re.findall(r'(?m)^(K[AI]\d{2})\s+(OR|IR)\s+([123])\b', artificial):
        result.append({'bearing_id': code, 'origin': 'artificial', 'component': location,
                       'diagnosis_class': location, 'extent_level': int(level),
                       'source_table': 4, 'source_pdf_page': 5})
    assert len(result) == 12
    real = paper_page(pdf, 6)
    real = real[real.index('\nKA04 fatigue:'):].split('\nOR: outer ring;', 1)[0]
    blocks = re.split(r'(?m)^(K[ABI]\d{2})(?=\s|$)', real)
    for code, block in zip(blocks[1::2], blocks[2::2]):
        text = ' '.join(block.split())
        if 'IR (+OR)' in text:
            location, group = 'IR (+OR)', 'mixed_or_secondary'
        elif 'OR + IR' in text:
            location, group = 'OR + IR', 'mixed_or_secondary'
        else:
            location = re.search(r'\b(OR|IR)\b', text).group(1)
            group = location
        level = re.search(r'\b[SRM]\s+(?:no repetition|random)\s+([123])\b', text)
        assert level, (code, text)
        result.append({'bearing_id': code, 'origin': 'accelerated_lifetime',
                       'component': location, 'diagnosis_class': group,
                       'extent_level': int(level.group(1)), 'source_table': 5, 'source_pdf_page': 6})
    assert len(result) == 26
    healthy = paper_page(pdf, 8)
    for code in re.findall(r'(?m)^(K\d{3})\s+[^\n]+', healthy):
        result.append({'bearing_id': code, 'origin': 'healthy', 'component': 'none',
                       'diagnosis_class': 'healthy', 'extent_level': None,
                       'source_table': 7, 'source_pdf_page': 8})
    assert len(result) == len({r['bearing_id'] for r in result}) == 32
    return sorted(result, key=lambda r: r['bearing_id'])


def condition_registry(html):
    parser = SourceTable(); parser.feed(html)
    result = []
    for cells in parser.rows:
        row = [' '.join(c['text'].split()) for c in cells]
        if len(row) != 5 or not re.fullmatch(r'N\d{2}_M\d{2}_F\d{2}', row[-1]):
            continue
        code = row[-1]
        item = {'condition': code, 'speed_rpm': int(row[1].replace(',', '')),
                'torque_Nm': float(row[2]), 'radial_force_N': int(row[3].replace(',', ''))}
        assert item['speed_rpm'] == int(code[1:3])*100
        assert item['torque_Nm'] == int(code[5:7])/10
        assert item['radial_force_N'] == int(code[9:11])*100
        result.append(item)
    assert len(result) == len({r['condition'] for r in result}) == 4
    return sorted(result, key=lambda r: r['condition'])


# This allocation is fixed before any Paderborn signal is inspected or model fitted.
FUTURE_SPLIT = {
    'training': ['K001', 'K002', 'KA01', 'KA05', 'KA06', 'KA04', 'KA15',
                 'KI01', 'KI05', 'KI04', 'KI14', 'KI16'],
    'validation': ['K003', 'KA03', 'KA16', 'KI03', 'KI17'],
    'reserved_calibration': ['K004', 'KA07', 'KA22', 'KI07', 'KI18'],
    'test': ['K005', 'K006', 'KA08', 'KA09', 'KA30', 'KI08', 'KI21'],
    'mixed_challenge': ['KB23', 'KB24', 'KB27'],
}


def parse_capture_name(name, bearings, conditions):
    match = re.fullmatch(r'(N\d{2}_M\d{2}_F\d{2})_(K(?:[ABI]\d{2}|\d{3}))_(\d+)\.mat', Path(name).name)
    if not match:
        raise ValueError('Invalid Paderborn capture name')
    condition, bearing, capture = match.groups(); capture = int(capture)
    if condition not in conditions or bearing not in bearings or not 1 <= capture <= 20:
        raise ValueError('Unknown bearing/condition or capture outside 1--20')
    return {'condition': condition, 'bearing_id': bearing, 'capture_index': capture,
            'independent_cluster': bearing}


def capacity(registry):
    damaged = [r for r in registry if r['diagnosis_class'] != 'healthy']
    ir = sum(r['diagnosis_class'] == 'IR' for r in registry)
    outer = sum(r['diagnosis_class'] == 'OR' for r in registry)
    healthy = sum(r['diagnosis_class'] == 'healthy' for r in registry)
    maximum = len(damaged)//2
    pure = min(ir, outer)
    return {'target': 'new-bearing mechanism-versus-other-mechanisms S batch',
            'alpha': .05, 'damaged_bearings': len(damaged),
            'optimistic_all_damaged_batches_before_training_or_test': maximum,
            'optimistic_after_one_test': rank_capacity(maximum-1),
            'pure_IR_OR_batches_before_training_or_test': pure,
            'pure_IR_OR_after_one_test': rank_capacity(pure-1),
            'three_class_diagnosis_triples_before_training_or_test': min(ir, outer, healthy),
            'minimum_calibration_batches': 19,
            'minimum_damaged_bearings_for_calibration_plus_one_test_before_training': 40,
            'finite_five_percent_S_gate_passes': rank_capacity(maximum-1)['finite_threshold_available'],
            'independence_is_verified': False,
            'bound_scope': 'non-overlapping bearing clusters; exchangeability not established'}
