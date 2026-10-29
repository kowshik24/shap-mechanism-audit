"""Download the prespecified CWRU 12 kHz drive-end fault recordings.

Only the official university table supplies file identities and labels. Raw files
are cached, and every recording is checksummed. No normal-file sampling rate is
assumed: this experiment identifies fault location among seeded-fault recordings.
"""
import hashlib
import json
from pathlib import Path
import re
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data' / 'cwru'
URL = 'https://engineering.case.edu/bearingdatacenter/12k-drive-end-bearing-fault-data'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    html = urllib.request.urlopen(URL, timeout=90).read().decode()
    (OUT / 'source_table.html').write_text(html)
    links = dict(re.findall(r'<a[^>]+href="([^"]+\.mat)"[^>]*>(.*?)</a>', html, re.S))
    records = []
    for href, text in links.items():
        label = re.sub('<[^>]+>', '', text).strip()
        match = re.fullmatch(r'(IR|B|OR)(007|014|021)(?:@6)?_([0-3])', label)
        if not match:
            continue
        kind, diameter, load = match.groups()
        # Outer-race position is fixed at 6 o'clock; other positions are excluded.
        if kind == 'OR' and '@6' not in label:
            continue
        url = href if href.startswith('https://') else 'https://engineering.case.edu' + href
        path = OUT / (label + '.mat')
        if not path.exists():
            payload = urllib.request.urlopen(url, timeout=120).read()
            tmp = path.with_suffix('.tmp'); tmp.write_bytes(payload); tmp.replace(path)
        records.append({'recording': label, 'fault': kind, 'diameter_in': int(diameter)/1000,
                        'load_hp': int(load), 'fs_hz': 12000, 'url': url,
                        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                        'bytes': path.stat().st_size})
        print(label, path.stat().st_size, flush=True)
    assert len(records) == 36, f'Expected 36 recordings, found {len(records)}'
    (OUT / 'manifest.json').write_text(json.dumps({'source': URL, 'records': records}, indent=2))


if __name__ == '__main__':
    main()
