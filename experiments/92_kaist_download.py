"""Stream and authenticate the published KAIST vibration archive; inspect MAT headers."""
import hashlib
import json
from pathlib import Path
import time
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'data/dataset_scout'


def main():
    item = next(x for x in json.loads((DEST/'kaist_load_files.json').read_text())
                if x['filename'] == 'vibration.zip')['content_details']
    path = DEST/'kaist_vibration.zip'
    partial = path.with_suffix('.zip.partial')
    if not path.exists():
        start = partial.stat().st_size if partial.exists() else 0
        request = urllib.request.Request(item['download_url'], headers={
            'User-Agent': 'Mozilla/5.0', 'Range': f'bytes={start}-'})
        with urllib.request.urlopen(request, timeout=120) as response:
            if start and response.status != 206:
                raise ValueError('Resume requires HTTP 206')
            with partial.open('ab' if start else 'wb') as out:
                count, last = start, time.monotonic()
                while block := response.read(4*1024*1024):
                    count += len(block)
                    if count > item['size']:
                        raise ValueError('Download exceeds published size')
                    out.write(block)
                    if time.monotonic()-last > 20:
                        print(f'Downloaded {count:,}/{item["size"]:,} bytes', flush=True)
                        last = time.monotonic()
        assert partial.stat().st_size == item['size']
        partial.rename(path)
    digest = hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()
    assert path.stat().st_size == item['size']
    assert digest == item['sha256_hash'], (digest, item['sha256_hash'])
    with zipfile.ZipFile(path) as z:
        entries = [dict(filename=i.filename, bytes=i.file_size, crc=i.CRC,
                        compression=i.compress_type) for i in z.infolist()]
    record = dict(archive=str(path.relative_to(ROOT)), bytes=path.stat().st_size,
                  sha256=digest, expected_sha256=item['sha256_hash'],
                  source=item['download_url'], entries=entries,
                  scope='Full archive authenticated against publisher SHA-256; no outcomes fitted')
    (ROOT/'experiments/results/kaist_archive.json').write_text(json.dumps(record, indent=2)+'\n')
    print('PASS: publisher SHA-256 verified; archive entries:', len(entries), flush=True)


if __name__ == '__main__':
    main()
