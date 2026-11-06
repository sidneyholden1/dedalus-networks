"""Check the bundled saved results against the release's SHA-256 manifest."""
import argparse
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--data', type=Path, default=Path(__file__).resolve().parent/'data')
args = parser.parse_args()
manifest = json.loads(Path(__file__).with_name('data-manifest.json').read_text())
total = 0
for item in manifest['files']:
    path = args.data/item['path']
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    if path.stat().st_size != item['bytes'] or digest.hexdigest() != item['sha256']:
        raise ValueError(f'Saved result differs from the manifest: {item["path"]}')
    total += item['bytes']
print(json.dumps(dict(passed=True, files=len(manifest['files']), bytes=total)))
