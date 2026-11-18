"""Stage Linux CPython 3.11 CUDA wheels without installing them.

Before submitting setup_baseline_gpu.sbatch, run from the repository root:
    python3 scripts/stage_baseline_wheels.py .venv/wheelhouse-cu128
Downloads resume across retries; existing wheels are checksum-verified.
"""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time
import urllib.parse
import urllib.request

DEST = Path(sys.argv[1]).resolve()
DEST.mkdir(parents=True, exist_ok=True)
TORCH_URL = 'https://download-r2.pytorch.org/whl/cu128/torch-2.10.0%2Bcu128-cp311-cp311-manylinux_2_28_x86_64.whl'
TORCH_SHA = '1d01ffaebf64715c0f507a39463149cb19e596ff702bd4bcf862601f2881dabc'


def fetch(url):
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read()


def checksum(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def download(item):
    final = DEST / item['filename']
    if final.exists() and checksum(final) == item['sha256']:
        print('Verified existing:', final.name, flush=True)
        return
    partial = final.with_suffix('.whl.part')
    started = time.monotonic()
    print('Downloading:', final.name, flush=True)
    subprocess.run([
        'curl', '--fail', '--location', '--silent', '--show-error',
        '--retry', '5', '--retry-delay', '2', '--connect-timeout', '30',
        '--continue-at', '-', '--output', str(partial), item['url'],
    ], check=True)
    if checksum(partial) != item['sha256']:
        raise RuntimeError('Checksum mismatch: ' + str(partial))
    partial.replace(final)
    print('Verified: {} ({:.1f} MB, {:.1f}s)'.format(
        final.name, final.stat().st_size / 1e6, time.monotonic() - started), flush=True)


index = fetch('https://download.pytorch.org/whl/cu128/torch/').decode()
assert TORCH_URL + '#sha256=' + TORCH_SHA in index, 'Official torch hash changed'
metadata = fetch(TORCH_URL + '.metadata').decode()
requirements = re.findall(r'^Requires-Dist: ((?:nvidia-[\w-]+|cuda-bindings|triton))==([^;\s]+)', metadata, re.M)
assert len(requirements) == 17, requirements
items = [{'filename': urllib.parse.unquote(TORCH_URL.rsplit('/', 1)[1]),
          'url': TORCH_URL, 'sha256': TORCH_SHA}]
for package, version in requirements:
    release = json.loads(fetch('https://pypi.org/pypi/{}/{}/json'.format(package, version)))
    matches = [f for f in release['urls'] if f['filename'].endswith('.whl')
               and 'manylinux' in f['filename'] and 'x86_64' in f['filename']
               and ('-cp311-cp311-' in f['filename'] or '-py3-none-' in f['filename'])]
    assert len(matches) == 1, (package, [f['filename'] for f in matches])
    f = matches[0]
    items.append({'filename': f['filename'], 'url': f['url'], 'sha256': f['digests']['sha256']})

(DEST / 'wheels.json').write_text(json.dumps(items, indent=2) + '\n')
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    list(pool.map(download, items))
(DEST / 'wheels.ready').write_text('All {} wheels checksum-verified.\n'.format(len(items)))
print('READY:', DEST, flush=True)
