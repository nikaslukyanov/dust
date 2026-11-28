"""Verify prepared token files against the data manifest."""
import argparse
import hashlib
import json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verify_data(data_dir, tokenizer, names=None):
    manifest_path = data_dir / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    files = manifest['files']
    verified = {}
    for name in names if names is not None else files:
        if Path(name).name != name or name not in files:
            raise ValueError(f'Invalid or missing data manifest entry: {name}')
        expected = files[name]['sha256']
        if sha256(data_dir / name) != expected:
            raise ValueError(f'Data checksum mismatch: {name}')
        verified[name] = expected
    if sha256(tokenizer) != manifest['tokenizer_sha256']:
        raise ValueError('Tokenizer checksum mismatch')
    return dict(manifest_sha256=sha256(manifest_path), tokenizer_sha256=manifest['tokenizer_sha256'], files=verified)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--tokenizer', type=Path, default=Path('tokenizer/tokenizer.json'))
    args = parser.parse_args()
    verify_data(args.directory, args.tokenizer)
    print('Verified data and tokenizer hashes')


if __name__ == '__main__':
    main()
