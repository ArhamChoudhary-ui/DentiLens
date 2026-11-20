"""Install the unmodified OrganUI anatomy asset and its source attribution."""
import argparse
from hashlib import sha256
from pathlib import Path
from urllib.request import urlopen

REVISION = 'f2cfe96f19dba7ec3982f2b06e6a34a1f6670450'
BASE = f'https://raw.githubusercontent.com/organui/teeth-3d/{REVISION}/public/models/'
FILES = {
    'teeth.glb': '89a1074b0806eb41c318d38c78a269134c5070be7b90cb2f93ab2de7d35ec239',
    'manifest.json': '396b609ca1c2794c641a793070cf23b28d2dca83db0ef347ca6e86e866608097',
    'ATTRIBUTION.md': '3386e7a8872d0efea86d736e08dfea8f0d894cdda426592a6f7ec0b7ecac28eb',
}


def install(destination, source=None):
    destination.mkdir(parents=True, exist_ok=True)
    for name, digest in FILES.items():
        target = destination / name
        if target.is_file() and sha256(target.read_bytes()).hexdigest() == digest:
            continue
        if source:
            payload = (source / name).read_bytes()
        else:
            with urlopen(BASE + name, timeout=30) as response:
                payload = response.read(2_000_000)
        if sha256(payload).hexdigest() != digest:
            raise ValueError(f'Checksum mismatch for {name}')
        target.write_bytes(payload)
    return destination


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, help='Checked-out OrganUI public/models directory')
    options = parser.parse_args()
    print(install(root / 'frontend/public/anatomy', options.source))
