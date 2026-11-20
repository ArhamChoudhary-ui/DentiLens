"""Check annotation conversion and train/test isolation on a tiny archive."""
import importlib.util
import json
from pathlib import Path
import zipfile

import pytest

spec = importlib.util.spec_from_file_location('prepare_intraoral',
    Path(__file__).resolve().parents[1] / 'scripts/prepare_intraoral.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def make_archive(path, row='1 0.1 0.2 0.4 0.2 0.4 0.6 0.1 0.6', duplicate=False):
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('data.yaml', 'nc: 2\nnames: [11, 327]\n')
        for split in ('train', 'valid', 'test'):
            archive.writestr(f'{split}/images/{split}.jpg', b'same' if duplicate else split.encode())
            archive.writestr(f'{split}/labels/{split}.txt', row + '\n')


def test_polygon_conversion_retains_splits_and_collapses_classes(tmp_path):
    archive, output = tmp_path / 'source.zip', tmp_path / 'prepared'
    make_archive(archive)
    report = module.prepare(archive, output)
    assert (output / 'train/labels/train.txt').read_text().split() == [
        '0', '0.25000000', '0.40000000', '0.30000000', '0.40000000']
    assert report['splits']['test'] == {'images': 1, 'boxes': 1}
    assert json.loads((output / 'preparation.json').read_text())['output_classes'] == ['tooth']


@pytest.mark.parametrize('row', ['1 nan 0.2 0.3 0.4', '3 0.2 0.2 0.3 0.4',
                                  '0 0.2 0.2 0.0 0.4'])
def test_invalid_annotations_leave_no_dataset(tmp_path, row):
    make_archive(tmp_path / 'source.zip', row)
    with pytest.raises(ValueError):
        module.prepare(tmp_path / 'source.zip', tmp_path / 'prepared')
    assert not (tmp_path / 'prepared').exists()


def test_cross_split_duplicate_is_rejected(tmp_path):
    make_archive(tmp_path / 'source.zip', duplicate=True)
    with pytest.raises(ValueError, match='Identical images'):
        module.prepare(tmp_path / 'source.zip', tmp_path / 'prepared')
