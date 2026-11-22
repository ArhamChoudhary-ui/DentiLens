from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))

import pytest
from make_test_video import make_video


@pytest.fixture(scope='session')
def video(tmp_path_factory):
    path=tmp_path_factory.mktemp('media')/'geometric.mp4'
    make_video(path)
    return path
