import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import pytest

from dentilens.video import VideoReader, windows
from dentilens.schema import validate, atomic_json
from dentilens.selection import select
from dentilens.temporal import nms, smooth
from dentilens.pipeline import analyze


def test_decode_and_source_alignment(video):
    reader=VideoReader(video)
    assert reader.info.duration == 120
    assert reader.info.fps == 10
    reader.verify()
    frames,times=reader.sample(12,16,count=16,size=64)
    assert len(frames)==16 and frames[0].shape==(64,96,3)
    assert times[0]==12 and times[-1]==15.9
    assert times == sorted(times)


def test_bad_video(tmp_path):
    broken=tmp_path/'broken.mp4'
    broken.write_text('not a video')
    with pytest.raises(ValueError):
        VideoReader(broken)


def test_intervals_cover_tail():
    values=list(windows(121.3))
    assert values[0]==(0,4)
    assert values[-1][1]==121.3
    assert all(a[1]>=b[0] for a,b in zip(values,values[1:]))
    with pytest.raises(ValueError):
        list(windows(math.nan))


def test_temporal_postprocessing():
    events=[dict(start=0,end=4,verb='cut',noun='onion',confidence=.9),
            dict(start=1,end=5,verb='cut',noun='onion',confidence=.8),
            dict(start=1,end=5,verb='wash',noun='onion',confidence=.7)]
    assert len(nms(events))==2
    np.testing.assert_allclose(smooth([[1,0],[0,1],[1,0]])[1],[2/3,1/3])


@pytest.mark.parametrize('strategy',['diversity','uniform','motion','confidence'])
def test_selector_constraints_under_random_candidates(strategy):
    rng=random.Random(2026)
    for _ in range(40):
        duration=rng.uniform(120,600)
        candidates=[]
        for i in range(100):
            start=rng.uniform(0,duration-4)
            candidates.append(dict(start=start,end=start+4,verb=str(i%4),noun=str(i%3),
                                   confidence=rng.random(),visual=rng.random()))
        highlights=select(candidates,duration,strategy=strategy)
        assert sum(h['end']-h['start'] for h in highlights)<=duration*.15+1e-8
        assert all(a['end']<=b['start'] for a,b in zip(highlights,highlights[1:]))
        assert all(0<=h['importance']<=1 for h in highlights)


def test_diversity_prefers_unseen_pair():
    candidates=[dict(start=0,end=4,verb='wash',noun='cup',confidence=.99),
                dict(start=4,end=8,verb='wash',noun='cup',confidence=.98),
                dict(start=8,end=12,verb='cut',noun='onion',confidence=.8)]
    selected=select(candidates,60)
    assert [h['start'] for h in selected]==[0,8]


@pytest.fixture
def valid_result():
    return dict(video_id='test',label='unclassified',actions=[],
                highlights=[dict(start=1.,end=3.,importance=.5)])


@pytest.mark.parametrize('bad',[float('nan'),float('inf'),True,'0.5',-1,2])
def test_invalid_importance_rejected(valid_result,bad):
    valid_result['highlights'][0]['importance']=bad
    with pytest.raises(ValueError):
        validate(valid_result,120)


def test_overlap_and_budget_rejected(valid_result):
    valid_result['highlights'] += [dict(start=2.,end=4.,importance=.3)]
    with pytest.raises(ValueError):
        validate(valid_result,120)
    valid_result['highlights']=[dict(start=0,end=19,importance=.3)]
    with pytest.raises(ValueError):
        validate(valid_result,120)


def test_vocabulary_rejected(valid_result):
    valid_result['actions']=[dict(start=0,end=1,verb='unknown',noun='cup')]
    with pytest.raises(ValueError):
        validate(valid_result,120,{('wash','cup')})


def test_atomic_failure_preserves_existing(tmp_path):
    output=tmp_path/'result.json'
    atomic_json(output,{'valid':True})
    with pytest.raises(ValueError):
        atomic_json(output,{'invalid':float('nan')})
    assert json.loads(output.read_text())=={'valid':True}
    assert len(list(tmp_path.iterdir()))==1


def test_default_requires_trained_weights(video):
    with pytest.raises(ValueError,match='No trained action'):
        analyze(video)


def test_motion_baseline_does_not_invent_actions(video):
    result,report=analyze(video,mode='motion-baseline')
    assert result['actions']==[] and result['label']=='unclassified'
    assert result['highlights']
    assert report['dental']['enabled'] is False
    assert report['official_schema_verified'] is False
    validate(result,120)


def test_cli_from_other_directory(video,tmp_path):
    root=Path(__file__).resolve().parents[1]
    result=subprocess.run(['bash',str(root/'run.sh'),str(video),str(tmp_path/'result.json'),
                           '--mode','infrastructure','--format','review'],cwd=tmp_path,capture_output=True,text=True,
                          env={**os.environ,'DENTILENS_PYTHON':sys.executable})
    assert result.returncode==0,result.stderr
    assert json.loads((tmp_path/'result.json').read_text())['actions']==[]
    assert 'DEVELOPMENT OUTPUT' in result.stderr


def test_cli_failure_no_output(video,tmp_path):
    output=tmp_path/'absent.json'
    root=Path(__file__).resolve().parents[1]
    result=subprocess.run([sys.executable,str(root/'code/main.py'),str(video),str(output)],capture_output=True)
    assert result.returncode==1 and not output.exists()


def test_input_cannot_be_output(video):
    root=Path(__file__).resolve().parents[1]
    size=video.stat().st_size
    result=subprocess.run([sys.executable,str(root/'code/main.py'),str(video),str(video)],capture_output=True)
    assert result.returncode==1 and video.stat().st_size==size
