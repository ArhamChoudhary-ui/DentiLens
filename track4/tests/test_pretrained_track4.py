import numpy as np
import pytest

from egosight.pretrained import frame_tensor, LABEL_MAP, load_verified
from egosight.contract import StudentKit
from evaluate_local import epic_scores
from evaluate_fpv_local import highlight_f1


def test_model_frame_preprocessing_preserves_channel_order():
    image=np.zeros((160,320,3),dtype=np.uint8)
    image[:,:,2]=255
    result=frame_tensor(image)
    assert result.shape==(3,224,224)
    assert result[0].min()==255 and result[2].max()==0


def test_weights_require_the_pinned_hash(tmp_path):
    p=tmp_path/'model.pth'
    with pytest.raises(ValueError,match='Missing'):
        load_verified(p,'bad')
    p.write_bytes(b'fixture')
    with pytest.raises(ValueError,match='hash mismatch'):
        load_verified(p,'bad')


def test_category_map_uses_exact_supported_vocabulary():
    kit=StudentKit()
    assert set(LABEL_MAP)<=kit.labels and 'ski' in LABEL_MAP and 'skiing' not in LABEL_MAP
    assert 'plane' not in LABEL_MAP


def test_epic_metric_uses_one_to_one_matches():
    action=dict(verb='take',noun='cup',segment=[0.,4.])
    pred={'X':{'actions':[dict(action,score=.8),dict(action,score=.7)],'highlights':[{'segment':[0.,2.]}]}}
    metrics=epic_scores(pred,{'X':[action]})
    assert metrics['local_action_map']['0.5']==1.
    assert metrics['local_coverage_at_half_action']['X']['covered_pairs']==1


def test_fpv_frame_metric_counts_highlight_once():
    scores=np.arange(20.)
    assert highlight_f1(scores,[{'segment':[17.,20.]}],20.)['f1']==1.
