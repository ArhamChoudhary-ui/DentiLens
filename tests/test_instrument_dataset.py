"""Regression checks for source integrity, geometry and independent metrics."""
import sys
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from build_dataset import build
from evaluate_classifier import classification_metrics
from evaluate_detector import detection_metrics
from instrument_dataset import SOURCES, leakage, load_dataset, parse_labels, validate, write_json
from instrument_identity import encoded_crop
from instrument_taxonomy import canonical

POLYGON = '0 .1 .1 .5 .1 .5 .5 .1 .5'


def test_source_mapping_and_invalid_geometry():
    assert canonical('Explorer No.5','dentalguard') == 'dental_explorer'
    assert canonical('periodontal probe?','dentalguard') == 'unknown'
    assert canonical('hand_gloved','dentalguard') == 'non_instrument'
    assert canonical('Burnisher','dentalguard') == 'other_instrument'
    assert parse_labels('',{0:'Mouth mirror'}) == []
    parsed = parse_labels(POLYGON, {0:'Mouth mirror'})
    assert parsed[0]['box'] == [.1,.1,.5,.5]
    for bad in ('1 .1 .1 .5 .1 .5 .5 .1 .5','0 .1 .1 .5 .1 .5 .5',
                '0 -.1 .1 .5 .1 .5 .5 .1 .5','0 .1 .1 .1 .1 .1 .1 .1 .1'):
        with pytest.raises((ValueError,KeyError)):
            parse_labels(bad,{0:'Mouth mirror'})


def test_exact_near_recording_and_patient_leakage():
    train = dict(id='a',recording='1',patient='p',split='train',sha256='exact',dhash='0')
    test = dict(train,id='b',split='test',dhash='3')
    report = leakage([train,test])
    assert report['recording_conflicts'] == ['1']
    assert report['patient_conflicts'] == ['p']
    assert report['exact_conflicts'] == ['exact']
    assert report['near_duplicate_pairs'] == [['b','a']]
    assert not leakage([train,dict(test,recording='2',patient=None,sha256='distinct',dhash='ffffffff')])['near_duplicate_pairs']


def test_archive_build_reproducibility_and_manifest_validation(tmp_path):
    archive = tmp_path/'archive.zip'
    source = SOURCES['dentalguard']
    image = np.zeros((32,40,3),dtype=np.uint8)
    cv2.rectangle(image,(3,3),(18,14),(200,230,255),-1)
    _, jpeg = cv2.imencode('.jpg', image)
    with ZipFile(archive,'w') as handle:
        handle.writestr(source['evidence'],source['url']+' '+source['license'])
        handle.writestr('data.yaml','names: {0: Mouth mirror}')
        for frame in (0,1):
            filename = f'101_frame_{frame}'
            handle.writestr(f'train/images/{filename}.jpg',jpeg.tobytes())
            handle.writestr(f'train/labels/{filename}.txt',POLYGON)
        handle.writestr('train/images/101_frame_2.jpg',jpeg.tobytes())
        handle.writestr('train/labels/101_frame_2.txt','')
    metadata,frames,crops = first = build(archive,{'101':'test'})
    assert first == build(archive,{'101':'test'})
    assert len(frames)==len(crops)==1 and len(metadata['exclusions'])==2
    assert encoded_crop(image,crops[0]['box'],{'color':True}).shape == (96,96,3)
    for sub,filename,data in [('manifests','dataset.json',metadata),('intraoral_detection','frames.json',frames),
                              ('instrument_classification','crops.json',crops)]:
        (tmp_path/sub).mkdir()
        write_json(tmp_path/sub/filename,data)
    stored,stored_frames,stored_crops = load_dataset(tmp_path)
    assert not validate(stored,stored_frames,stored_crops)['near_duplicate_pairs']
    with pytest.raises(ValueError,match='invalid parent'):
        validate(metadata,frames,[dict(crops[0],split='train')])
    with pytest.raises(ValueError,match='benchmark registry'):
        validate(dict(metadata,provenance={'license':None}),frames,crops)


def test_detection_rank_and_negative_classification():
    box = [.1,.1,.5,.5]
    expected = dict(label='dental_mirror',box=box)
    context = dict(original_class='hand_gloved',box=[.6,.6,.8,.8])
    result = detection_metrics([([expected],[(.9,box),(.8,box),(.7,context['box'])],[context])],.5)
    assert (result['tp'],result['fp'],result['fn']) == (1,2,0)
    assert result['ap50'] == result['map50_95'] == 1
    assert result['false_positives_overlapping_context'] == {'hand_gloved':1}
    rows = [dict(class_id='dental_mirror', confidence=.9, accepted=True,
                 alternatives=[{'class_id':'dental_mirror'}])]*2
    metrics = classification_metrics(['dental_mirror','non_instrument'],rows)
    assert metrics['raw_accuracy']==.5
    assert metrics['negative_false_acceptance']==1
    assert metrics['accepted_accuracy']==.5
