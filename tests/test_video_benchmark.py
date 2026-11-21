"""Permission, CVAT and complete-video benchmark regression tests."""
import sys
from pathlib import Path

import pytest

from test_full_exam import examination_video

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from video_benchmark_data import (check_isolation, digest, extract, read_cvat, register, require,
                                 reviewed_annotations, sampling, source_path)
from video_benchmark_metrics import track_metrics


def sample_record(path):
    return dict(id='synthetic-fixture',name='Decoder test signal',reference='Locally generated test signal',
                category='training_demo',recording_group='fixture',split='valid',status='authorized',
                identifiable=False,license='Locally generated test fixture',rights_reference='Test fixture generator',
                rights_reviewer='Fixture owner',permissions=dict(view=True,store=True,extract=True,annotate=True,
                                                               evaluate=True,retain=True),cfr_verified=True,path=str(path))


def test_permissions_quarantine_and_split_isolation(examination_video):
    entry=sample_record(examination_video)
    pending=register(dict(entry,status='pending',path='/missing'))
    assert pending['split']=='quarantine' and 'video' not in pending
    with pytest.raises(ValueError,match='quarantined'):
        source_path(pending,'evaluate')
    record=register(entry)
    assert record['video']==dict(width=320,height=180,fps=10.,frames=30,duration=3.)
    with pytest.raises(ValueError,match='Missing permission'):
        source_path(record,'train')
    with pytest.raises(ValueError,match='Explicit consent'):
        require(dict(record,identifiable=True),'evaluate')
    with pytest.raises(ValueError,match='expired'):
        require(dict(record,expires='2000-01-01'),'evaluate')
    with pytest.raises(ValueError,match='untouched'):
        register(dict(entry,split='test'))
    with pytest.raises(ValueError,match='recording_group'):
        check_isolation([record,dict(record,id='other',split='train')])
    with pytest.raises(ValueError,match='patient_group'):
        check_isolation([dict(record,patient_group='p'),dict(record,id='other',recording_group='g',
                        patient_group='p',split='train')])


def test_cvat_round_trip_and_tracking_metrics(examination_video,tmp_path):
    record=register(sample_record(examination_video))
    extract(record,tmp_path/'extracted')
    assert len(list((tmp_path/'extracted').glob('*.jpg')))==15
    cvat=tmp_path/'annotation.xml'
    cvat.write_text('''<annotations>
      <track id="10" label="dental_mirror">
        <box frame="0" xtl="32" ytl="18" xbr="96" ybr="54" outside="0"/>
        <box frame="2" xtl="64" ytl="18" xbr="128" ybr="54" outside="0" occluded="1"/>
        <box frame="4" xtl="64" ytl="18" xbr="128" ybr="54" outside="1"/>
      </track>
      <track id="11" label="dental_explorer">
        <box frame="2" xtl="64" ytl="18" xbr="128" ybr="54" outside="0"/>
        <box frame="4" xtl="64" ytl="18" xbr="128" ybr="54" outside="1"><attribute name="visibility">invisible</attribute></box>
        <box frame="6" xtl="64" ytl="18" xbr="128" ybr="54" outside="1"/>
      </track>
    </annotations>''')
    approval=dict(review_state='verified',reviewer='fixture-reviewer',annotation_sha256=digest(cvat),
                  source_sha256=record['sha256'],frames=list(sampling(record['video'])))
    with pytest.raises(ValueError,match='every scheduled'):
        reviewed_annotations(record,cvat,dict(approval,frames=[0,2]))
    truth=reviewed_annotations(record,cvat,approval)
    assert truth['events'][0]==dict(track_id='10',label='dental_mirror',entry=0.,last_seen=.2,exit=.4)
    assert truth['events'][1]['exit'] is None
    assert any(row['frame']==4 and row['visibility']=='invisible' and row['box'] is None for row in truth['objects'])
    known=[row for row in truth['objects'] if row['box'] is not None and row['track_id']=='10']
    predictions=[dict(frame=row['frame'],box=row['box'],track_id='same',
                      track_identity={'accepted':True,'class_id':'dental_mirror'}) for row in known]
    duplicate=dict(predictions[1],duplicate_of='one')
    result=track_metrics(truth,predictions+[duplicate],record['video'])
    assert result['ground_truth_detections']==3 and result['predicted_detections']==2
    assert result['idf1']==.8 and result['identity_switches']==0
    assert result['missed_visible_seconds']==pytest.approx(.2)
    assert result['correctly_named_visible_seconds']==pytest.approx(.4)
    cvat.write_text('<annotations><track id="0" label="dental_mirror"><box frame="1" xtl="0" ytl="0" xbr="900" ybr="40"/></track></annotations>')
    with pytest.raises(ValueError,match='Invalid CVAT box'):
        read_cvat(cvat,record['video'])


def test_annotation_comparison_separates_label_disagreement_from_missing_box():
    from video_benchmark_metrics import annotation_disagreements

    base=dict(track_id='t',frame=2,box=[.1,.1,.3,.3],label='dental_mirror',visibility='visible')
    first=dict(recording='source',frames=[0,2],objects=[base],review={'reviewer':'first'})
    second=dict(first,review={'reviewer':'second'},
                objects=[dict(base,label='dental_explorer'),dict(base,box=[.5,.5,.7,.7])])
    report=annotation_disagreements(first,second)
    assert [row['reason'] for row in report['disagreements']]==['class_or_visibility','unmatched_box']
    with pytest.raises(ValueError,match='different reviewer'):
        annotation_disagreements(first,first)


def test_tile_coordinates_and_multiscale_merging():
    from detector_tiles import merge, remap, windows

    footprints=windows(640,480,'tiles')
    assert len(footprints)==4
    assert remap([0,0,1,1],footprints[-1],640,480)==pytest.approx([.35,.35,1,1])
    assert windows(640,480,'roi')[0]==(0,0,640,480)
    nearby=[(.8,[.1,.1,.2,.2]),(.9,[.101,.1,.201,.2]),(.3,[.4,.4,.5,.5])]
    assert merge(nearby)==[nearby[1],nearby[2]]
    assert merge([])==[]


def test_clearance_packet_keeps_all_existing_leads_unverified(tmp_path):
    from audit_source_clearance import ACTIONS, package
    from instrument_dataset import ROOT, read_json

    registry=ROOT/'datasets/video_sources'
    output=tmp_path/'requests'
    package(registry,output,[])
    rows=read_json(output/'worksheet.json')
    assert len(rows)==13
    assert all(row['status']=='pending' and len(row['missing'])==8 for row in rows)
    assert all(set(row['permissions'])==set(ACTIONS) for row in rows)
    assert len(list(output.glob('draft-*.md')))==13
    assert 'huggingface.co' in (output/'draft-mobiusi-dental-action-sample.md').read_text()
    assert 'subject consent' in (output/'request-template.md').read_text()


def test_source_hash_and_derived_retention_gates(examination_video):
    approved=register(sample_record(examination_video))
    with pytest.raises(ValueError,match='authorization is inactive'):
        source_path(dict(approved,status='revoked'),'evaluate')
    with pytest.raises(ValueError,match='retain'):
        require(dict(approved,permissions={k:v for k,v in approved['permissions'].items() if k!='retain'}),'retain')
    with pytest.raises(ValueError,match='changed'):
        source_path(dict(approved,sha256='0'*64),'evaluate')
    with pytest.raises(ValueError,match='Duplicate video'):
        check_isolation([approved,dict(approved,id='copy')])
