"""Check instrument scan and review integration with deterministic test detections."""


from test_full_exam import examination_video, processor
from backend.main import create_app
from fastapi.testclient import TestClient
from instrument_video import group_instruments, measure_instruments
from full_video import video_info
from reviewed_video import instrument_marks


def test_instrument_groups_do_not_fill_unsampled_gaps():
    def row(frame, label):
        return dict(frame=frame, time=frame / 25, detection_id=f'instrument-{frame}-0',
                    track_id='I-4', confidence=.8, **{'class': label})
    rows = [row(0, 'Dental mirror'), row(5, 'Dental mirror'), row(10, 'Forceps'), row(20, 'Dental mirror')]
    spans = group_instruments(rows, 5)
    assert [item['samples'] for item in spans] == [2, 1, 1]
    assert spans[0]['end'] == .2


def test_instrument_scan_api_and_full_video_export(monkeypatch, examination_video, tmp_path):
    import backend.main as backend
    import time
    polygon = [[.3,.2],[.65,.3],[.65,.65],[.3,.55]]
    def scan(path, video, models):
        return measure_instruments(path, video, models, detector=lambda frame: [('Dental mirror', .8, polygon, 7)])
    monkeypatch.setattr(backend, 'instrument_model_ready', lambda root: True)
    monkeypatch.setattr(backend, 'measure_instruments', scan)
    storage = tmp_path / 'exam-store'
    with TestClient(create_app(storage, processor, serve_frontend=False)) as client:
        response = client.post('/api/exams', files={'video':('fixture.mp4', examination_video.read_bytes(),'video/mp4')})
        exam = response.json()
        path = '/api/exams/' + exam['id']
        for _ in range(120):
            if client.get(path).json()['state'] == 'ready':
                break
            time.sleep(.05)
        manual = client.post(path + '/instrument-reviews', json={
            'time': .22, 'decision': 'confirmed', 'label': 'Wire cutter', 'box': [.03,.2,.4,.5]})
        assert manual.status_code == 200, manual.text
        assert manual.json()['original_label'] is None and manual.json()['time'] == .2
        measured = client.post(path + '/instruments')
        assert measured.status_code == 200, measured.text
        assert measured.json()['samples'] == 15
        assert len(measured.json()['observations']) == 16
        assert not client.get(path + '/intelligence', params={'q':'Show mirror'}).json()['results']
        marks = instrument_marks(measured.json())
        assert list(marks) == [2] and len(marks[2]) == 1
        first = measured.json()['appearances'][0]['best']
        reviewed = client.post(path + '/instrument-reviews', json={
            'detection_id':first['detection_id'], 'decision':'confirmed', 'label':'Reviewed explorer',
            'note':'Training fixture', 'box':[.05,.2,.4,.5]})
        assert reviewed.json()['original_label'] == 'Dental mirror'
        assert not client.get(path + '/intelligence', params={'q':'Show mirror'}).json()['results']
        indexed = client.get(path + '/intelligence', params={'q':'Show explorer'}).json()['results'][0]
        assert indexed['status'] == 'confirmed' and indexed['time'] == first['time']
        assert not client.get(path + '/intelligence', params={'q':'Instrument at 00:00.10'}).json()['results']
        at_time = client.get(path + '/intelligence', params={'q':'Instrument at 00:00.20'}).json()['results']
        assert len(at_time) == 1 and at_time[0]['label'] == 'Wire cutter'
        assert at_time[0]['time'] == .2
        index = client.get(path + '/evidence-index').json()
        assert len(index['instruments']['reviews']) == 2
        marks = instrument_marks(index['instruments'])
        assert marks[0][0]['box'] == [.05,.2,.4,.5] and marks[0][0]['polygon'] is None
        assert {'Wire cutter','Reviewed explorer'} == {row['label'] for entries in marks.values() for row in entries}
        assert 'observations' not in client.get(path + '/export').json()['instrument_scan']
        result = client.post(path + '/video-export', json={'version':'confirmed','instruments':True})
        assert result.status_code == 200, result.text
        assert video_info(storage / exam['id'] / 'confirmed-export.mp4')['frames'] == 30
        assert reviewed.json()['review_action'] == 'corrected'
        assert client.get(path + '/still.jpg', params={'instrument': first['detection_id']}).headers['content-type'] == 'image/jpeg'
        rejected = client.post(path + '/instrument-reviews',json={'detection_id':first['detection_id'],'decision':'dismissed'}).json()
        assert rejected['review_action'] == 'rejected'
        assert [row['action'] for row in rejected['history']] == ['corrected','rejected']
        assert rejected['original_label'] == 'Dental mirror'
        assert not client.get(path + '/intelligence', params={'q':'Show explorer'}).json()['results']
        non_tool = client.post(path+'/instrument-reviews',json={'detection_id':first['detection_id'],
                  'decision':'dismissed','non_instrument':True}).json()
        assert non_tool['review_action']=='non_instrument'
        assert non_tool['original_label']=='Dental mirror'
        # Existing experimental sessions can load the canonical track timeline
        # without changing the raw detector sidecar or legacy appearance shape.
        import json
        session = storage / exam['id']
        original_boxes = [
            dict(detection_id=f'instrument-video-{frame}-0', time=frame/10, frame=frame,
                 confidence=.95, box=[.1,.1,.3,.4], **{'class':'Dental instrument'})
            for frame in (0,2,4)
        ]
        scan = session/'instruments-video.json'
        raw = json.dumps({'observations':original_boxes,'samples':3})
        scan.write_text(raw)
        response = client.get(path+'/instruments')
        assert response.status_code==200
        report = response.json()['report']
        assert report['diagnostics']['canonical_proposals']==3
        assert report['tracks'][0]['canonical_detection_ids']==[row['detection_id'] for row in original_boxes]
        assert any(row['time']==0 and row.get('track_id')=='V-1' for row in report['appearances'])
        assert scan.read_text()==raw




def test_identity_association_and_confidence():
    import numpy as np
    import torch
    from instrument_identity import associate, classify

    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(96*96,2))
    with torch.no_grad():
        model[1].weight.zero_()
        model[1].bias.copy_(torch.tensor([3., 0.]))
    info = {'classes':['dental_mirror','unknown'],'thresholds':{'dental_mirror':.9},'model':'test'}
    image = np.zeros((128,128,3),dtype=np.uint8)
    box = [.1,.1,.4,.4]
    prediction = classify(model,info,image,[box])[0]
    assert prediction['label']=='Dental mirror' and prediction['accepted']
    info['thresholds']['dental_mirror'] = .99
    assert not classify(model,info,image,[box])[0]['accepted']
    tracks = list(associate([{'box':box,'time':0}, {'box':[.12,.1,.4,.4],'time':.2},
                             {'box':[.6,.6,.9,.9],'time':.2}, {'box':box,'time':.8}]))
    assert [len(track) for track in tracks]==[2,1,1]


def test_moving_instrument_association_checks_both_directions():
    from instrument_identity import associate

    mirror = [
        {'box':[.16,.14,.36,.25],'time':1.0},
        {'box':[.25,.18,.48,.28],'time':1.2},
        {'box':[.35,.21,.58,.30],'time':1.4},
    ]
    explorer = {'box':[.66,.23,1,.29],'time':1.6}
    assert [len(track) for track in associate([*mirror,explorer])]==[3,1]
    ambiguous = {'box':[.08,.12,.28,.22],'time':1.2}
    assert [len(track) for track in associate([mirror[0],mirror[1],ambiguous])]==[1,1,1]


def test_identity_pipeline_keeps_evidence_and_abstains_on_disagreement(monkeypatch):
    import instrument_identity as identity

    base = {'same':{'class_id':'dental_mirror','accepted':True},
            'conflict':{'class_id':'dental_explorer','accepted':True},
            'other':{'class_id':'wire_cutter','accepted':True}}
    detail = {'same':{'class_id':'dental_mirror','accepted':True},
              'conflict':{'class_id':'dental_mirror','accepted':True},'other':None}
    for head in (base,detail):
        for row in head.values():
            if row:
                row['frame_accepted'] = row['accepted']
    monkeypatch.setattr(identity,'predict_video',lambda source,rows,models:[dict(model) for model,_ in models])
    result = identity.classify_pipeline(None,[],[(base,{}),(detail,{})])
    assert result['same']['primary_prediction']==base['same']
    assert result['conflict']['reason']=='classifier_disagreement' and not result['conflict']['accepted']
    assert result['other']==base['other']

def test_nested_boxes_preserve_one_mirror_track():
    from instrument_identity import associate

    left = [[.404,.261,.651,.348],[.400,.280,.638,.363],[.404,.275,.634,.362]]
    larger = [[.323,.193,.647,.349],[.318,.201,.629,.364],[.408,.196,.633,.358]]
    rows = [dict(detection_id=f'{i}-{j}',time=2.2+i*.2,box=box)
            for i,(a,b) in enumerate(zip(left,larger)) for j,box in enumerate((a,b))]
    assert [len(track) for track in associate(rows)]==[3]
    assert [row['duplicate_of'] for row in rows[1::2]]==['0-0','1-0','2-0']


def candidate(name,time,box,kind=None,score=.99):
    prediction = dict(class_id=kind,label=kind,confidence=score,accepted=bool(kind)) if kind else None
    return dict(detection_id=name,frame=round(time*25),time=time,box=box,identity=prediction,confidence=score)


def test_tracking_keeps_identity_through_occlusion_but_expires():
    from instrument_identity import describe_tracks
    mirror = [.25,.35,.49,.55]
    rows = [
        candidate('a',0,mirror,'dental_mirror'),
        candidate('b',.2,[.28,.37,.48,.54],'dental_mirror'),
        candidate('c',.6,[.29,.38,.47,.56]),
        candidate('d',1.2,mirror,'dental_mirror'),
    ]
    tracks,counts = describe_tracks(rows,{'fps':25,'duration':1.3},[])
    assert [track['samples'] for track in tracks]==[3,1]
    assert tracks[0]['identity']['supporting_frames']==2
    assert tracks[0]['gaps'][0]['reason']=='missed_sample'
    assert rows[2]['track_identity']['class_id']=='dental_mirror'
    assert tracks[0]['termination_status']=='track_lost'
    assert rows[3]['instrument_status']=='unclassified_candidate'
    assert counts['one_frame_tracks']==1


def test_context_annotation_does_not_discard_a_mixed_crop():
    from instrument_identity import describe_tracks
    box = [.32,.31,.55,.55]
    context = dict(class_id='hand_gloved',label='Gloved hand',accepted=True,confidence=.98)
    rows = [candidate(str(i),i/5,box) for i in range(2)]
    for row in rows:
        row['identity'] = dict(class_id='unknown',accepted=False,reason='class_unknown',confidence=.88,context=context)
    tracks,counts = describe_tracks(rows,{'fps':25,'duration':.3},[])
    assert counts['proposal_statuses']=={'context_detected':2}
    assert tracks[0]['context_identity']['label']=='Gloved hand'
    assert tracks[0]['identity'] is None
    review = {'detection_id':'0','decision':'dismissed','review_action':'non_instrument'}
    _,counts = describe_tracks(rows,{'fps':25,'duration':.3},[review])
    assert counts['proposal_statuses']['non_instrument']==1
    assert rows[0]['identity']['context']==context


def test_overlap_cannot_merge_different_instrument_classes_or_erase_boxes():
    from instrument_identity import describe_tracks
    a = candidate('mirror',0,[.32,.31,.55,.55],'dental_mirror')
    b = candidate('explorer',0,[.33,.31,.54,.55],'dental_explorer')
    c = candidate('duplicate',0,[.33,.32,.53,.53],'dental_mirror',.91)
    original = (a['box'][:],b['box'][:],c['box'][:])
    tracks,counts = describe_tracks([a,b,c],{'fps':25,'duration':.1},[])
    assert counts['raw_proposals']==3 and counts['canonical_proposals']==2
    assert c['duplicate_of']=='mirror'
    assert b['track_id']!=a['track_id']
    assert (a['box'],b['box'],c['box'])==original
    assert any(item['raw_detection_ids']==['mirror','duplicate'] for item in tracks)
