import io
import zipfile

from fastapi.testclient import TestClient

from backend.main import create_app


def test_action_settings_and_upload_rejection(tmp_path):
    app = create_app(storage=tmp_path, serve_frontend=False)
    workspace = next(route.endpoint.__self__ for route in app.routes if route.path == "/api/action/settings")

    with TestClient(app) as client:
        assert client.get('/api/health').json()['status'] == 'ok'
        assert 'ski' in client.get('/api/action/settings').json()['labels']
        assert client.get('/api/action/runs').json() == []
        path = '/api/action/runs'
        assert client.post(path, files={'video': ('text.txt', b'bad')}).status_code == 415
        assert client.post(path, files={'video': ('unreadable.mp4', b'bad')}).status_code == 422
        assert not list(tmp_path.iterdir())
        assert workspace.slot.acquire(blocking=False)
        try:
            assert client.post(path, files={'video': ('busy.mp4', b'bad')}).status_code == 409
        finally:
            workspace.slot.release()


def test_action_dataset_and_metrics_endpoints(tmp_path):
    app = create_app(storage=tmp_path, serve_frontend=False)

    with TestClient(app) as client:
        # Test dataset endpoint
        dataset_res = client.get('/api/action/dataset')
        assert dataset_res.status_code == 200
        dataset = dataset_res.json()
        assert dataset['summary']['total_manifest'] == 187
        assert dataset['summary']['eligible'] == 185
        assert dataset['summary']['validated'] in (102, 103, 104)
        assert dataset['summary']['missing'] in (81, 82, 83)
        assert dataset['summary']['excluded'] == 2
        assert dataset['disk']['free_gb'] > 0
        assert dataset['network']['measured_rate_kbps'] in (275.4, 319.5)
        assert len(dataset['videos']) == 187
        assert 'pipeline' in dataset
        assert 'previously_validated' in dataset['summary']

        # Test batch status and retry
        batch_status_res = client.get('/api/action/batch/status')
        assert batch_status_res.status_code == 200
        assert 'is_running' in batch_status_res.json()

        batch_retry_res = client.post('/api/action/batch/retry')
        assert batch_retry_res.status_code == 200
        assert batch_retry_res.json()['status'] == 'ok'

        # Test metrics endpoint
        metrics_res = client.get('/api/action/metrics')
        assert metrics_res.status_code == 200
        metrics = metrics_res.json()
        assert metrics['summary']['overall_accuracy'] == 0.784
        assert metrics['models']['activity_classification']['top1_accuracy'] == 0.784
        assert len(metrics['highlight_ablation']) == 4
        assert metrics['highlight_ablation'][0]['method'] == 'Multi-Signal Utility (EgoSight XR)'

        # Test export/all endpoint
        export_res = client.get('/api/action/export/all')
        assert export_res.status_code == 200
        assert export_res.headers['content-type'] == 'application/zip'
        with zipfile.ZipFile(io.BytesIO(export_res.content), 'r') as zf:
            namelist = zf.namelist()
            assert len(namelist) in (102, 103, 104)
            assert 'Surfing_7bzMGVKaiCQ.json' in namelist

        # Test acquire endpoint
        bad_req = client.post('/api/action/batch/acquire', json={})
        assert bad_req.status_code == 400

        excluded_req = client.post('/api/action/batch/acquire', json={'video_id': 'Scuba_KD74Ywuy9ng'})
        assert excluded_req.status_code == 400

        validated_req = client.post('/api/action/batch/acquire', json={'video_id': 'Surfing_7bzMGVKaiCQ'})
        assert validated_req.status_code == 200
        assert validated_req.json()['status'] == 'already_validated'

        missing_req = client.post('/api/action/batch/acquire', json={'video_id': 'P01_11'})
        assert missing_req.status_code in (200, 507)

