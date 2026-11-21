import json

from dentilens.contract import ManifestEntry, StudentKit
from run_manifest import run_manifest


def test_manifest_batch_preserves_results_and_resumes(tmp_path, monkeypatch):
    videos, outputs = tmp_path / 'videos', tmp_path / 'outputs'
    videos.mkdir()
    outputs.mkdir()
    report = tmp_path / 'audit.json'
    kit = StudentKit()
    first, second, missing, duplicate, corrupt = 'first', 'second', 'missing', 'duplicate', 'corrupt'
    kit.entries = {key: ManifestEntry(key, 40, 'epic_kitchens_100')
                   for key in (first, second, missing, duplicate, corrupt)}

    def output(video_id):
        return dict(team_id='TEST', video_id=video_id, duration_sec=40,
                    label={'name': 'cooking', 'score': 1}, actions=[],
                    highlights=[{'segment': [0, 4]}])

    def infer(command, **_kwargs):
        from pathlib import Path
        target = Path(command[3])
        assert target.stem == second
        target.write_text(json.dumps(output(second)))

    (videos / 'second.MP4').touch()
    (videos / 'duplicate.mp4').touch()
    (videos / 'nested').mkdir()
    (videos / 'nested/duplicate.mp4').touch()
    (videos / 'missing.mov').touch()
    first_json = outputs / 'first.json'
    original = json.dumps(output(first), indent=4)
    first_json.write_text(original)
    (outputs / 'corrupt.json').write_text('{invalid')
    (outputs / 'impostor.json').write_text(original)
    monkeypatch.setattr('run_manifest.subprocess.run', infer)
    audit = run_manifest(videos, outputs, report, 'TEST', kit=kit)
    assert audit['valid_outputs'] == 2
    assert audit['duplicate_output_ids'] == {first: ['first.json', 'impostor.json']}
    assert audit['missing_input_ids'] == [first, missing, corrupt]
    assert audit['videos'][missing]['other_extensions'] == ['missing.mov']
    assert audit['status_counts'] == {'reused': 1, 'processed': 1, 'missing_input': 1,
                                     'duplicate_input': 1, 'invalid_output': 1}
    assert first_json.read_text() == original

    def unexpected_inference(*_args, **_kwargs):
        raise AssertionError('A valid output was processed twice')

    monkeypatch.setattr('run_manifest.subprocess.run', unexpected_inference)
    resumed = run_manifest(videos, outputs, report, 'TEST', kit=kit)
    assert resumed['videos'][second]['status'] == 'reused'
    assert resumed['videos'][second]['inference_seconds'] == audit['videos'][second]['inference_seconds']
