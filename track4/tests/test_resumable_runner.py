"""Tests for the resumable dataset acquisition and processing runner."""
from hashlib import md5
import json
from pathlib import Path
import pytest

from check_submission import StudentKit
from resumable_acquisition_runner import (
    AcquisitionState,
    FREE_RESERVE_GIB,
    check_storage_safety,
    verify_media_file,
)


def test_acquisition_state_initialization_and_resume(tmp_path):
    state_file = tmp_path / 'acquisition_state.json'
    kit = StudentKit(Path('track4/reference/student_package'))
    excluded = {'Scuba_KD74Ywuy9ng', 'Scuba_Um4kqB8bQBg'}
    eligible_keys = [k for k in kit.entries if k not in excluded]

    state = AcquisitionState(state_file, team_id='Rajasthani', eligible_keys=eligible_keys)
    assert state.get_counts()['total_eligible'] == 185
    assert state.get_counts()['validated'] == 0

    # Mark some as validated
    state.update('P01_13', status='validated', duration_sec=93.76)
    assert state.get_status('P01_13') == 'validated'
    assert state.get_counts()['validated'] == 1

    # Reload state from disk to test persistence
    state2 = AcquisitionState(state_file, team_id='Rajasthani', eligible_keys=eligible_keys)
    assert state2.get_status('P01_13') == 'validated'
    assert state2.get_counts()['validated'] == 1


def test_storage_safety_check(tmp_path):
    # Safe check when requested size is small
    is_safe, free_gb, msg = check_storage_safety(tmp_path, needed_bytes=100 * 1024 * 1024, reserve_bytes=1024 * 1024)
    assert is_safe
    assert free_gb > 0

    # Unsafe check when needed_bytes exceeds free headroom
    is_safe, _, msg = check_storage_safety(tmp_path, needed_bytes=100 * 1024 ** 4, reserve_bytes=1024 * 1024)
    assert not is_safe
    assert 'insufficient' in msg.lower()


def test_verify_media_file_md5(tmp_path):
    fake_video = tmp_path / 'test.mp4'
    payload = b'test video content for md5 verification'
    fake_video.write_bytes(payload)
    expected_hash = md5(payload).hexdigest()

    # Matching hash
    is_valid, err = verify_media_file(fake_video, expected_md5=expected_hash, check_opencv=False)
    assert is_valid
    assert err is None

    # Mismatched hash
    is_valid, err = verify_media_file(fake_video, expected_md5='bad_hash', check_opencv=False)
    assert not is_valid
    assert 'checksum mismatch' in err.lower()
