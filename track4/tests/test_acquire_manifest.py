from hashlib import md5
from types import SimpleNamespace

from acquire_manifest import epic_download, mapped_sources


def test_original_source_mapping_preserves_fpv_variant_ids(tmp_path):
    metadata = {
        'epic_55_splits.csv': 'video_id,split\nP01_11,test\n',
        'md5.csv': ('file_remote_path,md5\nvideos/test/P01/P01_11.MP4,old\n'
                    'P02/videos/P02_109.MP4,new\n'),
        'errata.csv': 'rdsf_path,dropbox_path\n',
        'fpvsum-videolist.csv': ('video_name,youtube_id\n'
                                  'BikePolo_5yf797Lh4Oc_1,6oOSbkUiJY0\n'),
    }
    for filename, value in metadata.items():
        (tmp_path / filename).write_text(value)
    kit = SimpleNamespace(entries={
        'P01_11': SimpleNamespace(dataset='epic_kitchens_100'),
        'P02_109': SimpleNamespace(dataset='epic_kitchens_100'),
        'BikePolo_5yf797Lh4Oc_1': SimpleNamespace(dataset='fpvsum'),
    })
    result = mapped_sources(kit, tmp_path)
    assert result['P01_11'][0].endswith('/videos/test/P01/P01_11.MP4')
    assert result['P02_109'][0].endswith('/P02/videos/P02_109.MP4')
    assert result['BikePolo_5yf797Lh4Oc_1'] == ('https://www.youtube.com/watch?v=6oOSbkUiJY0', None)


def test_fully_downloaded_partial_is_verified_without_network(tmp_path):
    partial = tmp_path / 'clip.part'
    output = tmp_path / 'clip.mp4'
    payload = b'complete source bytes'
    partial.write_bytes(payload)
    epic_download(output, partial, 'unused', md5(payload).hexdigest(), len(payload))
    assert output.read_bytes() == payload
    assert not partial.exists()
