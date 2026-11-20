"""Local dental video upload and review application."""
import json
from pathlib import Path
import subprocess
import sys
import uuid
import tempfile

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'code'))
from dental_modes import TRAINED_MODES
from dental_video import export_clips
RESULTS = ROOT / 'results'
RUNTIME = ROOT / 'runtime'

st.set_page_config(page_title='DentiLens XR', page_icon='🦷', layout='wide')
st.markdown('DENTILENS XR · TEAM RAJASTHANI')
st.title('A clearer view of your dental footage')
st.write('Upload an examination video, inspect AI-generated tooth regions, and revisit selected moments.')
st.caption('Research prototype · Every prediction requires professional review. '
           'Review tooth labels and possible cavity flags against the original video.')

with st.sidebar:
    st.header('Review workspace')
    st.write('Videos and results stay on this computer. Analysis runs locally on CPU.')
    st.caption('Future XR vision: hands-free recording and review through compatible glasses.')
    st.divider()
    st.caption('Vellore Institute of Technology, Vellore')
    st.caption('Arham Choudhary · Aadeesh Sanghi')

upload_tab, review_tab = st.tabs(['Analyze a video', 'Review saved results'])

with upload_tab:
    uploaded = st.file_uploader('Choose a dental examination video', type=['mp4'])
    available = []
    sam_ready = (ROOT / 'models/grounded/sam2/model.safetensors').is_file()
    dino_ready = (ROOT / 'models/grounded/dino/model.safetensors').is_file()
    if sam_ready and dino_ready:
        available.append(('Text-guided tooth regions + SAM', 'grounded-sam2'))
        available.append(('Teeth and dental mirror head (experimental)', 'teeth-mirror'))
    for mode, (checkpoint, label, _) in TRAINED_MODES.items():
        if mode in ('tooth-sam2', 'tooth-types') and not sam_ready:
            continue
        if mode == 'cavity-review' and not dino_ready:
            continue
        if all((ROOT / 'models' / f'{checkpoint}.{extension}').is_file() for extension in ('pt', 'json')):
            available.append((label, mode))
    if not available:
        st.info('Install the local model files using the repository setup instructions to enable analysis.')
    else:
        choice = st.selectbox('Analysis model', available, format_func=lambda value: value[0])
        if choice[1] == 'tooth-sam2':
            st.warning('This experimental detector produced glove and face false positives on the supplied videos. '
                       'Its still-image test score does not establish reliable video segmentation.')
        if choice[1] == 'tooth-fdi':
            st.info('Use footage in standard orientation. Confirm jaw and patient side before saving a tooth number.')
        if choice[1] == 'cavity-review':
            st.info('Flags are for visual review. A frame without a flag does not rule out decay.')
        st.caption('One frame per second is analyzed. CPU processing can take several minutes. '
                   'Highlights use original footage and omit audio.')
        if uploaded is not None:
            st.video(uploaded)
        if st.button('Analyze video', type='primary', disabled=uploaded is None):
            identifier = uuid.uuid4().hex[:12]
            RUNTIME.mkdir(exist_ok=True)
            source = RUNTIME / f'{identifier}.mp4'
            source.write_bytes(uploaded.getbuffer())
            destination = RESULTS / f'upload-{identifier}'
            command = [sys.executable, str(ROOT / 'code/dental_video.py'), str(source),
                       str(destination), '--engine', choice[1], '--device', 'cpu']
            with st.status('Analyzing video…', expanded=True) as status:
                st.write('Running the selected model and saving timestamped evidence.')
                completed = subprocess.run(command, capture_output=True, text=True)
                if completed.returncode:
                    status.update(label='Analysis could not finish', state='error')
                    st.error('Check that the video is a readable MP4 and the model files are installed.')
                    with st.expander('Processing details'):
                        st.code(completed.stderr[-5000:])
                else:
                    (destination / 'source.json').write_text(json.dumps({
                        'path': str(source), 'display_name': uploaded.name}, indent=2))
                    st.session_state['last_result'] = destination.name
                    status.update(label='Analysis ready', state='complete')
                    st.success('Open “Review saved results” to inspect the predictions.')

with review_tab:
    directories = sorted((p for p in RESULTS.glob('*') if not p.name.startswith('.') and (p / 'report.json').is_file()),
                         key=lambda p: p.stat().st_mtime, reverse=True)
    if not directories:
        st.info('Your analyzed videos will appear here.')
    else:
        last = st.session_state.get('last_result')
        default_index = next((i for i, p in enumerate(directories) if p.name == last), None)
        if default_index is None:
            default_index = next((i for i, p in enumerate(directories) if
                json.loads((p / 'report.json').read_text())['model']['name'].startswith('Grounding DINO')), 0)
        selected = st.selectbox('Saved analysis', directories,
            index=default_index,
            format_func=lambda path: path.name)
        report = json.loads((selected / 'report.json').read_text())
        source_info = selected / 'source.json'
        source = None
        display_name = report['video']
        if source_info.exists():
            source_data = json.loads(source_info.read_text())
            source = Path(source_data['path'])
            display_name = source_data.get('display_name', display_name)
        st.subheader(display_name)
        columns = st.columns(3)
        columns[0].metric('Source duration', f"{report['duration_sec']:.1f} s")
        columns[1].metric('Frames analyzed', report['sampled_frames'])
        columns[2].metric('Frames with predictions', len({o['source_frame'] for o in report['observations']}))
        st.caption(f"Model: {report['model']['name']}. Counts include repeated teeth across frames and do not measure accuracy.")
        if report['model']['name'].startswith('DentiLens intraoral'):
            st.warning('Experimental video results. Supplied clips exposed non-tooth false positives; inspect every prediction.')
        left, right = st.columns([1.6, 1])
        review_dir = selected / 'review'
        entries = {}
        for path in sorted(review_dir.glob('*.json')):
            entries[path.stem] = json.loads(path.read_text())
        with left:
            st.subheader('Timestamped visual review')
            lanes = ['Possible cavities', 'Manual bookmarks'] if report.get('capabilities', {}).get(
                'possible_cavity_flags') else ['Teeth', 'Instruments', 'Manual bookmarks']
            lane = st.radio('Timeline', lanes, horizontal=True,
                            key=f'lane-{selected.name}')
            moments = {}
            if lane == 'Manual bookmarks':
                moments = {key: value for key, value in entries.items() if key.startswith('manual-')}
            else:
                for index, observation in enumerate(report['observations']):
                    label = observation['model_label']
                    category = observation.get('category')
                    matches_lane = (
                        (lane == 'Teeth' and 'mirror' not in label and category != 'cavity_candidate')
                        or (lane == 'Instruments' and 'mirror' in label)
                        or (lane == 'Possible cavities' and category == 'cavity_candidate'))
                    if not matches_lane:
                        continue
                    moments[f'model-{index}'] = observation
            timestamp, marker = 0.0, None
            if moments:
                def moment_caption(key):
                    item = moments[key]
                    label = f"FDI {item['fdi_number']}?" if 'fdi_number' in item else item.get(
                        'model_label', 'Manual bookmark')
                    status = entries.get(key, {}).get('decision', 'Unreviewed')
                    return f"{item['timestamp_sec']:.2f} s · {label} · {status}"

                marker = st.selectbox('Jump to a moment', list(moments), format_func=moment_caption,
                                      key=f'moment-{selected.name}-{lane}')
                moment = moments[marker]
                timestamp = moment['timestamp_sec']
                if 'evidence' in moment:
                    st.image(str(selected / moment['evidence']), width='stretch')
                    if 'fdi_number' in moment:
                        st.write(f"Suggested FDI {moment['fdi_number']} · {moment['tooth_name']}")
                    if 'confidence' in moment:
                        st.caption(f"Model score {moment['confidence']:.0%} · verify against the source footage.")
                    if lane == 'Possible cavities':
                        st.caption('Orange boxes mark areas flagged for visual review.')
            else:
                message = ('No possible cavity flags passed the threshold. Review the source footage.'
                           if lane == 'Possible cavities' else 'No moments in this lane yet.')
                st.info(message)
            source_ready = source is not None and source.is_file()
            if source_ready:
                st.video(str(source), start_time=timestamp)

            manual = st.checkbox('Add a bookmark at another time', value=marker is None,
                                 key=f'manual-{selected.name}-{marker}')
            current = entries.get(marker, {}) if not manual else {}
            with st.form(f'review-{selected.name}-{marker}-{manual}'):
                if manual:
                    timestamp = st.number_input('Bookmark time in seconds', min_value=0.0,
                        max_value=float(report['duration_sec']), value=float(timestamp), step=0.1)
                decision = st.selectbox('Review decision', ['Confirm', 'Dismiss'],
                                        index=int(current.get('decision') == 'Dismiss'))
                tooth = st.text_input('Tooth name or FDI number (optional)', value=current.get('tooth', ''),
                                      placeholder='For example, upper right central incisor or 11')
                flag = st.checkbox('Flag unusual appearance for dentist review', value=current.get('flag', False))
                note = st.text_area('Reviewer note', value=current.get('note', ''), max_chars=500)
                if st.form_submit_button('Save review'):
                    review_dir.mkdir(exist_ok=True)
                    key = f'manual-{uuid.uuid4().hex}' if manual else marker
                    record = dict(timestamp_sec=timestamp, decision=decision, tooth=tooth.strip(),
                                  flag=flag, note=note.strip(), entered_by='reviewer')
                    with tempfile.NamedTemporaryFile(mode='w', dir=review_dir, delete=False) as stream:
                        json.dump(record, stream, indent=2)
                    Path(stream.name).replace(review_dir / f'{key}.json')
                    st.rerun()
            st.caption('Saved tooth names, numbers and flags reflect the reviewer’s decisions.')
            if current.get('flag'):
                st.warning('Reviewer flagged this moment for dentist review.')
            with st.expander('Simulated XR preview'):
                st.markdown(f"**{timestamp:.2f} s** · {current.get('tooth') or 'Review bookmark'}")
                if current.get('flag'):
                    st.write('Flagged for dentist review')
                if current.get('note'):
                    st.text(current['note'])
                st.caption('On-screen preview of a future glasses display.')
        with right:
            st.subheader('Review highlights')
            confirmed = sorted((key for key, value in entries.items() if value['decision'] == 'Confirm'),
                               key=lambda key: entries[key]['timestamp_sec'])
            if entries:
                st.download_button('Download reviewer notes',
                    json.dumps({'video': display_name, 'bookmarks': list(entries.values())}, indent=2),
                    file_name='dentilens-reviewed-bookmarks.json', mime='application/json')
            if source_ready and confirmed:
                chosen = st.multiselect('Moments for reviewed reel', confirmed, default=confirmed,
                    format_func=lambda key: f"{entries[key]['timestamp_sec']:.2f} s · "
                                            f"{entries[key]['tooth'] or 'Confirmed moment'}")
                reel_key = (str(selected), tuple((key, entries[key]['timestamp_sec']) for key in chosen))
                if st.button('Build reviewed reel', disabled=not chosen):
                    windows = []
                    duration = report['duration_sec']
                    for time in sorted(entries[key]['timestamp_sec'] for key in chosen):
                        start = min(max(0, time - 1), max(0, duration - 3))
                        end = min(start + 3, duration)
                        if windows and start <= windows[-1]['end_sec']:
                            windows[-1]['end_sec'] = max(windows[-1]['end_sec'], end)
                        else:
                            windows.append(dict(start_sec=start, end_sec=end))
                    try:
                        with tempfile.TemporaryDirectory() as directory:
                            target = Path(directory) / 'reviewed.mp4'
                            export_clips(source, target, windows)
                            st.session_state['reviewed_reel'] = (reel_key, target.read_bytes())
                    except (subprocess.SubprocessError, OSError):
                        st.error('The reviewed reel could not be exported.')
                reel = st.session_state.get('reviewed_reel')
                if reel and reel[0] == reel_key:
                    st.video(reel[1])
                    st.download_button('Download reviewed reel', reel[1],
                                       file_name='dentilens-reviewed.mp4', mime='video/mp4')

            if (selected / 'highlights.mp4').is_file():
                st.video(str(selected / 'highlights.mp4'))
                st.caption('Selected from model observations; not a clinical importance assessment.')
                st.download_button('Download highlight', (selected / 'highlights.mp4').read_bytes(),
                                   file_name='dentilens-highlight.mp4', mime='video/mp4')
            else:
                st.info('No highlight was generated. A highlight requires a prediction and at least '
                        'four seconds within the 15% duration budget.')
            st.download_button('Download analysis report', (selected / 'report.json').read_bytes(),
                               file_name='dentilens-report.json', mime='application/json')
            with st.expander('Capabilities and limitations'):
                for limitation in report['limitations']:
                    st.write('• ' + limitation)

test_record = ROOT / 'reference/intraoral-detector-test.json'
if test_record.is_file():
    with st.expander('Detector training and evaluation'):
        metrics = json.loads(test_record.read_text())
        columns = st.columns(3)
        columns[0].metric('Test images', metrics['images'])
        columns[1].metric('Box mAP50', f"{metrics['mAP50']*100:.1f}%")
        columns[2].metric('Box mAP50–95', f"{metrics['mAP50_95']*100:.1f}%")
        st.write('These metrics cover tooth detection boxes in the supplied still-image test split. '
                 'Patient overlap across splits is unknown. Video checks showed false positives and missed teeth, '
                 'so the trained detector remains experimental. SAM mask accuracy has not been measured.')
        chart = ROOT / 'reference/intraoral-training.png'
        if chart.is_file():
            st.image(str(chart), width='stretch')
