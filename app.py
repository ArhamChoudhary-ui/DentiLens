"""Local dental video upload and review application."""
from collections import deque
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
from annotated_reel import context_windows, export_annotated_reel, review_marks
from review_preview import focused_tooth_preview
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
        available.append(('Tooth outlines only', 'grounded-sam2'))
        available.append(('Teeth and dental mirror head (experimental)', 'teeth-mirror'))
    for mode, (checkpoint, label, _) in TRAINED_MODES.items():
        if mode in ('tooth-sam2', 'tooth-types') and not sam_ready:
            continue
        if mode in ('tooth-fdi', 'cavity-review') and not dino_ready:
            continue
        if all((ROOT / 'models' / f'{checkpoint}.{extension}').is_file() for extension in ('pt', 'json')):
            available.append((label, mode))
    if not available:
        st.info('Install the local model files using the repository setup instructions to enable analysis.')
    else:
        choice = st.selectbox('What should DentiLens look for?', available, index=None,
                              placeholder='Choose an analysis mode', format_func=lambda value: value[0])
        mode = choice[1] if choice else None
        if mode in ('grounded-sam2', 'tooth-sam2'):
            st.caption('This mode draws tooth outlines. Choose Tooth numbers for suggested numbers and names.')
        if mode == 'tooth-sam2':
            st.warning('This experimental detector produced glove and face false positives on the supplied videos. '
                       'Its still-image test score does not establish reliable video segmentation.')
        if mode == 'tooth-fdi':
            st.info('Use footage in standard orientation. Confirm jaw and patient side before saving a tooth number.')
        if mode == 'cavity-review':
            st.info('Flags are for visual review. A frame without a flag does not rule out decay.')
        st.caption('Analysis samples one frame per second. You can export an annotated reel from the review tab.')
        if uploaded is not None:
            st.video(uploaded)
        if st.button('Analyze video', type='primary', disabled=uploaded is None or choice is None):
            identifier = uuid.uuid4().hex[:12]
            RUNTIME.mkdir(exist_ok=True)
            source = RUNTIME / f'{identifier}.mp4'
            source.write_bytes(uploaded.getbuffer())
            destination = RESULTS / f'upload-{identifier}'
            command = [sys.executable, str(ROOT / 'code/dental_video.py'), str(source),
                       str(destination), '--engine', mode, '--device', 'cpu', '--progress']
            with st.status('Analyzing video…', expanded=True) as status:
                progress = st.progress(0.0, text='Loading the selected model…')
                diagnostics = deque(maxlen=25)
                with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                      text=True, errors='replace') as process:
                    for line in process.stdout:
                        parts = line.split()
                        if len(parts) == 3 and parts[0] == 'DENTILENS_PROGRESS':
                            current, total = map(int, parts[1:])
                            message = (f'Analyzed {current} of {total} frames'
                                       if current < total else 'Saving evidence and highlights…')
                            progress.progress(min(0.99, current / total), text=message)
                        else:
                            diagnostics.append(line)
                    exit_code = process.wait()
                if exit_code:
                    progress.empty()
                    status.update(label='Analysis could not finish', state='error')
                    st.error('Analysis stopped. Open processing details to see the error.')
                    with st.expander('Processing details'):
                        st.code(''.join(diagnostics)[-5000:])
                else:
                    (destination / 'source.json').write_text(json.dumps({
                        'path': str(source), 'display_name': uploaded.name}, indent=2))
                    st.session_state['last_result'] = destination.name
                    progress.progress(1.0, text='Analysis complete')
                    status.update(label='Analysis ready', state='complete')
                    st.success(f'{choice[0]} finished. Open “Review saved results” to inspect the output.')

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
        features = report.get('capabilities', {})
        if not any(features.get(name) for name in (
                'individual_tooth_numbering', 'tooth_type_suggestions', 'possible_cavity_flags')):
            st.caption('This analysis has no tooth names or numbers. Run Tooth numbers to generate FDI suggestions.')
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
                    if 'fdi_number' in moment:
                        st.write(f"**Suggested FDI {moment['fdi_number']} · {moment['tooth_name']}**")
                        preview = focused_tooth_preview(source, moment)
                        if preview is not None:
                            st.image(preview, width='stretch', caption='Selected tooth · model suggestion')
                        else:
                            st.image(str(selected / moment['evidence']), width='stretch')
                        with st.expander('All predictions on this frame'):
                            st.image(str(selected / moment['evidence']), width='stretch')
                    else:
                        st.image(str(selected / moment['evidence']), width='stretch')
                        if report.get('capabilities', {}).get('tooth_type_suggestions'):
                            st.write(f"Suggested tooth type · {moment['model_label']}")
                    if 'confidence' in moment:
                        st.caption(f"Model score {moment['confidence']:.0%} · verify against the source footage.")
                    if lane == 'Possible cavities':
                        st.caption('Orange boxes mark areas flagged for visual review.')
            else:
                message = ('No possible cavity flags passed the threshold. Review the source footage.'
                           if lane == 'Possible cavities' else 'No moments in this lane yet.')
                if not report['observations'] and lane != 'Possible cavities':
                    message = 'No predictions passed this model’s threshold. You can still add a manual bookmark.'
                st.info(message)
            source_ready = source is not None and source.is_file()
            if source_ready:
                with st.expander('Original video', expanded=not moments):
                    st.video(str(source), start_time=timestamp)
            else:
                st.info('The source MP4 is unavailable. Saved evidence and notes are still available.')

            manual = marker is None or st.checkbox('Add a bookmark at another time',
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
                    if not manual and 'model_label' in moments[marker]:
                        suggestion = moments[marker]
                        record['model_suggestion'] = {
                            field: suggestion[field] for field in ('model_label', 'fdi_number', 'tooth_name', 'confidence')
                            if field in suggestion}
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
            st.subheader('Annotated review reel')
            if entries:
                st.download_button('Download reviewer notes',
                    json.dumps({'video': display_name, 'bookmarks': list(entries.values())}, indent=2),
                    file_name='dentilens-reviewed-bookmarks.json', mime='application/json')
            confirmed = sorted((key for key, value in entries.items() if value['decision'] == 'Confirm'),
                               key=lambda key: entries[key]['timestamp_sec'])
            if source_ready:
                choices = []
                if marker is not None and entries.get(marker, {}).get('decision') != 'Dismiss':
                    choices.append('Selected moment')
                if confirmed:
                    choices.append('Confirmed moments')
                if report['highlights'] and report['observations']:
                    choices.append('AI highlights')
                if choices:
                    content = st.selectbox('Reel contents', choices, key=f'reel-content-{selected.name}')
                    if content == 'Selected moment':
                        marks = review_marks(report, entries, [marker])
                        windows = context_windows([moment['timestamp_sec']], report['duration_sec'])
                    elif content == 'Confirmed moments':
                        chosen = st.multiselect('Moments to include', confirmed, default=confirmed,
                            format_func=lambda key: f"{entries[key]['timestamp_sec']:.2f} s · "
                                                    f"{entries[key]['tooth'] or 'Confirmed moment'}")
                        marks = review_marks(report, entries, chosen)
                        windows = context_windows([item['timestamp_sec'] for item in marks], report['duration_sec'])
                    else:
                        windows = report['highlights']
                        marks = review_marks(report, entries)
                    reel_key = (selected.name, content, json.dumps([windows, marks], sort_keys=True))
                    st.caption('Playback briefly pauses on each annotated frame. The reel includes source times, '
                               'model labels and saved review decisions. Exported reels are silent.')
                    if st.button('Build annotated reel', disabled=not marks):
                        try:
                            with st.spinner('Rendering annotated video…'), tempfile.TemporaryDirectory() as directory:
                                target = Path(directory) / 'reviewed.mp4'
                                export_annotated_reel(source, target, report, windows, marks, selected)
                                st.session_state['annotated_reel'] = (reel_key, target.read_bytes())
                        except (ValueError, OSError, RuntimeError) as exc:
                            st.error(f'Could not build the reel. {exc}')
                    reel = st.session_state.get('annotated_reel')
                    if reel and reel[0] == reel_key:
                        st.video(reel[1])
                        st.download_button('Download annotated reel', reel[1],
                                           file_name='dentilens-annotated.mp4', mime='video/mp4')
                else:
                    st.info('Select a prediction or save a confirmed bookmark to export a reel.')
            if (selected / 'highlights.mp4').is_file():
                with st.expander('Original AI highlight'):
                    st.video(str(selected / 'highlights.mp4'))
                    st.download_button('Download original highlight', (selected / 'highlights.mp4').read_bytes(),
                                       file_name='dentilens-highlight.mp4', mime='video/mp4')
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
