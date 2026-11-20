"""Local dental video upload and review application."""
import json
from pathlib import Path
import subprocess
import sys
import uuid

import streamlit as st

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'
RUNTIME = ROOT / 'runtime'

st.set_page_config(page_title='DentiLens XR', page_icon='🦷', layout='wide')
st.markdown('DENTILENS XR · TEAM RAJASTHANI')
st.title('A clearer view of your dental footage')
st.write('Upload an examination video, inspect AI-generated tooth regions, and revisit selected moments.')
st.caption('Research prototype · Every prediction requires professional review. '
           'No diagnosis, tooth numbering, procedure recognition, or live XR connection.')

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
    if sam_ready and (ROOT / 'models/grounded/dino/model.safetensors').is_file():
        available.append(('Text-guided tooth regions + SAM', 'grounded-sam2'))
    if sam_ready and (ROOT / 'models/tooth-detector.pt').is_file() and (ROOT / 'models/tooth-detector.json').is_file():
        available.append(('Experimental trained tooth detector + SAM', 'tooth-sam2'))
    if not available:
        st.info('Install the local model files using the repository setup instructions to enable analysis.')
    else:
        choice = st.selectbox('Analysis model', available, format_func=lambda value: value[0])
        if choice[1] == 'tooth-sam2':
            st.warning('This experimental detector produced glove and face false positives on the supplied videos. '
                       'Its still-image test score does not establish reliable video segmentation.')
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
                st.write('Locating tooth regions, saving timestamped evidence, and selecting highlights.')
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
        with left:
            st.subheader('Timestamped visual review')
            frames = {}
            for observation in report['observations']:
                frames.setdefault(observation['source_frame'], observation)
            if frames:
                frame = st.select_slider('Review bookmark', options=list(frames),
                    format_func=lambda index: f"{frames[index]['timestamp_sec']:.2f} s")
                observation = frames[frame]
                st.image(str(selected / observation['evidence']), width='stretch')
                st.caption('Green contours are model predictions. Check them against the original video.')
                if source is not None and source.is_file():
                    st.video(str(source), start_time=observation['timestamp_sec'])
            else:
                st.info('No tooth regions passed this model’s threshold. This does not establish that teeth are absent.')
        with right:
            st.subheader('Review highlights')
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
