import { useEffect, useRef, useState } from 'react'
import { Activity, AlertCircle, ArrowDownToLine, ArrowRight, CheckCircle2, Clock3, FileJson, Film, LayoutGrid, Loader2, Settings2, Sparkles, UploadCloud, X } from 'lucide-react'
import { api, uploadWithProgress, UploadProgress } from './api'
import './egosight.css'

const repo = 'https://github.com/ArhamChoudhary-ui/EgoSight-XR'
const guide = repo + '/tree/xr-4-video-intelligence/track4'
type Span = [number, number]
type Result = {
  duration_sec: number
  label: { name: string; score: number }
  actions: { segment: Span; verb: string; noun: string; score: number }[]
  highlights: { segment: Span }[]
}
type HighlightEvidence = { verb?: string; noun?: string; confidence?: number; fallback?: boolean }

export type SceneItem = {
  index: number
  timestamp: number
  timestamp_formatted: string
  caption: string
  context?: string
  thumbnail?: string
  model?: string
}

export type SceneUnderstanding = {
  status: 'ready' | 'running' | 'failed' | 'pending'
  model: string
  device?: string
  frame_count?: number
  scenes: SceneItem[]
  error?: string
}

export type Run = {
  id: string; name: string; bytes: number; duration: number; created: string
  status: 'running' | 'validating' | 'ready' | 'failed'; error?: string
  data?: Result; model?: string; label_source?: string; warnings?: string[]; manifest_match?: boolean; highlight_evidence?: HighlightEvidence[]
  scene_understanding?: SceneUnderstanding
}
type Settings = { models: Record<string, boolean>; labels: string[]; benchmark?: { valid: number; eligible: number; epic: number; fpv: number; date: string } }
type View = 'Overview' | 'Video Intelligence' | 'Analysis History' | 'Benchmark & Validation' | 'Settings'
const tabs = [
  { title: 'Overview', Icon: LayoutGrid }, { title: 'Video Intelligence', Icon: Film },
  { title: 'Analysis History', Icon: Clock3 }, { title: 'Benchmark & Validation', Icon: Activity },
  { title: 'Settings', Icon: Settings2 },
] as const
const endpoint = (id: string) => '/api/action/runs/' + id
const clock = (t: number) => `${Math.floor(t / 60)}:${(t % 60).toFixed(2).padStart(5, '0')}`
const errorText = (error: unknown) => error instanceof Error ? error.message : String(error)

const activityName = (label: string) => ({ rockclimbing: 'Rock climbing', skate: 'Skateboarding', ski: 'Skiing' }[label] ?? label)

function highlightReason(evidence?: HighlightEvidence) {
  if (!evidence) return 'Run this video again to see the saved ranking signals.'
  if (evidence.fallback) return 'The selector used the first four seconds because no full-length candidate was scored.'
  const label = evidence.verb ? [evidence.verb, evidence.noun].filter(Boolean).join(' ') : activityName(evidence.noun ?? '')
  const score = !evidence.confidence ? '' : ` (${Math.round(evidence.confidence * 100)}% window score)`
  if (label && score) return `The ranking used the window prediction “${label}”${score}, alongside the duration and overlap rules.`
  return 'The selector ranked this window using the available scores and applied the duration and overlap rules.'
}

function Review({ run }: { run: Run }) {
  const player = useRef<HTMLVideoElement>(null)
  const stopAt = useRef<number | null>(null)
  const [playhead, setPlayhead] = useState(0)
  const [highlight, setHighlight] = useState(0)
  const [copied, setCopied] = useState('')
  const [scenes, setScenes] = useState<SceneUnderstanding | undefined>(run.scene_understanding)
  const [scenesLoading, setScenesLoading] = useState(false)

  const selectedSpan = run.data?.highlights[highlight]?.segment
  const selectedEvidence = run.highlight_evidence?.[highlight]
  const data = run.data!
  const actions = [...data.actions].sort((a, b) => a.segment[0] - b.segment[0])
  const isRecognizedAction = (verb: string, noun: string) => verb !== 'other' || noun !== 'other'
  const hasRecognizedActions = actions.some(a => isRecognizedAction(a.verb, a.noun))
  const laneEnds: number[] = []
  const actionLanes = actions.map(action => {
    const openLane = laneEnds.findIndex(end => end <= action.segment[0])
    const lane = openLane >= 0 ? openLane : laneEnds.length
    laneEnds[lane] = action.segment[1]
    return lane
  })

  useEffect(() => {
    setScenes(run.scene_understanding)
    if (!run.scene_understanding || run.scene_understanding.status === 'pending') {
      setScenesLoading(true)
      api<SceneUnderstanding>(endpoint(run.id) + '/scenes')
        .then(res => setScenes(res))
        .catch(() => {})
        .finally(() => setScenesLoading(false))
    }
  }, [run.id, run.scene_understanding])

  function seek(start: number, end?: number) {
    if (!player.current) return
    stopAt.current = end ?? null
    player.current.currentTime = start
    if (end !== undefined) void player.current.play().catch(() => {})
  }
  const spanStyle = ([start, end]: Span) => ({
    left: Math.min(100, 100 * start / data.duration_sec) + '%',
    width: Math.max(0, Math.min(100 * (end - start) / data.duration_sec, 100 - 100 * start / data.duration_sec)) + '%',
  })
  return <div className="eg-review">
    <div>
      {/* 1. Video Player */}
      <video className="eg-player" ref={player} controls playsInline preload="metadata" src={endpoint(run.id)+'/video'}
        onTimeUpdate={()=>{const v=player.current!; setPlayhead(v.currentTime); if(stopAt.current!==null&&v.currentTime>=stopAt.current){const end=stopAt.current;stopAt.current=null;v.pause();v.currentTime=end;setPlayhead(end)}}}/>
      
      {/* 2. Proposal / Highlight Timeline */}
      <div className="eg-timeline" aria-label="Video timeline">
        {actions.length > 0 && <div className="eg-track"><span className="eg-track-name">{hasRecognizedActions ? 'Actions' : 'Proposals'}</span><div style={{height: Math.max(13, laneEnds.length * 11 + 2)}}>
          {actions.map(({segment,verb,noun,score},i)=><button className="eg-action-mark" key={i} style={{...spanStyle(segment), top: 1 + actionLanes[i] * 11, height: 10}} onClick={()=>seek(segment[0])}
            title={isRecognizedAction(verb, noun) ? `${verb} ${noun} · ${clock(segment[0])}–${clock(segment[1])}` : `Action proposal ${i+1} (${Math.round(score*100)}%) · ${clock(segment[0])}–${clock(segment[1])}`}
            aria-label={isRecognizedAction(verb, noun) ? `Seek to ${verb} ${noun} at ${clock(segment[0])}` : `Seek to proposal ${i+1} at ${clock(segment[0])}`}/>)}
        </div></div>}
        <div className="eg-track"><span className="eg-track-name">Highlights</span><div>
          {data.highlights.map(({segment},i)=><button className="eg-highlight-mark" key={i} style={spanStyle(segment)} onClick={()=>{setHighlight(i);seek(...segment)}}
            title={`Play highlight ${i+1}`} aria-label={`Play highlight ${i+1}`}/>)}
        </div></div>
        <div className="eg-axis"><span>0:00</span><span>{clock(playhead)} / {clock(data.duration_sec)}</span></div>
        <input className="eg-scrub" type="range" min={0} max={data.duration_sec} step={0.01} value={Math.min(playhead, data.duration_sec)}
          aria-label="Scrub video" onChange={e=>seek(e.currentTarget.valueAsNumber)}/>
      </div>

      {/* 3. AI Scene Understanding */}
      <section className="eg-section eg-scene-understanding" aria-label="AI Scene Understanding">
        <div className="eg-scene-header">
          <div>
            <h2>AI Scene Understanding {scenes?.scenes?.length ? <span>{scenes.scenes.length} moments</span> : null}</h2>
            <p className="eg-scene-subtitle">Explore what is visually happening throughout your video.</p>
          </div>
          <span className="eg-scene-tag" title="Pretrained Salesforce/blip-image-captioning-base visual perception">
            <Sparkles size={13}/> BLIP Vision-Language
          </span>
        </div>

        {scenesLoading && (
          <div className="eg-scene-status" role="status">
            <Loader2 size={16} className="eg-spin"/>
            <span>Examining representative frames from your video...</span>
          </div>
        )}

        {!scenesLoading && scenes?.status === 'failed' && (
          <div className="eg-scene-fallback" role="alert">
            <AlertCircle size={15}/>
            <span>Scene descriptions could not be generated. Your video analysis and highlights are still available.</span>
          </div>
        )}

        {scenes?.scenes && scenes.scenes.length > 0 && (
          <div className="eg-scene-grid">
            {scenes.scenes.map((scene, idx) => {
              const isActive = Math.abs(playhead - scene.timestamp) < 2.5
              return (
                <div
                  key={idx}
                  className={`eg-scene-card ${isActive ? 'eg-active' : ''}`}
                  onClick={() => seek(scene.timestamp)}
                  role="button"
                  tabIndex={0}
                  onKeyDown={e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); seek(scene.timestamp); } }}
                  title={`Jump to ${scene.timestamp_formatted}`}
                >
                  <div className="eg-scene-thumb-wrap">
                    {scene.thumbnail ? (
                      <img src={scene.thumbnail} alt={`Visual frame at ${scene.timestamp_formatted}`} loading="lazy" />
                    ) : (
                      <div className="eg-scene-thumb-placeholder"><Film size={18}/></div>
                    )}
                    <span className="eg-scene-time">{scene.timestamp_formatted}</span>
                    {scene.context && <span className="eg-scene-badge">{scene.context}</span>}
                  </div>
                  <div className="eg-scene-body">
                    <p className="eg-scene-caption">“{scene.caption}”</p>
                  </div>
                </div>
              )
            })}
          </div>
        )}
      </section>

      {/* 4. Highlights */}
      <div className="eg-section"><h2>Highlights <span>{data.highlights.length}</span></h2>
        {data.highlights.map(({segment},i)=><button className="eg-row" key={i} onClick={()=>{setHighlight(i);seek(...segment)}}>
          <Film size={16}/><span>Highlight {i+1}</span><time>{clock(segment[0])}–{clock(segment[1])}</time><span>{(segment[1]-segment[0]).toFixed(1)} s</span>
        </button>)}
      </div>

      {/* 5. Why This Moment Explanation */}
      {selectedSpan && <div className="eg-reason" aria-live="polite">
        <h2>Why This Moment?</h2>
        <small>{clock(selectedSpan[0])}–{clock(selectedSpan[1])} · {selectedEvidence?.fallback ? 'Heuristic fallback' : selectedEvidence ? 'Model-ranked highlight' : 'Ranking details unavailable'}</small>
        <p>{highlightReason(selectedEvidence)}</p>
      </div>}

      {/* 6. Output JSON */}
      <details className="eg-section"><summary><FileJson size={17}/> Output JSON</summary>
        <div className="eg-actions"><button className="eg-button" onClick={async()=>{
          try{const r=await fetch(endpoint(run.id)+'/result');if(!r.ok)throw Error();await navigator.clipboard.writeText(await r.text());setCopied('Copied')}
          catch {setCopied('Copy unavailable')}
        }}>{copied||'Copy JSON'}</button><a className="eg-button" href={endpoint(run.id)+'/result'}><ArrowDownToLine size={15}/> Download JSON</a></div>
        <pre className="eg-json">{JSON.stringify(data,null,2).split(/("(?:[^"\\]|\\.)*"|\b\d+(?:\.\d+)?\b)/g).map((token,i)=><span key={i} className={token.startsWith('"')?'eg-string':/^\d/.test(token)?'eg-number':undefined}>{token}</span>)}</pre>
      </details>
    </div>

    {/* Sidebar Evidence */}
    <aside className="eg-evidence">
      <h2>Activity</h2><p className="eg-label">{run.label_source === 'Model prediction' && data.label.name === 'other' && !data.label.score ? 'Activity uncertain' : activityName(data.label.name)}</p>
      <p>{run.label_source === 'Model prediction' ? data.label.name==='other'&&data.label.score===0?'The full-video model abstained.':`${Math.round(data.label.score*100)}% full-video score` : 'Category supplied by the benchmark guide'}</p>
      <p className="eg-model">{run.model}</p>
      <p className="eg-valid"><CheckCircle2 size={16}/> Local validation passed</p>
      <p>Checked with the team-created Track 4 validator.</p>
      {!run.manifest_match && <p>Development clip. This output is outside the Samsung manifest.</p>}
      {run.warnings?.map(w=><p className="eg-warning" key={w}>{w}</p>)}
      <h2>{hasRecognizedActions ? 'Recognized Actions' : 'Action Proposals'} <span>{actions.length}</span></h2>
      {!actions.length ? <p>This model scores activities and selects highlights. Timed verb–noun actions are available on the EPIC-KITCHENS path.</p> :
      <>
        {!hasRecognizedActions && <p style={{fontSize: '.78rem', color: 'var(--eg-muted)', margin: '4px 0 10px'}}>
          Localized temporal action intervals proposed by the R3D-18 backbone transfer. Individual verb–noun interaction heads are not active on this benchmark.
        </p>}
        <div className="eg-action-list">{actions.map(({segment,verb,noun,score},i)=><button className="eg-row" key={i} onClick={()=>seek(segment[0])}>
          <span>{i+1}</span><strong>{isRecognizedAction(verb, noun) ? `${verb} ${noun}` : `Action Proposal ${i+1}`}</strong><time>{clock(segment[0])}–{clock(segment[1])}</time><span>{Math.round(score*100)}%</span>
        </button>)}</div>
      </>}
    </aside>
  </div>
}

export function EgoSight() {
  const [view, setView] = useState<View>('Overview')
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState('')
  const [duration, setDuration] = useState<number | null>(null)
  const [error, setError] = useState('')
  const [uploading, setUploading] = useState(false)
  const [uploadProgress, setUploadProgress] = useState<UploadProgress>({ percent: 0, loaded: 0, total: 0 })
  const [uploadStatus, setUploadStatus] = useState('Preparing video')
  const [run, setRun] = useState<Run>()
  const [history, setHistory] = useState<Run[]>([])
  const [settings, setSettings] = useState<Settings>()
  const [offline, setOffline] = useState(false)

  const input = useRef<HTMLInputElement>(null)
  const abortControllerRef = useRef<AbortController | null>(null)
  const uploadingRef = useRef(false)
  const active = run?.status === 'running' || run?.status === 'validating'

  function loadHistory() {
    return api<Run[]>('/api/action/runs').then(setHistory).catch(() => setOffline(true))
  }

  useEffect(() => {
    api<Settings>('/api/action/settings').then(setSettings).catch(() => setOffline(true))
    void loadHistory()

    const parseHash = () => {
      const match = window.location.hash.match(/run=([a-f0-9]+)/i)
      if (match) {
        api<Run>(endpoint(match[1]))
          .then(openRun)
          .catch(() => {})
      }
    }
    parseHash()
    window.addEventListener('hashchange', parseHash)
    return () => window.removeEventListener('hashchange', parseHash)
  }, [])

  useEffect(() => {
    setDuration(null)
    if (!file) {
      setPreview('')
      return
    }
    const url = URL.createObjectURL(file)
    setPreview(url)
    return () => URL.revokeObjectURL(url)
  }, [file])

  useEffect(() => {
    if (!run || !active) return
    let live = true
    const timer = setInterval(async () => {
      try {
        const next = await api<Run>(endpoint(run.id))
        if (live) {
          setRun(next)
          setError('')
          if (next.status === 'ready' || next.status === 'failed') void loadHistory()
        }
      } catch (e) {
        if (live) setError(errorText(e))
      }
    }, 900)
    return () => {
      live = false
      clearInterval(timer)
    }
  }, [run?.id, active])

  function choose(f?: File) {
    if (!f || active || uploading) return
    if (!/\.mp4$/i.test(f.name) || f.size > 500 * 1024 * 1024) {
      setError('Choose an MP4 smaller than 500 MB.')
      return
    }
    setError('')
    setFile(f)
    setRun(undefined)
    setView('Video Intelligence')
    void startUpload(f)
  }

  async function startUpload(f: File) {
    if (uploadingRef.current) return
    uploadingRef.current = true
    setUploading(true)
    setError('')
    setUploadProgress({ percent: 0, loaded: 0, total: f.size })
    setUploadStatus('Preparing video')

    const controller = new AbortController()
    abortControllerRef.current = controller

    const body = new FormData()
    body.append('video', f)

    try {
      setUploadStatus('Uploading video')
      const newRun = await uploadWithProgress<Run>(
        '/api/action/runs',
        body,
        p => {
          setUploadProgress(p)
          if (p.percent < 100) {
            setUploadStatus(`Uploading video (${p.percent}%)`)
          } else {
            setUploadStatus('Upload complete · Starting AI analysis')
          }
        },
        controller.signal
      )
      setRun(newRun)
      setUploadStatus('Analysis started')
      void loadHistory()
    } catch (err) {
      if (err instanceof DOMException && err.name === 'AbortError') {
        setError('Upload cancelled.')
      } else {
        setError(errorText(err))
      }
      setFile(null)
    } finally {
      setUploading(false)
      uploadingRef.current = false
      abortControllerRef.current = null
    }
  }

  function cancelUpload() {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
    }
    setUploading(false)
    uploadingRef.current = false
    setFile(null)
    setUploadProgress({ percent: 0, loaded: 0, total: 0 })
  }

  function openRun(r: Run) {
    setRun(r)
    setFile(null)
    setError('')
    setView('Video Intelligence')
    try {
      if (window.location.hash !== '#run=' + r.id) {
        window.location.hash = 'run=' + r.id
      }
    } catch {}
  }

  function browse() {
    setView('Video Intelligence')
    input.current?.click()
  }

  return <div className="egosight">
    <nav className="eg-nav" aria-label="EgoSight workspace">
      <button className="eg-brand" onClick={() => setView('Overview')}>
        <img src="/egosight.svg" alt=""/>
        <span>EgoSight <b>XR</b></span>
      </button>
      <div>{tabs.map(({ title, Icon }) => <button key={title} aria-current={view === title ? 'page' : undefined} onClick={() => setView(title)}><Icon size={17}/>{title}</button>)}</div>
      <small>Team Rajasthani · Track 4</small>
    </nav>
    <main className="eg-main">
      <input hidden ref={input} type="file" accept=".mp4,video/mp4" onChange={e => { choose(e.target.files?.[0]); e.target.value = '' }} aria-label="Choose an MP4 video"/>

      {view === 'Overview' && <>
        <section className="eg-hero">
          <div>
            <h1>See the Actions.<br/>Understand the Moment.</h1>
            <p>Review first-person footage by action. Jump to the moment you need, replay highlights, explore AI scene descriptions, and export the model’s predictions.</p>
            <button className="eg-button eg-primary" onClick={browse}><UploadCloud size={18}/> Upload video <ArrowRight size={16}/></button>
          </div>
          <figure><img src="/review-cover.jpg" alt="Illustration of hands setting an oven timer from a first-person perspective"/><figcaption>Concept illustration</figcaption></figure>
        </section>
        <section className="eg-overview">
          <h2>Recent analyses</h2>
          {history.length ? history.slice(0, 4).map(r => <button className="eg-row" key={r.id} onClick={() => openRun(r)}>
            <Film size={17}/><strong>{r.name}</strong><time>{r.data?.label.name ?? r.status}</time><ArrowRight size={16}/>
          </button>) : <p>Upload your first MP4 to start a review.</p>}
        </section>
      </>}

      {view === 'Video Intelligence' && <>
        <header className="eg-heading">
          <div>
            <h1>Video Intelligence</h1>
            {run && <p>{run.name}</p>}
          </div>
          {(file || run) && <button className="eg-button" disabled={active || uploading} onClick={browse}>Replace video</button>}
        </header>

        {error && <p className="eg-warning" role="alert">{error}</p>}

        {/* Real-time upload progress card */}
        {uploading && (
          <div className="eg-upload-card" role="status">
            <div className="eg-upload-header">
              <div className="eg-upload-title">
                <Film size={20}/>
                <strong>{file?.name || 'Selected video'}</strong>
              </div>
              <span className="eg-upload-meta">
                {file ? `${(file.size / (1024 * 1024)).toFixed(1)} MB` : ''}
              </span>
            </div>
            <div className="eg-progress-track">
              <div className="eg-progress-bar" style={{ width: `${uploadProgress.percent}%` }} />
            </div>
            <div className="eg-progress-footer">
              <span className="eg-status-badge">
                <Loader2 size={13} className="eg-spin"/> {uploadStatus}
              </span>
              <span>
                {(uploadProgress.loaded / (1024 * 1024)).toFixed(1)} MB of {(uploadProgress.total / (1024 * 1024)).toFixed(1)} MB ({uploadProgress.percent}%)
              </span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: 4 }}>
              <button className="eg-cancel-button" onClick={cancelUpload}>Cancel upload</button>
            </div>
          </div>
        )}

        {/* Drop zone when idle */}
        {!file && !run && !uploading && (
          <div className="eg-drop" onDragOver={e => e.preventDefault()} onDrop={e => { e.preventDefault(); choose(e.dataTransfer.files[0]) }}>
            <UploadCloud size={36}/>
            <h2>Drop an MP4 here</h2>
            <p>MP4 up to 500 MB. At least 4 seconds.</p>
            <button className="eg-button eg-primary" onClick={browse}>Browse files</button>
          </div>
        )}

        {/* Active Analysis Processing Screen */}
        {run && active && (
          <div className="eg-processing-card" role="status">
            {preview && (
              <video className="eg-player" controls playsInline muted src={preview} />
            )}
            <div className="eg-processing-box">
              <div className="eg-processing-header">
                <Activity size={24} className="eg-spin" style={{ color: 'var(--eg-accent)' }} />
                <div>
                  <h2>Analyzing Video with EgoSight XR</h2>
                  <p>{run.name} · {duration !== null ? `${clock(duration)} · ` : ''}{(run.bytes / (1024 * 1024)).toFixed(1)} MB</p>
                </div>
              </div>
              <div className="eg-progress-track">
                <div className="eg-progress-bar eg-indeterminate" />
              </div>
              <div className="eg-stepper">
                <div className="eg-step-item eg-done">
                  <CheckCircle2 size={15}/> <span>Upload & Video Decoding completed</span>
                </div>
                <div className={`eg-step-item ${run.status === 'running' ? 'eg-current' : 'eg-done'}`}>
                  {run.status === 'running' ? <Loader2 size={15} className="eg-spin"/> : <CheckCircle2 size={15}/>}
                  <span>Spatiotemporal Feature Extraction (R3D-18 / 3D CNN)</span>
                </div>
                <div className={`eg-step-item ${run.status === 'validating' ? 'eg-current' : ''}`}>
                  {run.status === 'validating' ? <Loader2 size={15} className="eg-spin"/> : <Clock3 size={15}/>}
                  <span>Benchmark Contract Validation & Highlight Candidate Scoring</span>
                </div>
                <div className="eg-step-item">
                  <Sparkles size={15}/> <span>BLIP Visual Scene Understanding</span>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Failed Run Screen */}
        {run?.status === 'failed' && (
          <div className="eg-upload-card" style={{ borderColor: '#683a3a', background: '#221515' }} role="alert">
            <div className="eg-upload-header">
              <div className="eg-upload-title" style={{ color: '#ffb3b3' }}>
                <AlertCircle size={20}/>
                <strong>Analysis Encountered an Error</strong>
              </div>
            </div>
            <p style={{ margin: 0, color: 'var(--eg-muted)', fontSize: '0.88rem' }}>{run.error || 'The model could not process this video.'}</p>
            <div style={{ display: 'flex', gap: 10, marginTop: 8 }}>
              <button className="eg-button eg-primary" onClick={browse}>Try another video</button>
            </div>
          </div>
        )}

        {/* Ready Run Review */}
        {run?.status === 'ready' && <Review key={run.id} run={run} />}
      </>}

      {view === 'Analysis History' && <>
        <h1>Analysis History</h1>
        {history.length ? history.map(r => <button className="eg-row" key={r.id} onClick={() => openRun(r)}>
          <Film size={17}/><strong>{r.name}</strong><time>{new Date(r.created).toLocaleDateString()}</time><span>{r.data ? `${r.data.label.name} · ${r.data.actions.length} ${r.data.actions.some(a => a.verb !== 'other' || a.noun !== 'other') ? 'actions' : 'proposals'} · ${r.data.highlights.length} highlights` : r.status}</span>
        </button>) : <p>No analyses saved yet.</p>}
      </>}

      {view === 'Benchmark & Validation' && <>
        <h1>Benchmark & Validation</h1>
        <p>Samsung’s manifest lists 138 EPIC-KITCHENS and 49 FPVSum clips. Two unavailable FPVSum clips were excluded, leaving 185 eligible videos.</p>
        {settings?.benchmark && <p><strong>{settings.benchmark.valid} of {settings.benchmark.eligible}</strong> outputs passed the local checker on {settings.benchmark.date}. The staging ZIP includes {settings.benchmark.epic} EPIC-KITCHENS and {settings.benchmark.fpv} FPVSum outputs.</p>}
        <p>EPIC-KITCHENS tests timed verb–noun predictions. FPVSum tests video categories and highlights. Ground-truth actions count toward coverage when at least half their duration falls inside the selected highlights.</p>
        <h2>Supported video labels</h2>
        <p>{settings?.labels.join(', ') || 'Connect the local backend to read the vocabulary.'}</p>
        <a href={guide}>Read the model and validation documentation</a>
      </>}

      {view === 'Settings' && <>
        <h1>Settings</h1>
        <p>Local backend · {offline ? 'Connection unavailable' : settings ? 'Connected' : 'Checking connection'}</p>
        <h2>Model files</h2>
        {settings && Object.entries(settings.models).map(([name, ready]) => <p key={name}>{name} · {ready ? 'Available locally' : 'Weights missing'}</p>)}
        <p>BLIP Scene Understanding · Available (Salesforce/blip-image-captioning-base)</p>
        <a href={guide}>Checkpoint setup</a>
        <p>Team Rajasthani · Vellore Institute of Technology, Vellore</p>
      </>}
    </main>
  </div>
}
