import { useEffect, useRef, useState } from 'react'
import { Activity, ArrowDownToLine, ArrowRight, CheckCircle2, Clock3, FileJson, Film, LayoutGrid, Settings2, UploadCloud } from 'lucide-react'
import { api } from './api'
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
type Run = {
  id: string; name: string; bytes: number; duration: number; created: string
  status: 'running' | 'validating' | 'ready' | 'failed'; error?: string
  data?: Result; model?: string; label_source?: string; warnings?: string[]; manifest_match?: boolean; highlight_evidence?: HighlightEvidence[]
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
  const selectedSpan = run.data?.highlights[highlight]?.segment
  const selectedEvidence = run.highlight_evidence?.[highlight]
  const data = run.data!
  const actions = [...data.actions].sort((a, b) => a.segment[0] - b.segment[0])
  const laneEnds: number[] = []
  const actionLanes = actions.map(action => {
    const openLane = laneEnds.findIndex(end => end <= action.segment[0])
    const lane = openLane >= 0 ? openLane : laneEnds.length
    laneEnds[lane] = action.segment[1]
    return lane
  })
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
      <video className="eg-player" ref={player} controls playsInline preload="metadata" src={endpoint(run.id)+'/video'}
        onTimeUpdate={()=>{const v=player.current!; setPlayhead(v.currentTime); if(stopAt.current!==null&&v.currentTime>=stopAt.current){const end=stopAt.current;stopAt.current=null;v.pause();v.currentTime=end;setPlayhead(end)}}}/>
      <div className="eg-timeline" aria-label="Video timeline">
        {actions.length > 0 && <div className="eg-track"><span className="eg-track-name">Actions</span><div style={{height: Math.max(13, laneEnds.length * 11 + 2)}}>
          {actions.map(({segment,verb,noun},i)=><button className="eg-action-mark" key={i} style={{...spanStyle(segment), top: 1 + actionLanes[i] * 11, height: 10}} onClick={()=>seek(segment[0])}
            title={`${verb} ${noun} · ${clock(segment[0])}–${clock(segment[1])}`} aria-label={`Seek to ${verb} ${noun} at ${clock(segment[0])}`}/>)}
        </div></div>}
        <div className="eg-track"><span className="eg-track-name">Highlights</span><div>
          {data.highlights.map(({segment},i)=><button className="eg-highlight-mark" key={i} style={spanStyle(segment)} onClick={()=>{setHighlight(i);seek(...segment)}}
            title={`Play highlight ${i+1}`} aria-label={`Play highlight ${i+1}`}/>)}
        </div></div>
        <div className="eg-axis"><span>0:00</span><span>{clock(playhead)} / {clock(data.duration_sec)}</span></div>
        <input className="eg-scrub" type="range" min={0} max={data.duration_sec} step={0.01} value={Math.min(playhead, data.duration_sec)}
          aria-label="Scrub video" onChange={e=>seek(e.currentTarget.valueAsNumber)}/>
      </div>
      <div className="eg-section"><h2>Highlights <span>{data.highlights.length}</span></h2>
        {data.highlights.map(({segment},i)=><button className="eg-row" key={i} onClick={()=>{setHighlight(i);seek(...segment)}}>
          <Film size={16}/><span>Highlight {i+1}</span><time>{clock(segment[0])}–{clock(segment[1])}</time><span>{(segment[1]-segment[0]).toFixed(1)} s</span>
        </button>)}
      </div>
      {selectedSpan && <div className="eg-reason" aria-live="polite">
        <h2>Why This Moment?</h2>
        <small>{clock(selectedSpan[0])}–{clock(selectedSpan[1])} · {selectedEvidence?.fallback ? 'Heuristic fallback' : selectedEvidence ? 'Model-ranked highlight' : 'Ranking details unavailable'}</small>
        <p>{highlightReason(selectedEvidence)}</p>
      </div>}
      <details className="eg-section"><summary><FileJson size={17}/> Output JSON</summary>
        <div className="eg-actions"><button className="eg-button" onClick={async()=>{
          try{const r=await fetch(endpoint(run.id)+'/result');if(!r.ok)throw Error();await navigator.clipboard.writeText(await r.text());setCopied('Copied')}
          catch {setCopied('Copy unavailable')}
        }}>{copied||'Copy JSON'}</button><a className="eg-button" href={endpoint(run.id)+'/result'}><ArrowDownToLine size={15}/> Download JSON</a></div>
        <pre className="eg-json">{JSON.stringify(data,null,2).split(/("(?:[^"\\]|\\.)*"|\b\d+(?:\.\d+)?\b)/g).map((token,i)=><span key={i} className={token.startsWith('"')?'eg-string':/^\d/.test(token)?'eg-number':undefined}>{token}</span>)}</pre>
      </details>
    </div>
    <aside className="eg-evidence">
      <h2>Activity</h2><p className="eg-label">{run.label_source === 'Model prediction' && data.label.name === 'other' && !data.label.score ? 'Activity uncertain' : activityName(data.label.name)}</p>
      <p>{run.label_source === 'Model prediction' ? data.label.name==='other'&&data.label.score===0?'The full-video model abstained.':`${Math.round(data.label.score*100)}% full-video score` : 'Category supplied by the benchmark guide'}</p>
      <p className="eg-model">{run.model}</p>
      <p className="eg-valid"><CheckCircle2 size={16}/> Local validation passed</p>
      <p>Checked with the team-created Track 4 validator.</p>
      {!run.manifest_match && <p>Development clip. This output is outside the Samsung manifest.</p>}
      {run.warnings?.map(w=><p className="eg-warning" key={w}>{w}</p>)}
      <h2>Actions <span>{actions.length}</span></h2>
      {!actions.length ? <p>This model scores activities and selects highlights. Timed verb–noun actions are available on the EPIC-KITCHENS path.</p> :
      <div className="eg-action-list">{actions.map(({segment,verb,noun,score},i)=><button className="eg-row" key={i} onClick={()=>seek(segment[0])}>
        <span>{i+1}</span><strong>{verb} {noun}</strong><time>{clock(segment[0])}–{clock(segment[1])}</time><span>{Math.round(score*100)}%</span>
      </button>)}</div>}
    </aside>
  </div>
}

export function EgoSight() {
  const [view,setView] = useState<View>('Overview')
  const [file,setFile] = useState<File | null>(null)
  const [preview,setPreview] = useState('')
  const [duration,setDuration] = useState<number | null>(null)
  const [error,setError] = useState('')
  const [uploading,setUploading] = useState(false)
  const [run,setRun] = useState<Run>()
  const [history,setHistory] = useState<Run[]>([])
  const [settings,setSettings] = useState<Settings>()
  const [offline,setOffline] = useState(false)
  const input = useRef<HTMLInputElement>(null)
  const active = run?.status==='running'||run?.status==='validating'
  function loadHistory() {
    return api<Run[]>('/api/action/runs').then(setHistory).catch(()=>setOffline(true))
  }
  useEffect(()=>{
    api<Settings>('/api/action/settings').then(setSettings).catch(()=>setOffline(true))
    void loadHistory()
  },[])
  useEffect(()=>{
    setDuration(null)
    if(!file){setPreview('');return}
    const url=URL.createObjectURL(file);setPreview(url)
    return ()=>URL.revokeObjectURL(url)
  },[file])
  useEffect(()=>{
    if(!run||!active)return
    let live=true
    const timer=setInterval(async()=>{
      try{
        const next=await api<Run>(endpoint(run.id))
        if(live){setRun(next);setError('');if(next.status==='ready'||next.status==='failed')void loadHistory()}
      }catch(e){if(live)setError(errorText(e))}
    },900)
    return ()=>{live=false;clearInterval(timer)}
  },[run?.id,active])
  function choose(f?: File) {
    if(!f||active||uploading)return
    if(!/\.mp4$/i.test(f.name)||f.size>500*1024*1024){setError('Choose an MP4 smaller than 500 MB.');return}
    setError('');setFile(f);setRun(undefined);setView('Video Intelligence')
  }
  async function analyze() {
    if(!file||uploading)return
    setUploading(true);setError('')
    const body=new FormData();body.append('video',file)
    try{setRun(await api<Run>('/api/action/runs',{method:'POST',body}));setFile(null)}
    catch(e){setError(errorText(e))}finally{setUploading(false)}
  }
  function openRun(r: Run) {setRun(r);setFile(null);setError('');setView('Video Intelligence')}
  function browse(){setView('Video Intelligence');input.current?.click()}
  return <div className="egosight">
    <nav className="eg-nav" aria-label="EgoSight workspace">
      <button className="eg-brand" onClick={()=>setView('Overview')}><img src="/egosight.svg" alt=""/><span>EgoSight <b>XR</b></span></button>
      <div>{tabs.map(({title,Icon})=><button key={title} aria-current={view===title?'page':undefined} onClick={()=>setView(title)}><Icon size={17}/>{title}</button>)}</div>
      <small>Team Rajasthani · Track 4</small>
    </nav>
    <main className="eg-main">
      <input hidden ref={input} type="file" accept=".mp4,video/mp4" onChange={e=>{choose(e.target.files?.[0]);e.target.value=''}} aria-label="Choose an MP4 video"/>
      {view==='Overview'&&<>
        <section className="eg-hero">
          <div><h1>See the Actions.<br/>Understand the Moment.</h1>
          <p>Review first-person footage by action. Jump to the moment you need, replay highlights and export the model’s predictions.</p>
          <button className="eg-button eg-primary" onClick={browse}><UploadCloud size={18}/> Upload video <ArrowRight size={16}/></button></div>
          <figure><img src="/review-cover.jpg" alt="Illustration of hands setting an oven timer from a first-person perspective"/><figcaption>Concept illustration</figcaption></figure>
        </section>
        <section className="eg-overview"><h2>Recent analyses</h2>
          {history.length ? history.slice(0,4).map(r=><button className="eg-row" key={r.id} onClick={()=>openRun(r)}><Film size={17}/><strong>{r.name}</strong><time>{r.data?.label.name??r.status}</time><ArrowRight size={16}/></button>) : <p>Upload your first MP4 to start a review.</p>}
        </section>
      </>}
      {view==='Video Intelligence'&&<>
        <header className="eg-heading"><div><h1>Video Intelligence</h1>{run&&<p>{run.name}</p>}</div>
        {(file||run)&&<button className="eg-button" disabled={active||uploading} onClick={browse}>Replace video</button>}</header>
        {error&&<p className="eg-warning" role="alert">{error}</p>}
        {!file&&!run&&<div className="eg-drop" onDragOver={e=>e.preventDefault()} onDrop={e=>{e.preventDefault();choose(e.dataTransfer.files[0])}}>
          <UploadCloud size={36}/><h2>Drop an MP4 here</h2><p>MP4 up to 500 MB. At least 4 seconds.</p>
          <button className="eg-button eg-primary" onClick={browse}>Browse files</button>
        </div>}
        {file&&<div className="eg-selected">
          {preview&&<video className="eg-player" controls playsInline src={preview} onLoadedMetadata={e=>setDuration(e.currentTarget.duration)}
            onError={()=>setError('The browser could not preview this MP4. The engine can still try to decode it.')}/>}
          <p>{file.name} · {(file.size/1024**2).toFixed(1)} MB{duration!==null&&` · ${clock(duration)}`}</p>
          <div className="eg-actions"><button className="eg-button eg-primary" disabled={uploading||duration!==null&&duration<4} onClick={()=>void analyze()}>{uploading?'Uploading…':'Analyze video'}</button>
          <button className="eg-button" disabled={uploading} onClick={()=>setFile(null)}>Remove</button></div>
        </div>}
        {run&&active&&<div className="eg-progress" role="status"><Activity size={22}/><h2>{run.status==='validating'?'Validating output':'Analyzing video'}</h2><p>Local inference may take a few minutes.</p></div>}
        {run?.status==='failed'&&<p className="eg-warning" role="alert">{run.error}</p>}
        {run?.status==='ready'&&<Review key={run.id} run={run}/>}
      </>}
      {view==='Analysis History'&&<><h1>Analysis History</h1>{history.length?history.map(r=><button className="eg-row" key={r.id} onClick={()=>openRun(r)}>
        <Film size={17}/><strong>{r.name}</strong><time>{new Date(r.created).toLocaleDateString()}</time><span>{r.data?`${r.data.label.name} · ${r.data.actions.length} actions · ${r.data.highlights.length} highlights`:r.status}</span>
      </button>):<p>No analyses saved yet.</p>}</>}
      {view==='Benchmark & Validation'&&<><h1>Benchmark & Validation</h1>
        <p>Samsung’s manifest lists 138 EPIC-KITCHENS and 49 FPVSum clips. Two unavailable FPVSum clips were excluded, leaving 185 eligible videos.</p>
        {settings?.benchmark && <p><strong>{settings.benchmark.valid} of {settings.benchmark.eligible}</strong> outputs passed the local checker on {settings.benchmark.date}. The staging ZIP includes {settings.benchmark.epic} EPIC-KITCHENS and {settings.benchmark.fpv} FPVSum outputs.</p>}
        <p>EPIC-KITCHENS tests timed verb–noun predictions. FPVSum tests video categories and highlights. Ground-truth actions count toward coverage when at least half their duration falls inside the selected highlights.</p>
        <h2>Supported video labels</h2><p>{settings?.labels.join(', ')||'Connect the local backend to read the vocabulary.'}</p>
        <a href={guide}>Read the model and validation documentation</a></>}
      {view==='Settings'&&<><h1>Settings</h1><p>Local backend · {offline?'Connection unavailable':settings?'Connected':'Checking connection'}</p>
        <h2>Model files</h2>{settings&&Object.entries(settings.models).map(([name,ready])=><p key={name}>{name} · {ready?'Available locally':'Weights missing'}</p>)}
        <a href={guide}>Checkpoint setup</a><p>Team Rajasthani · Vellore Institute of Technology, Vellore</p></>}
    </main>
  </div>
}
