import { useEffect, useMemo, useRef, useState } from 'react'
import type { FormEvent } from 'react'
import { createRoot } from 'react-dom/client'
import {
  Activity, ArrowDownToLine, ArrowLeft, ArrowUpRight, Bookmark, Check, ChevronRight,
  CircleAlert, Clapperboard, Crosshair, FileText, Film, Flag, Glasses,
  Layers3, LoaderCircle, Mic, Play, Plus, ScanLine, Search, Upload, Video, X,
} from 'lucide-react'
import { Bar, BarChart, ResponsiveContainer, Tooltip } from 'recharts'
import { api, clock, send } from './api'
import type { Exam, Finding, MediaInfo, Report, Review } from './api'
import './styles.css'

const profiles = [
  { id: 'fast', name: 'Fast', detail: 'Detect every 2 s' },
  { id: 'balanced', name: 'Balanced', detail: 'Detect every 1 s' },
  { id: 'detailed', name: 'Detailed', detail: 'Detect every 0.5 s' },
]

function sizeText(bytes: number) { return `${(bytes / 1024 / 1024).toFixed(1)} MB` }
function path(id: string, endpoint = '') { return `/api/exams/${id}${endpoint}` }
function media(id: string, name: string) { return path(id, `/media/${name}`) }
function message(e: unknown) { return e instanceof Error ? e.message : 'Something went wrong.' }

function UploadScreen({ back, submit }: { back: () => void; submit: (exam: Exam) => void }) {
  const input = useRef<HTMLInputElement>(null)
  const [file, setFile] = useState<File | null>(null)
  const [url, setUrl] = useState('')
  const [meta, setMeta] = useState<{ duration: number; width: number; height: number }>()
  const [profile, setProfile] = useState('balanced')
  const [device, setDevice] = useState('auto')
  const [cavities, setCavities] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [ready, setReady] = useState(true)
  useEffect(() => {
    api<{models_ready: boolean}>('/api/health').then(data => setReady(data.models_ready)).catch(e => setError(message(e)))
  }, [])
  useEffect(() => {
    setMeta(undefined)
    if (!file) { setUrl(''); return }
    const objectUrl = URL.createObjectURL(file)
    setUrl(objectUrl)
    return () => URL.revokeObjectURL(objectUrl)
  }, [file])
  async function upload(e: FormEvent) {
    e.preventDefault()
    if (!file || busy) return
    setError('')
    if (file.size > 500 * 1024 * 1024) { setError('Choose a recording under 500 MB.'); return }
    setBusy(true)
    const body = new FormData()
    body.append('video', file)
    body.append('profile', profile)
    body.append('device', device)
    body.append('cavities', String(cavities))
    try { submit(await api<Exam>('/api/exams', { method: 'POST', body })) }
    catch (e) { setError(message(e)) }
    finally { setBusy(false) }
  }
  return <form onSubmit={upload} className="upload-page">
    <button type="button" className="subtle back" onClick={back}><ArrowLeft size={16}/> Workspace</button>
    <div className="upload-heading">
      <div className="eyebrow"><ScanLine size={15}/> EXAMINATION INTAKE</div>
      <h1>New examination</h1>
      <p>Upload dental footage to create a full annotated recording and a review timeline.</p>
    </div>
    <div className="upload-grid">
      <div className="panel upload-media">
        <input ref={input} type="file" accept=".mp4,.mov,.m4v,.webm" aria-label="Choose examination video"
          className="sr-only" onChange={e => setFile(e.target.files?.[0] ?? null)} />
        {file ? <>
          <video key={url} src={url} controls playsInline onLoadedMetadata={e => {
            const v=e.currentTarget; setMeta({duration: v.duration, width: v.videoWidth, height: v.videoHeight})
          }}/>
          <div className="file-row"><Film size={19}/><div><strong>{file.name}</strong><small>
            {[sizeText(file.size), meta && clock(meta.duration), meta && `${meta.width} × ${meta.height}`].filter(Boolean).join(' · ')}
          </small></div><button type="button" className="icon-button" aria-label="Remove video" onClick={() => { setFile(null); if (input.current) input.current.value='' }}><X size={18}/></button></div>
        </> : <button type="button" className="drop" onClick={() => input.current?.click()}
          onDragOver={e => e.preventDefault()} onDrop={e => { e.preventDefault(); setFile(e.dataTransfer.files[0] ?? null) }}>
          <span className="upload-icon"><Upload size={25}/></span><strong>Choose a dental video</strong>
          <span>Drop an MP4, MOV, M4V or WebM here</span><small>Up to 500 MB or 10 minutes</small>
        </button>}
      </div>
      <section className="panel settings">
        <h2>Analysis settings</h2>
        <p>Every setting exports the entire recording. More frequent detection takes longer.</p>
        <div className="settings-options" role="group" aria-label="Analysis frequency">
          {profiles.map(item => <button type="button" key={item.id} className={profile===item.id?'active':''}
            aria-pressed={profile===item.id} onClick={() => setProfile(item.id)}><strong>{item.name}</strong><small>{item.detail}</small></button>)}
        </div>
        <label className="field">Compute device<select value={device} onChange={e => setDevice(e.target.value)}>
          <option value="auto">Automatic · Apple GPU when available</option>
          <option value="cpu">CPU</option><option value="mps">Prefer Apple GPU · CPU fallback</option>
        </select></label>
        <label className="toggle"><input type="checkbox" checked={cavities} onChange={e => setCavities(e.target.checked)}/>
          <span>Include possible cavity flags<small>Experimental research mode</small></span></label>
        {!ready && <div className="error">Local detector files are missing. See the repository setup instructions.</div>}
        {error && <div role="alert" className="error">{error}</div>}
        <button className="primary wide" disabled={!file || busy || !ready}>
          {busy ? <LoaderCircle className="spin" size={17}/> : <ScanLine size={17}/>} {busy ? 'Uploading…' : 'Analyze full recording'}
        </button>
      </section>
    </div>
  </form>
}

function MediaMetrics({ info, bytes }: { info?: MediaInfo | null; bytes: number }) {
  if (!info) return <span>{sizeText(bytes)}</span>
  return <div className="metadata"><span><Film size={14}/>{clock(info.duration)}</span><span>{info.width} × {info.height}</span>
    <span>{Number(info.fps.toFixed(2))} fps</span><span>{sizeText(bytes)} upload</span></div>
}

function ActivityChart({ report }: { report: Report }) {
  const total = report.video.duration
  const bins = Array.from({length: 24}, (_, i) => ({time: clock(total * i / 24), count: 0}))
  report.findings.forEach(item => bins[Math.min(23, Math.floor(item.time / total * 24))].count++)
  return <div className="chart" aria-label="Suggested bookmark distribution">
    <ResponsiveContainer width="100%" height={54}><BarChart data={bins} barCategoryGap="18%">
      <Tooltip labelFormatter={(_, items) => items[0]?.payload.time} formatter={(n) => [n, 'Suggestions']}
        cursor={{fill:'#263740'}} contentStyle={{background:'#17262e',border:'1px solid #354752',borderRadius:8,fontSize:12}}/>
      <Bar dataKey="count" fill="#79d8af" radius={[2,2,0,0]} minPointSize={2}/>
    </BarChart></ResponsiveContainer>
  </div>
}

type LocalSpeech = {
  processLocally?: boolean
  lang: string
  onresult: ((event: { results: ArrayLike<ArrayLike<{ transcript: string }>> }) => void) | null
  onerror: (() => void) | null
  onend: (() => void) | null
  start: () => void
}

function XR({ exam, refresh }: { exam: Exam; refresh: () => Promise<void> }) {
  const player = useRef<HTMLVideoElement>(null)
  const [command, setCommand] = useState('')
  const [notice, setNotice] = useState('')
  const [listening, setListening] = useState(false)
  const [time, setTime] = useState(0)
  async function execute(text: string) {
    const phrase = text.toLowerCase().trim().replace(/[.!?]+$/, '')
    const video=player.current
    if (!video) return
    try {
      if (phrase === 'bookmark this' || phrase === 'bookmark') {
        const timestamp=Math.min(video.currentTime, Math.max(0, (exam.report?.video.duration ?? 0) - .001))
        await send(path(exam.id,'/reviews'), {time:timestamp, label:'XR bookmark',decision:'confirmed'})
        await refresh()
        setNotice(`Bookmark saved at ${clock(timestamp)}.`)
      } else if (phrase === 'pause') { video.pause(); setNotice('Paused.') }
      else if (phrase === 'play') { await video.play(); setNotice('Playing.') }
      else { setNotice('Try “bookmark this”, “pause” or “play”.') }
    } catch(e) { setNotice(message(e)) }
  }
  function listen() {
    const w = window as unknown as {SpeechRecognition?: new()=>LocalSpeech; webkitSpeechRecognition?: new()=>LocalSpeech}
    const Recognition = w.SpeechRecognition ?? w.webkitSpeechRecognition
    if (!Recognition) { setNotice('This browser has no local speech recognition. Type the command instead.'); return }
    const recognition = new Recognition()
    if (!('processLocally' in recognition)) {
      setNotice('Local speech recognition is unavailable in this browser. Use the typed command.'); return
    }
    recognition.processLocally = true
    recognition.lang = 'en-US'
    recognition.onresult = e => { const heard=e.results[0][0].transcript; setCommand(heard); void execute(heard) }
    recognition.onerror = () => setNotice('Local speech recognition failed. Type the command or use Bookmark this.')
    recognition.onend = () => setListening(false)
    try { recognition.start(); setListening(true); setNotice('Listening locally…') }
    catch (e) { setNotice(message(e)) }
  }
  return <div className="xr-page panel">
    <div className="section-heading"><div><h2>Simulated glasses view</h2><p>Try hands-free bookmarking on the uploaded recording.</p></div><Glasses size={25}/></div>
    <div className="xr-video"><video ref={player} controls src={media(exam.id,'source.mp4')} playsInline
      onTimeUpdate={e=>setTime(e.currentTarget.currentTime)}/><Crosshair className="reticle" size={39}/><span className="xr-time">{clock(time)}</span></div>
    <form className="command" onSubmit={e => {e.preventDefault(); void execute(command)}}>
      <input value={command} onChange={e=>setCommand(e.target.value)} aria-label="Simulated voice command" placeholder='Type “bookmark this”'/>
      <button className="subtle"><Play size={16}/> Run command</button>
      <button type="button" className="subtle" onClick={listen} disabled={listening}><Mic size={16}/> {listening?'Listening':'Microphone'}</button>
      <button type="button" className="primary" onClick={()=>void execute('bookmark this')}><Bookmark size={16}/> Bookmark this</button>
    </form>
    {notice && <p role="status" className="notice">{notice}</p>}
  </div>
}

function PatientView({ exam, moments, reel }: { exam: Exam; moments: Review[]; reel: string }) {
  const player = useRef<HTMLVideoElement>(null)
  return <section className="panel patient-page">
    <div className="section-heading"><div><h2>Patient explanation</h2><p>Play the recording and jump to moments confirmed by the reviewer.</p></div><Play size={21}/></div>
    <video ref={player} src={media(exam.id,'source.mp4')} controls playsInline preload="metadata"/>
    {moments.length ? <div className="patient-moments">
      {moments.slice().sort((a,b)=>a.time-b.time).map(item=><button key={item.id} onClick={()=>{
        if (player.current) { player.current.currentTime=item.time; player.current.pause() }
      }}>
        <Play size={15}/><time>{clock(item.time)}</time><span><strong>{item.label || 'Confirmed moment'}</strong>{item.note && <small>{item.note}</small>}</span>
      </button>)}
    </div> : <p className="quiet">Confirm a bookmark in Review video to show it here.</p>}
    {reel && <a className="subtle" href={reel+(reel.includes('?')?'&':'?')+'download=true'}><ArrowDownToLine size={16}/> Download reviewed reel</a>}
  </section>
}

function ExamReview({ exam, refresh }: { exam: Exam; refresh: () => Promise<void> }) {
  const report=exam.report!
  const player=useRef<HTMLVideoElement>(null)
  const pending=useRef<number | null>(null)
  const [playerMode,setPlayerMode]=useState<'annotated' | 'source'>('annotated')
  const [time,setTime]=useState(0)
  const [selected,setSelected]=useState<Finding | null>(report.findings[0] ?? null)
  const [query,setQuery]=useState('')
  const [label,setLabel]=useState(selected?.label ?? '')
  const [note,setNote]=useState('')
  const [flag,setFlag]=useState(false)
  const [feedback,setFeedback]=useState('')
  const [reel,setReel]=useState(exam.reel_url ?? '')
  const [building,setBuilding]=useState(false)
  const [saving,setSaving]=useState(false)
  const [picked,setPicked]=useState<string[] | null>(null)
  const [tab,setTab]=useState<'review' | 'patient' | 'xr'>('review')
  const reviews=exam.reviews ?? []
  const confirmed=reviews.filter(item=>item.decision==='confirmed')
  const choice=(picked ?? confirmed.map(item=>item.id)).filter(id=>confirmed.some(item=>item.id===id))
  useEffect(()=>{
    if (!selected) return
    const saved=reviews.find(item=>item.id===selected.id)
    setLabel(saved?.label ?? selected.label)
    setNote(saved?.note ?? '')
    setFlag(saved?.flagged ?? false)
  },[selected, exam.reviews])
  function seek(seconds: number) {
    const at=Math.min(Math.max(0,seconds),Math.max(0,report.video.duration-.001))
    const v=player.current
    if (v && v.readyState >= 1) { v.currentTime=at; v.pause() }
    else pending.current=at
    setTime(at)
  }
  function chooseFinding(item: Finding) { setSelected(item); seek(item.time); setFeedback('') }
  function manual() {
    setSelected(null); setLabel(''); setNote(''); setFlag(false); setFeedback('')
  }
  async function save(decision: Review['decision']) {
    if (saving) return
    setSaving(true); setFeedback('')
    try {
      await send(path(exam.id,'/reviews'), {time: Math.min(time,report.video.duration-.001),
        finding_id:selected?.id ?? null,label:label.trim(),note:note.trim(),flagged:flag,decision})
      await refresh()
      setFeedback(decision==='confirmed'?'Bookmark saved.':'Suggestion dismissed.')
    } catch(e) { setFeedback(message(e)) }
    finally { setSaving(false) }
  }
  async function buildReel() {
    setBuilding(true);setFeedback('')
    try {
      const result=await send<{url:string}>(path(exam.id,'/reel'),{bookmarks:choice})
      setReel(`${result.url}?v=${Date.now()}`)
    } catch (e) { setFeedback(message(e)) }
    finally { setBuilding(false) }
  }
  const tracks=useMemo(()=>{
    const q=query.toLowerCase().trim()
    return report.tracks.filter(t=>!q || [t.label,t.id,t.fdi_number,t.tooth_type].some(v=>v?.toLowerCase().includes(q)))
  },[query,report.tracks])
  const coverage=Math.round(100 * report.stats.annotated_frames / report.video.frames)
  return <>
    <header className="exam-heading">
      <div><div className="eyebrow"><span className="status-dot"/> ANALYSIS READY</div>
        <h1>Examination review</h1><div className="record-name">{exam.title}</div><MediaMetrics info={exam.metadata ?? report.input} bytes={exam.bytes}/></div>
      <a className="primary" href={media(exam.id,'annotated.mp4')+'?download=1'}><ArrowDownToLine size={17}/> Annotated MP4</a>
    </header>
    <div className="workspace-tabs" aria-label="Examination views">
      <button className={tab==='review'?'active':''} onClick={()=>setTab('review')}><ScanLine size={16}/> Review video</button>
      <button className={tab==='patient'?'active':''} onClick={()=>setTab('patient')}><Play size={16}/> Patient view</button>
      <button className={tab==='xr'?'active':''} onClick={()=>setTab('xr')}><Glasses size={17}/> XR preview</button>
    </div>
    {tab==='xr' ? <XR exam={exam} refresh={refresh}/> : tab==='patient' ?
      <PatientView exam={exam} moments={confirmed} reel={reel}/> : <div className="review-grid">
      <div className="review-main">
        <section className="panel player-panel">
          <div className="player-heading"><div><Video size={16}/> <strong>Examination video</strong></div>
            <div className="segment" aria-label="Video stream">{(['annotated','source'] as const).map(mode=>
              <button key={mode} className={playerMode===mode?'active':''} onClick={()=>{
                if(mode!==playerMode){ pending.current=player.current?.currentTime ?? time; setPlayerMode(mode) }
              }}>{mode==='source'?'Original':'Annotated'}</button>)}</div></div>
          <video ref={player} key={playerMode} src={media(exam.id,`${playerMode}.mp4`)} controls playsInline preload="metadata" aria-label="Examination playback"
            onLoadedMetadata={e=>{if(pending.current!==null){e.currentTarget.currentTime=pending.current;pending.current=null}}}
            onTimeUpdate={e=>setTime(e.currentTarget.currentTime)}/>
          <div className="timeline">
            <div className="time-labels"><strong>Examination timeline</strong><span>{clock(time)} / {clock(report.video.duration)}</span></div>
            <div className="timeline-pins" aria-label="Bookmark timeline">
              <span className="time-fill" style={{width:`${100*time/report.video.duration}%`}}/>
              {report.findings.map(item=><button key={item.id} title={`${clock(item.time)} · ${item.label}`}
                aria-label={`Jump to ${item.label} at ${clock(item.time)}`} onClick={()=>chooseFinding(item)}
                className={`pin ${selected?.id===item.id?'selected':''} ${item.category==='cavity_candidate'?'flag-pin':''}`}
                style={{left:`${100 * item.time / report.video.duration}%`}}/>)}
              {confirmed.filter(item=>!item.finding_id).map(item=><button key={item.id}
                className="pin manual-pin" title={`${clock(item.time)} · ${item.label || 'Manual bookmark'}`}
                aria-label={`Jump to confirmed bookmark at ${clock(item.time)}`}
                onClick={()=>{manual();seek(item.time)}} style={{left:`${100*item.time/report.video.duration}%`}}/>)}
            </div>
            <ActivityChart report={report}/>
          </div>
        </section>
        <div className="metrics">
          <div><span>Recording frames</span><strong>{report.video.frames.toLocaleString()}</strong></div>
          <div><span>Frames with overlays</span><strong>{coverage}%</strong></div>
          <div><span>Tracked regions</span><strong>{report.tracks.length}</strong></div>
        </div>
        <section className="panel tracks">
          <div className="section-heading"><div><h2>Find a tooth</h2><p>Search a predicted type, FDI number or track ID.</p></div><Layers3 size={20}/></div>
          <label className="search"><Search size={16}/><input placeholder="Search e.g. central incisor or 21" aria-label="Search tracked teeth"
            value={query} onChange={e=>setQuery(e.target.value)}/></label>
          <div className="track-list">
            {tracks.map(track=><button key={track.id} onClick={()=>{
              const finding=report.findings.find(f=>f.track_id===track.id)
              if(finding) chooseFinding(finding)
              else { manual(); seek(track.start) }
            }}><span><strong>{track.label}</strong><small>{track.id} · {track.frames} frames</small></span>
              <span>{clock(track.start)}–{clock(track.end)}<ChevronRight size={15}/></span></button>)}
            {!tracks.length && <p className="quiet">No tracked regions match this search.</p>}
          </div>
        </section>
      </div>
      <aside className="review-side">
        <section className="panel inspector">
          <div className="section-heading"><div><h2>Review a moment</h2><p>Confirm a suggestion or add a bookmark.</p></div><button className="icon-button" onClick={manual} title="Add manual bookmark" aria-label="Add manual bookmark"><Plus size={19}/></button></div>
          <div className="finding-list">
            {report.findings.map(item=>{
              const saved=reviews.find(review=>review.id===item.id)
              return <button className={selected?.id===item.id?'active':''} key={item.id} onClick={()=>chooseFinding(item)}>
                <time>{clock(item.time)}</time><span>{item.label}</span>
                {saved ? <small className={`review-state ${saved.decision}`}>{saved.decision==='confirmed'?'Confirmed':'Dismissed'}</small> : <ChevronRight size={15}/>}
              </button>
            })}
            {!report.findings.length && <p className="quiet">No AI suggestions passed the threshold. Bookmark a moment manually.</p>}
          </div>
          {selected && <div className="evidence">
            <img src={path(exam.id,`/images/${selected.image}`)} alt={`Annotated source frame at ${clock(selected.time)}`}/>
            <div><span>AI suggestion · {clock(selected.time)}</span><a aria-label="Download evidence image"
              href={path(exam.id,`/images/${selected.image}?download=true`)}><ArrowDownToLine size={15}/></a></div>
          </div>}
          <div className="review-form">
            <label className="field">{selected?'Reviewed tooth label':'Bookmark label'}<input value={label} onChange={e=>setLabel(e.target.value)}
              maxLength={100} placeholder={selected?'Confirm or correct the suggestion':'At the current video time'}/></label>
            <label className="field">Reviewer note<textarea value={note} onChange={e=>setNote(e.target.value)} rows={2} maxLength={500} placeholder="What should the patient or dentist revisit?"/></label>
            <label className="toggle"><input type="checkbox" checked={flag} onChange={e=>setFlag(e.target.checked)}/><span>Flag for review</span></label>
            <div className="actions"><button className="primary" disabled={saving} onClick={()=>void save('confirmed')}><Check size={16}/>{selected?'Confirm bookmark':'Save bookmark'}</button>
              {selected && <button className="subtle" disabled={saving} onClick={()=>void save('dismissed')}>Dismiss</button>}</div>
            {feedback && <p role="status" className="notice">{feedback}</p>}
          </div>
        </section>
        <section className="panel export-panel">
          <div className="section-heading"><div><h2>Reviewed exports</h2><p>{confirmed.length} confirmed {confirmed.length===1?'bookmark':'bookmarks'}</p></div><Bookmark size={19}/></div>
          {confirmed.map(item=><label className="toggle bookmark-row" key={item.id}><input type="checkbox"
            checked={choice.includes(item.id)} onChange={()=>setPicked(choice.includes(item.id)?choice.filter(id=>id!==item.id):[...choice,item.id])}/>
            <span><strong>{item.label || 'Confirmed moment'}</strong><small>{clock(item.time)} {item.flagged && <Flag size={12}/>} {item.note}</small></span></label>)}
          <button className="subtle wide" disabled={!choice.length||building||choice.length>20} onClick={()=>void buildReel()}>
            {building?<LoaderCircle className="spin" size={16}/>:<Clapperboard size={16}/>} {building?'Building…':'Build reviewed reel'}</button>
          {choice.length>20 && <p className="quiet">Select up to 20 bookmarks per reel.</p>}
          {reel && <><video src={reel} controls playsInline/><a className="subtle wide" href={reel+(reel.includes('?')?'&':'?')+'download=true'}><ArrowDownToLine size={16}/> Download reviewed reel</a></>}
          <a className="subtle wide" href={path(exam.id,'/export')}><FileText size={16}/> Download examination report</a>
          {report.findings.length>0 && <a className="subtle wide" href={path(exam.id,'/evidence.zip')}><ArrowDownToLine size={16}/> Download evidence images</a>}
          <a className="download-link" href={media(exam.id,'timeline.jsonl')+'?download=true'}>Frame annotations <ArrowUpRight size={13}/></a>
        </section>
        <details className="panel limits"><summary>Analysis details</summary>
          <p>{report.profile} · {report.device.toUpperCase()} · {report.stats.detection_frames} detection passes</p>
          <p>{report.stats.numbered_frames} frames with a stable FDI suggestion.</p>
          {report.cavity_review && <p>{report.stats.cavity_frames} frames with a possible cavity flag.</p>}
          <ul>{report.limitations.map(item=><li key={item}>{item}</li>)}</ul>
        </details>
      </aside>
    </div>}
  </>
}

function App() {
  const [exams,setExams]=useState<Exam[]>([])
  const [id,setId]=useState<string|null>(null)
  const [exam,setExam]=useState<Exam>()
  const [uploading,setUploading]=useState(false)
  const [pollVersion,setPollVersion]=useState(0)
  const [error,setError]=useState('')
  async function list() { const data=await api<Exam[]>('/api/exams');setExams(data);return data }
  useEffect(()=>{
    list().then(data=>{if(data[0])setId(data[0].id)}).catch(e=>setError(message(e)))
  },[])
  useEffect(()=>{
    if(!id)return
    let active=true
    let timer:ReturnType<typeof setTimeout>
    async function poll(){
      try{
        const result=await api<Exam>(path(id!))
        if(!active)return
        setExam(result);setError('')
        setExams(rows=>rows.map(row=>row.id===result.id?{...row,state:result.state}:row))
        if(result.state==='queued'||result.state==='running')timer=setTimeout(poll,1250)
      }catch(e){ if(active){setError(message(e));timer=setTimeout(poll,3000)} }
    }
    void poll()
    return ()=>{active=false;clearTimeout(timer)}
  },[id,pollVersion])
  async function refresh() {
    if (!id) return
    const updated=await api<Exam>(path(id))
    setExam(current=>current?.id===id ? updated : current)
  }
  function choose(next: string) {
    if (next!==id) { setExam(undefined); setId(next) }
    setUploading(false)
  }
  function created(next: Exam) {setExams(old=>[next,...old]);setExam(next);setId(next.id);setUploading(false)}
  async function retry() {
    if(!exam)return
    try {
      const result=await api<Exam>(path(exam.id,'/retry'),{method:'POST'})
      setExam(result)
      setPollVersion(version=>version+1)
    } catch(e){setError(message(e))}
  }
  return <div className="app-shell">
    <nav className="sidebar" aria-label="Examination workspace">
      <button className="brand" onClick={()=>setUploading(false)}><span className="brand-icon"><ScanLine size={23}/></span><span>DentiLens <b>XR</b><small>Examination workspace</small></span></button>
      <button className="primary new-exam" onClick={()=>setUploading(true)}><Plus size={18}/> New examination</button>
      <div className="sidebar-caption"><span>EXAMINATIONS</span><span>{exams.length}</span></div>
      <div className="exam-list">
        {exams.map(row=><button key={row.id} className={!uploading&&id===row.id?'active':''} onClick={()=>choose(row.id)}>
          <Film size={16}/><span><strong title={row.title}>{row.title}</strong><small>{row.state} · {row.profile}</small></span>{id===row.id&&!uploading&&<ChevronRight size={15}/>}</button>)}
        {!exams.length&&<small>No recordings yet</small>}
      </div>
      <div className="sidebar-bottom"><Activity size={17}/><span>Local workspace<small>Files stay on this Mac</small></span></div>
    </nav>
    <main>
      {error && <div role="alert" className="error page-error">{error}</div>}
      {uploading ? <UploadScreen back={()=>setUploading(false)} submit={created}/> :
        exam?.state==='ready' && exam.report ? <ExamReview key={exam.id} exam={exam} refresh={refresh}/> :
        exam ? <div className="processing panel"><div className="eyebrow">{exam.state==='failed'?'ANALYSIS STOPPED':'FULL RECORDING ANALYSIS'}</div>
          <h1>{exam.state==='failed'?'Analysis needs attention':'Analyzing your examination'}</h1><p>{exam.title}</p>
          <MediaMetrics info={exam.metadata} bytes={exam.bytes}/>
          {exam.state==='failed' ? <><div className="error"><CircleAlert size={16}/> {exam.error}</div><button className="primary" onClick={()=>void retry()}>Retry analysis</button></>
            : <><div className="progress-track" role="progressbar" aria-label="Analysis progress" aria-valuenow={Math.round(exam.progress*100)}><span style={{width:`${exam.progress*100}%`}}/></div>
              <div className="progress-label"><span><LoaderCircle size={16} className="spin"/> {exam.stage}</span><strong>{Math.round(exam.progress*100)}%</strong></div>
              <video className="processing-preview" src={path(exam.id,'/preview')} controls playsInline preload="metadata"
                aria-label="Uploaded recording preview"/></>}
        </div> :
          <div className="welcome"><span className="upload-icon"><ScanLine size={31}/></span><h1>Review a dental examination</h1>
            <p>Turn a recording into a full annotated video, then find teeth and save the moments you want to explain.</p>
            <button className="primary" onClick={()=>setUploading(true)}><Plus size={17}/> New examination</button></div>}
    </main>
  </div>
}
createRoot(document.getElementById('root')!).render(<App/>)
