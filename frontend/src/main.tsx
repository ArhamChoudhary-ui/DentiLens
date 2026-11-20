import { lazy, Suspense, useEffect, useMemo, useRef, useState } from 'react'
import type { FormEvent } from 'react'
import { createRoot } from 'react-dom/client'
import {
  Activity, ArrowDownToLine, ArrowLeft, Bookmark, Check, ChevronRight,
  CircleAlert, Clapperboard, Crosshair, FileText, Film, Flag, Glasses,
  LoaderCircle, Mic, Play, Plus, ScanLine, Search, Upload, X,
} from 'lucide-react'
import { Bar, BarChart, ResponsiveContainer, Tooltip } from 'recharts'
import { api, clock, send } from './api'
import type { Box, Exam, Finding, MediaInfo, Report, Review } from './api'
import { FrameAnalysis, FullVideoExport, RegionEditor, ToothChart } from './review-tools'
import { ExamIntelligence } from './intelligence'
import { Instruments } from './instruments'
import './styles.css'

const Anatomy = lazy(() => import('./anatomy'))

const profiles = [
  { id: 'fast', name: 'Fast', detail: 'Detect every 2 s' },
  { id: 'balanced', name: 'Balanced', detail: 'Detect every 1 s' },
  { id: 'detailed', name: 'Detailed', detail: 'Detect every 0.5 s' },
]
const examStates: Record<Exam['state'], string> = {
  queued: 'Queued', running: 'Analyzing', ready: 'Ready', failed: 'Failed', canceling: 'Stopping', cancelled: 'Cancelled',
}
type Feedback = { text: string; error?: boolean }

function sizeText(bytes: number) { return `${(bytes / 1024 / 1024).toFixed(1)} MB` }
function path(id: string, endpoint = '') { return `/api/exams/${id}${endpoint}` }
function media(id: string, name: string) { return path(id, `/media/${name}`) }
function message(e: unknown) { return e instanceof Error ? e.message : 'The request failed. Try again.' }
function reviewState(review?: Review) {
  return review?.decision==='confirmed' ? 'Confirmed' : review?.decision==='dismissed' ? 'Dismissed' : 'Unreviewed'
}
function ActionFeedback({ feedback }: { feedback?: Feedback }) {
  return feedback ? <p role={feedback.error?'alert':'status'} className={feedback.error?'error':'notice'}>{feedback.text}</p> : null
}

function UploadScreen({ back, submit }: { back: () => void; submit: (exam: Exam) => void }) {
  const input = useRef<HTMLInputElement>(null)
  const [file, setFile] = useState<File | null>(null)
  const [url, setUrl] = useState('')
  const [meta, setMeta] = useState<{ duration: number; width: number; height: number }>()
  const [profile, setProfile] = useState('balanced')
  const [name, setName] = useState('')
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
    body.append('name', name.trim())
    body.append('device', device)
    body.append('cavities', String(cavities))
    try { submit(await api<Exam>('/api/exams', { method: 'POST', body })) }
    catch (e) { setError(message(e)) }
    finally { setBusy(false) }
  }
  return <form onSubmit={upload} className="upload-page">
    <button type="button" className="subtle back" disabled={busy} onClick={back}><ArrowLeft size={16}/> Back</button>
    <div className="upload-heading">
      <h1>New examination</h1>
    </div>
    <div className="upload-grid">
      <div className="panel upload-media">
        <input ref={input} type="file" accept=".mp4,.mov,.m4v,.webm" aria-label="Choose examination video"
          className="sr-only" disabled={busy} onChange={e => { setError(''); setFile(e.target.files?.[0] ?? null) }} />
        {file ? <>
          <video key={url} src={url} controls playsInline onLoadedMetadata={e => {
            const v=e.currentTarget; setMeta({duration: v.duration, width: v.videoWidth, height: v.videoHeight})
          }}/>
          <div className="file-row"><Film size={19}/><div><strong>{file.name}</strong><small>
            {[sizeText(file.size), meta && clock(meta.duration), meta && `${meta.width} × ${meta.height}`].filter(Boolean).join(' · ')}
          </small></div><button type="button" className="icon-button" aria-label="Remove video" disabled={busy} onClick={() => { setFile(null); setError(''); if (input.current) input.current.value='' }}><X size={18}/></button></div>
        </> : <button type="button" className="drop" onClick={() => input.current?.click()}
          onDragOver={e => e.preventDefault()} onDrop={e => { e.preventDefault(); setError(''); setFile(e.dataTransfer.files[0] ?? null) }}>
          <span className="upload-icon"><Upload size={25}/></span><strong>Choose a video</strong>
          <span>Or drop an MP4, MOV, M4V or WebM here</span><small>Up to 500 MB and 10 minutes</small>
        </button>}
      </div>
      <section className="panel settings">
        <h2>Analysis settings</h2>
        <p>Upload footage you have permission to review. Files stay on this Mac.</p>
        <label className="field">Examination name<input value={name} maxLength={100} disabled={busy}
          placeholder="Optional pseudonym" onChange={e=>setName(e.target.value)}/></label>
        <p>More frequent detection takes longer.</p>
        <div className="settings-options" role="group" aria-label="Analysis frequency">
          {profiles.map(item => <button type="button" key={item.id} className={profile===item.id?'active':''}
            aria-pressed={profile===item.id} disabled={busy} onClick={() => setProfile(item.id)}><strong>{item.name}</strong><small>{item.detail}</small></button>)}
        </div>
        <label className="field">Device<select value={device} disabled={busy} onChange={e => setDevice(e.target.value)}>
          <option value="auto">Automatic · Apple GPU if available</option>
          <option value="cpu">CPU</option>
        </select></label>
        <label className="toggle"><input type="checkbox" checked={cavities} disabled={busy} onChange={e => setCavities(e.target.checked)}/>
          <span>Flag possible cavities<small>Research only</small></span></label>
        {!ready && <div role="alert" className="error">Models are missing. Follow the README setup steps.</div>}
        {error && <div role="alert" className="error">{error}</div>}
        <button className="primary wide" disabled={!file || busy || !ready}>
          {busy ? <LoaderCircle className="spin" size={17}/> : <ScanLine size={17}/>} {busy ? 'Uploading…' : 'Analyze video'}
        </button>
      </section>
    </div>
  </form>
}

function MediaMetrics({ info, bytes }: { info?: MediaInfo | null; bytes: number }) {
  if (!info) return <span>{sizeText(bytes)}</span>
  return <div className="metadata"><span><Film size={14}/>{clock(info.duration)}</span><span>{info.width} × {info.height}</span>
    <span>{Number(info.fps.toFixed(2))} fps</span><span>{sizeText(bytes)}</span></div>
}

function ActivityChart({ report }: { report: Report }) {
  const total = report.video.duration
  const bins = Array.from({length: 24}, (_, i) => ({time: clock(total * i / 24), count: 0}))
  report.findings.forEach(item => bins[Math.min(23, Math.floor(item.time / total * 24))].count++)
  return <div className="chart" aria-label="AI suggestions by time">
    <ResponsiveContainer width="100%" height={54}><BarChart data={bins} barCategoryGap="18%">
      <Tooltip labelFormatter={(_, items) => items[0]?.payload.time} formatter={(n) => [n, 'AI suggestions']}
        cursor={{fill:'#263c3b'}} contentStyle={{background:'#1c242c',border:'1px solid #53646f',color:'#e1e6ea',borderRadius:8,fontSize:12}}/>
      <Bar dataKey="count" fill="#799485" radius={[2,2,0,0]} minPointSize={2}/>
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

function XR({ exam, refresh, openReview }: { exam: Exam; refresh: () => Promise<void>; openReview: (time: number) => void }) {
  const player = useRef<HTMLVideoElement>(null)
  const [command, setCommand] = useState('')
  const [notice, setNotice] = useState<Feedback>()
  const [listening, setListening] = useState(false)
  const [time, setTime] = useState(0)
  const [note,setNote] = useState('')
  const [addingNote,setAddingNote] = useState(false)
  const bookmarks = (exam.reviews ?? []).filter(item=>item.decision==='confirmed').sort((a,b)=>a.time-b.time)
  const nearest = exam.report?.findings.find(item=>Math.abs(item.time-time)<.4)
  async function execute(text: string) {
    const phrase = text.toLowerCase().trim().replace(/[.!?]+$/, '')
    const video=player.current
    if (!video) return
    try {
      if (phrase === 'bookmark this' || phrase === 'bookmark') {
        const timestamp=Math.min(video.currentTime, Math.max(0, (exam.report?.video.duration ?? 0) - .001))
        await send(path(exam.id,'/reviews'), {time:timestamp, label:'XR bookmark',note,decision:'confirmed'})
        await refresh()
        setNote('');setAddingNote(false)
        setNotice({text: `Bookmark saved at ${clock(timestamp, true)}.`})
      } else if (phrase === 'pause') { video.pause(); setNotice({text:'Paused.'}) }
      else if (phrase === 'play' || phrase === 'resume') { await video.play(); setNotice({text:'Playing.'}) }
      else if (phrase === 'open review') { openReview(video.currentTime) }
      else if (phrase === 'add note') { setAddingNote(true); video.pause() }
      else if (phrase === 'next bookmark' || phrase === 'previous bookmark') {
        const next = phrase === 'next bookmark' ? bookmarks.find(item=>item.time>video.currentTime+.04)
          : bookmarks.slice().reverse().find(item=>item.time<video.currentTime-.04)
        if (next) {video.currentTime=next.time; video.pause();setNotice({text:clock(next.time,true)})}
        else setNotice({text:'No bookmark in that direction.'})
      } else if (phrase.startsWith('show the ')) {
        const name=phrase.slice(9)
        const match=exam.report?.findings.find(item=>item.label.toLowerCase().includes(name))
        if (match) { video.currentTime=match.time;video.pause();setNotice({text:match.label}) }
        else setNotice({text:'No matching tooth suggestion.'})
      } else { setNotice({text:'Try “bookmark this”, “show the canine” or “next bookmark”.'}) }
    } catch(e) { setNotice({text:message(e), error:true}) }
  }
  function listen() {
    const w = window as unknown as {SpeechRecognition?: new()=>LocalSpeech; webkitSpeechRecognition?: new()=>LocalSpeech}
    const Recognition = w.SpeechRecognition ?? w.webkitSpeechRecognition
    if (!Recognition) { setNotice({text:'Local speech is unavailable. Type a command.', error:true}); return }
    const recognition = new Recognition()
    if (!('processLocally' in recognition)) {
      setNotice({text:'Local speech is unavailable. Type a command.', error:true}); return
    }
    recognition.processLocally = true
    recognition.lang = 'en-US'
    recognition.onresult = e => { const heard=e.results[0][0].transcript; setCommand(heard); void execute(heard) }
    recognition.onerror = () => setNotice({text:'The microphone failed. Type a command.', error:true})
    recognition.onend = () => setListening(false)
    try { recognition.start(); setListening(true); setNotice({text:'Listening…'}) }
    catch (e) { setNotice({text:message(e), error:true}) }
  }
  return <div className="xr-page panel">
    <div className="section-heading"><h2>Simulated glasses view</h2><Glasses size={25}/></div>
    <div className="xr-video"><video ref={player} controls src={media(exam.id,'source.mp4')} playsInline
      onTimeUpdate={e=>setTime(e.currentTarget.currentTime)}/><Crosshair className="reticle" size={39}/><span className="xr-time">{clock(time,true)}</span>{nearest && <span className="xr-tooth">{nearest.label} · Suggestion</span>}</div>
    <form className="command" onSubmit={e => {e.preventDefault(); void execute(command)}}>
      <input value={command} onChange={e=>setCommand(e.target.value)} aria-label="Simulated voice command" placeholder='Type “bookmark this”'/>
      <button className="subtle" disabled={!command.trim()}><Play size={16}/> Run command</button>
      <button type="button" className="subtle" onClick={listen} disabled={listening}><Mic size={16}/> {listening?'Listening…':'Use microphone'}</button>
      <button type="button" className="primary" onClick={()=>void execute('bookmark this')}><Bookmark size={16}/> Bookmark this</button>
    </form>
    {addingNote && <label className="field">Note for the next bookmark<textarea maxLength={500} rows={2} value={note}
      onChange={e=>setNote(e.target.value)}/></label>}
    <div className="actions">
      <button className="subtle compact-action" onClick={()=>void execute('previous bookmark')}>Previous bookmark</button>
      <button className="subtle compact-action" onClick={()=>void execute('next bookmark')}>Next bookmark</button>
      <button className="subtle compact-action" onClick={()=>void execute('add note')}>Add note</button>
      <button className="subtle compact-action" onClick={()=>void execute('open review')}>Open review</button>
    </div>
    <ActionFeedback feedback={notice}/>
  </div>
}

function PatientView({ exam, moments, reel, openAnatomy }: { exam: Exam; moments: Review[]; reel: string; openAnatomy:(fdi:string)=>void }) {
  const player = useRef<HTMLVideoElement>(null)
  const [selected,setSelected]=useState<string>()
  const [compare,setCompare]=useState<string>()
  const first=moments.find(item=>item.id===selected)
  const second=moments.find(item=>item.id===compare && item.id!==selected)
  const still=(item: Review)=>path(exam.id,`/still.jpg?bookmark=${item.id}&v=${encodeURIComponent(item.updated ?? '')}`)
  return <section className="panel patient-page">
    <div className="section-heading"><h2>Confirmed bookmarks</h2><Play size={21}/></div>
    <video ref={player} src={media(exam.id,'source.mp4')} controls playsInline preload="metadata"/>
    {moments.length ? <div className="patient-moments">
      {moments.slice().sort((a,b)=>a.time-b.time).map(item=><button key={item.id} onClick={()=>{
        if (player.current) { player.current.currentTime=item.time; player.current.pause() }
        setSelected(item.id)
      }} aria-pressed={selected===item.id}>
        <Play size={15}/><time>{clock(item.time, true)}</time><span><strong>{item.label || 'Bookmark'}</strong>
        {item.fdi_number && <small>Tooth {item.fdi_number}</small>}{item.note && <small>{item.note}</small>}</span>
      </button>)}
    </div> : <p className="quiet">Confirm a bookmark in Review video.</p>}
    {first && <div className="patient-stills">
      {first.fdi_number && <div><button className="subtle compact-action" onClick={()=>{
        if(first.fdi_number)openAnatomy(first.fdi_number)
      }}>View tooth {first.fdi_number} in 3D</button></div>}
      <div className="still-comparison">
        {[first,second].filter((item): item is Review=>Boolean(item)).map(item=><figure key={item.id}>
          <img src={still(item)} alt={item.label || 'Reviewed examination frame'}/><figcaption>{clock(item.time,true)} · {item.label || 'Bookmark'}</figcaption>
        </figure>)}
      </div>
      {moments.length>1 && <label className="field">Compare with<select value={second?.id ?? ''} onChange={e=>setCompare(e.target.value)}>
        <option value="">Choose a bookmark</option>
        {moments.filter(item=>item.id!==first.id).map(item=><option key={item.id} value={item.id}>{clock(item.time,true)} · {item.label || 'Bookmark'}</option>)}
      </select></label>}
    </div>}
    {reel && <a className="subtle" href={reel+(reel.includes('?')?'&':'?')+'download=true'}><ArrowDownToLine size={16}/> Download reviewed reel</a>}
  </section>
}

function ExamReview({ exam, refresh, remove }: { exam: Exam; refresh: () => Promise<void>; remove: () => Promise<void> }) {
  const report=exam.report!
  const player=useRef<HTMLVideoElement>(null)
  const pending=useRef<number | null>(null)
  const [playerMode,setPlayerMode]=useState<'annotated' | 'source'>('annotated')
  const [time,setTime]=useState(0)
  const [selected,setSelected]=useState<Finding | null>(report.findings[0] ?? null)
  const [manualTime,setManualTime]=useState<number | null>(null)
  const [query,setQuery]=useState('')
  const [label,setLabel]=useState(selected?.label ?? '')
  const [note,setNote]=useState('')
  const [flag,setFlag]=useState(false)
  const [fdi,setFdi]=useState('')
  const [box,setBox]=useState<Box | null>(null)
  const [feedback,setFeedback]=useState<Feedback>()
  const [reelFeedback,setReelFeedback]=useState<Feedback>()
  const [reel,setReel]=useState(exam.reel_url ?? '')
  const [building,setBuilding]=useState(false)
  const [saving,setSaving]=useState(false)
  const [picked,setPicked]=useState<string[] | null>(null)
  const [tab,setTab]=useState<'review' | 'teeth' | 'patient' | 'xr' | 'intelligence' | 'anatomy' | 'instruments'>('review')
  const [anatomyTooth,setAnatomyTooth]=useState('')
  const [chartTooth,setChartTooth]=useState('')
  const reviews=exam.reviews ?? []
  const confirmed=reviews.filter(item=>item.decision==='confirmed')
  const selectedReview=reviews.find(item=>item.id===selected?.id)
  const targetTime=selected?.time ?? manualTime ?? time
  const choice=(picked ?? confirmed.map(item=>item.id)).filter(id=>confirmed.some(item=>item.id===id))
  useEffect(()=>{
    if (!selected) return
    const saved=reviews.find(item=>item.id===selected.id)
    setLabel(saved?.label ?? selected.label.replace(/^FDI \d{2} · /,''))
    setNote(saved?.note ?? '')
    setFlag(saved?.flagged ?? false)
    setFdi(saved?.fdi_number ?? '')
    setBox(saved?.box ?? null)
  },[selected, exam.reviews])
  function seek(seconds: number) {
    const at=Math.min(Math.max(0,seconds),Math.max(0,report.video.duration-.001))
    const v=player.current
    if (v && v.readyState >= 1) { v.currentTime=at; v.pause() }
    else pending.current=at
    setTime(at)
  }
  function openAnatomy(code:string) {setAnatomyTooth(code);setTab('anatomy')}
  function chooseFinding(item: Finding) { setSelected(item); seek(item.time); setFeedback(undefined) }
  function manual(seconds: number) {
    const at=Math.min(seconds,Math.max(0,report.video.duration-.001))
    setSelected(null); setManualTime(at); setLabel(''); setNote(''); setFlag(false); setFdi(''); setBox(null); setFeedback(undefined)
    seek(at)
  }
  function openReview(seconds:number) {
    const finding=report.findings.find(item=>item.time===seconds)
    if (finding) chooseFinding(finding)
    else manual(seconds)
    setTab('review')
  }
  async function save(decision: Review['decision']) {
    if (saving) return
    if (decision==='confirmed' && ((fdi && !/^[1-4][1-8]$/.test(fdi)) ||
        (box && (box[0]>=box[2] || box[1]>=box[3])))) {
      setFeedback({text:'Check the FDI number and box dimensions.', error:true});return
    }
    setSaving(true); setFeedback(undefined)
    try {
      const saved=await send<Review>(path(exam.id,'/reviews'), {time: Math.min(targetTime,report.video.duration-.001),
        finding_id:selected?.id ?? null,label:label.trim(),note:note.trim(),flagged:flag,fdi_number:fdi || null,box,decision})
      await refresh()
      setFeedback({text:decision==='confirmed'?`Bookmark saved at ${clock(saved.time, true)}.`:'AI suggestion dismissed.'})
    } catch(e) { setFeedback({text:message(e), error:true}) }
    finally { setSaving(false) }
  }
  async function buildReel() {
    setBuilding(true);setReelFeedback(undefined)
    try {
      const result=await send<{url:string}>(path(exam.id,'/reel'),{bookmarks:choice})
      setReel(`${result.url}?v=${Date.now()}`)
      setReelFeedback({text:'Reel ready.'})
    } catch (e) { setReelFeedback({text:message(e), error:true}) }
    finally { setBuilding(false) }
  }
  const tracks=useMemo(()=>{
    const q=query.toLowerCase().trim()
    return report.tracks.filter(t=>!q || [t.label,t.id,t.fdi_number,t.tooth_type].some(v=>v?.toLowerCase().includes(q)))
  },[query,report.tracks])
  const coverage=Math.round(100 * report.stats.annotated_frames / report.video.frames)
  return <>
    <header className="exam-heading">
      <div>
        <h1>Examination review</h1><div className="record-name">{exam.title}</div><MediaMetrics info={exam.metadata ?? report.input} bytes={exam.bytes}/></div>
    </header>
    <div className="workspace-tabs" aria-label="Examination views">
      <button className={tab==='review'?'active':''} aria-pressed={tab==='review'} onClick={()=>setTab('review')}><ScanLine size={16}/> Review video</button>
      <button className={tab==='teeth'?'active':''} aria-pressed={tab==='teeth'} onClick={()=>setTab('teeth')}>Tooth chart</button>
      <button className={tab==='intelligence'?'active':''} aria-pressed={tab==='intelligence'} onClick={()=>setTab('intelligence')}>Search & quality</button>
      <button className={tab==='instruments'?'active':''} aria-pressed={tab==='instruments'} onClick={()=>setTab('instruments')}>Instruments</button>
      <button className={tab==='anatomy'?'active':''} aria-pressed={tab==='anatomy'} onClick={()=>{setAnatomyTooth('');setTab('anatomy')}}>3D anatomy</button>
      <button className={tab==='patient'?'active':''} aria-pressed={tab==='patient'} onClick={()=>setTab('patient')}><Play size={16}/> Patient view</button>
      <button className={tab==='xr'?'active':''} aria-pressed={tab==='xr'} onClick={()=>setTab('xr')}><Glasses size={17}/> XR preview</button>
    </div>
    {tab==='xr' ? <XR exam={exam} refresh={refresh} openReview={seconds=>{seek(seconds);setTab('review')}}/> : tab==='teeth' ?
      <ToothChart exam={exam} jump={item=>{chooseFinding(item);setTab('review')}} jumpToTime={openReview} initial={chartTooth}
        openAnatomy={openAnatomy}/> : tab==='instruments' ?
      <Instruments examId={exam.id} duration={report.video.duration} currentTime={time} jump={seconds=>{setPlayerMode('source');openReview(seconds)}} refresh={refresh}/> : tab==='anatomy' ?
      <Suspense fallback={<p role="status">Loading anatomy…</p>}><Anatomy examId={exam.id} initial={anatomyTooth} jump={openReview}
        openChart={code=>{setChartTooth(code);setTab('teeth')}}/></Suspense> : tab==='intelligence' ?
      <ExamIntelligence exam={exam} jump={openReview}/> : tab==='patient' ?
      <PatientView exam={exam} moments={confirmed} reel={reel} openAnatomy={openAnatomy}/> : <div className="review-grid">
      <div className="review-main">
        <section className="panel player-panel">
          <div className="player-heading"><strong>Video</strong>
            <div className="player-controls">
              <div className="segment" role="group" aria-label="Video version">{(['annotated','source'] as const).map(mode=>
                <button key={mode} className={playerMode===mode?'active':''} aria-pressed={playerMode===mode} onClick={()=>{
                  if(mode!==playerMode){ pending.current=player.current?.currentTime ?? time; setPlayerMode(mode) }
                }}>{mode==='source'?'Original':'Annotated'}</button>)}</div>
              <a className="subtle" href={media(exam.id,'annotated.mp4')+'?download=1'}><ArrowDownToLine size={15}/> Download annotated MP4</a>
            </div>
          </div>
          <video ref={player} key={playerMode} src={media(exam.id,`${playerMode}.mp4`)} controls playsInline preload="metadata" aria-label="Examination playback"
            onLoadedMetadata={e=>{
              const position=pending.current ?? time
              if (position>0) e.currentTarget.currentTime=position
              pending.current=null
            }}
            onTimeUpdate={e=>setTime(e.currentTarget.currentTime)}/>
          <div className="timeline">
            <div className="time-labels"><strong>Timeline</strong><span>{clock(time, true)} / {clock(report.video.duration, true)}</span></div>
            <div className="timeline-pins" aria-label="Bookmark timeline">
              <span className="time-fill" style={{width:`${100*time/report.video.duration}%`}}/>
              {report.findings.map(item=><button key={item.id} title={`${clock(item.time, true)} · ${item.label}`}
                aria-label={`Jump to ${item.label} at ${clock(item.time, true)}`} onClick={()=>chooseFinding(item)}
                aria-pressed={selected?.id===item.id}
                className={`pin ${selected?.id===item.id?'selected':''} ${item.category==='cavity_candidate'?'flag-pin':''}`}
                style={{left:`${100 * item.time / report.video.duration}%`}}/>)}
              {confirmed.filter(item=>!item.finding_id).map(item=><button key={item.id}
                className="pin manual-pin" title={`${clock(item.time, true)} · ${item.label || 'Bookmark'}`}
                aria-label={`Jump to bookmark at ${clock(item.time, true)}`}
                onClick={()=>seek(item.time)} style={{left:`${100*item.time/report.video.duration}%`}}/>)}
            </div>
          </div>
        </section>
        <div className="metrics">
          <div><span>Video frames</span><strong>{report.video.frames.toLocaleString()}</strong></div>
          <div><span>Frames with overlays</span><strong>{coverage}%</strong></div>
          <div><span>Tracked regions</span><strong>{report.tracks.length}</strong></div>
        </div>
        <section className="panel tracks">
          <h2>Find a tooth</h2>
          <label className="search"><Search size={16}/><input placeholder="Tooth name, FDI number or track ID" aria-label="Search tooth tracks"
            value={query} onChange={e=>setQuery(e.target.value)}/></label>
          <div className="track-list">
            {tracks.map(track=><button key={track.id} onClick={()=>{
              const finding=report.findings.find(f=>f.track_id===track.id)
              if(finding) chooseFinding(finding)
              else manual(track.start)
            }}><span><strong>{track.label}</strong><small>{track.id} · {track.frames} frames</small></span>
              <span>{clock(track.start)}–{clock(track.end)}<ChevronRight size={15}/></span></button>)}
            {!tracks.length && <p className="quiet">{report.tracks.length?'No matches. Try another term.':'No tooth tracks for this video.'}</p>}
          </div>
        </section>
      </div>
      <aside className="review-side">
        <section className="panel inspector">
          <div className="section-heading"><h2>AI suggestions</h2>
            <button className="subtle compact-action" disabled={saving} onClick={()=>manual(player.current?.currentTime ?? time)}>
              <Plus size={15}/> Add bookmark
            </button>
          </div>
          <div className="finding-list">
            {report.findings.map(item=>{
              const saved=reviews.find(review=>review.id===item.id)
              return <button className={selected?.id===item.id?'active':''} aria-pressed={selected?.id===item.id} key={item.id} onClick={()=>chooseFinding(item)}>
                <time>{clock(item.time, true)}</time><span>{item.label}</span>
                {saved ? <small className={`review-state ${saved.decision}`}>{reviewState(saved)}</small> : <ChevronRight size={15}/>}
              </button>
            })}
            {!report.findings.length && <p className="quiet">No AI suggestions. Add a bookmark from the video.</p>}
          </div>
          {selected && <div className="evidence">
            <img src={path(exam.id,`/images/${selected.image}`)} alt={`Annotated frame at ${clock(selected.time, true)}`}/>
            <div><span>{clock(selected.time, true)} <span className={`review-state ${selectedReview?.decision ?? ''}`}>· {reviewState(selectedReview)}</span></span><a aria-label="Download evidence image" title="Download evidence image"
              href={path(exam.id,`/images/${selected.image}?download=true`)}><ArrowDownToLine size={15}/></a></div>
          </div>}
          <div className="review-form">
            {!selected && <small>Bookmark at {clock(targetTime, true)}</small>}
            <label className="field">Bookmark label<input value={label} onChange={e=>setLabel(e.target.value)}
              disabled={saving} maxLength={100} placeholder="Optional"/></label>
            <details className="correction-options"><summary>FDI number and region</summary>
              <label className="field">Confirmed FDI number<input value={fdi} inputMode="numeric" maxLength={2}
                placeholder={selected?.fdi_number ?? 'Optional'} onChange={e=>setFdi(e.target.value.replace(/\D/g,''))} disabled={saving}/></label>
              <RegionEditor key={selected?.id ?? 'manual'} src={path(exam.id,`/still.jpg?time=${Math.min(targetTime,report.video.duration-.001)}`)}
                value={box ?? (selected?.box?.map((value,index)=>value / (index % 2 ? report.video.height : report.video.width)) as Box | undefined) ?? null} change={setBox}/>
            </details>
            <label className="field">Note<textarea value={note} onChange={e=>setNote(e.target.value)}
              disabled={saving} rows={2} maxLength={500} placeholder="Optional"/></label>
            <label className="toggle"><input type="checkbox" checked={flag} disabled={saving} onChange={e=>setFlag(e.target.checked)}/><span>Flag for review</span></label>
            <div className="actions"><button className="primary" disabled={saving} onClick={()=>void save('confirmed')}><Check size={16}/>
              {saving?'Saving…':!selected?'Save bookmark':selectedReview?.decision==='confirmed'?'Save changes':'Confirm bookmark'}
            </button>
              {selected && selectedReview?.decision!=='dismissed' && <button className="subtle" disabled={saving} onClick={()=>void save('dismissed')}>Dismiss</button>}</div>
            <ActionFeedback feedback={feedback}/>
          </div>
        </section>
        <section className="panel export-panel">
          <div className="section-heading"><div><h2>Exports</h2>
            <p>{confirmed.length?`${choice.length} of ${confirmed.length} bookmarks selected`:'Confirm a bookmark to build a reel.'}</p></div><Bookmark size={19}/></div>
          {confirmed.map(item=><label className="toggle bookmark-row" key={item.id}><input type="checkbox"
            checked={choice.includes(item.id)} disabled={building} onChange={()=>{
              setPicked(choice.includes(item.id)?choice.filter(id=>id!==item.id):[...choice,item.id])
              setReelFeedback(undefined)
            }}/>
            <span><strong>{item.label || 'Bookmark'}</strong><small>{clock(item.time, true)} {item.flagged && <Flag size={12}/>} {item.note}</small></span></label>)}
          <button className="subtle wide" disabled={!choice.length||building||choice.length>20} onClick={()=>void buildReel()}>
            {building?<LoaderCircle className="spin" size={16}/>:<Clapperboard size={16}/>} {building?'Building…':'Build reviewed reel'}</button>
          {choice.length>20 && <p className="quiet">Select up to 20 bookmarks per reel.</p>}
          {confirmed.length>0 && choice.length===0 && <p className="quiet">Select bookmarks to build a reel.</p>}
          <ActionFeedback feedback={reelFeedback}/>
          {reel && <><video src={reel} controls playsInline/><a className="subtle wide" href={reel+(reel.includes('?')?'&':'?')+'download=true'}><ArrowDownToLine size={16}/> Download reviewed reel</a></>}
          <div className="export-links">
            <a className="download-link" href={path(exam.id,'/export')}><FileText size={15}/> Download report</a>
            {report.findings.length>0 && <a className="download-link" href={path(exam.id,'/evidence.zip')}><ArrowDownToLine size={15}/> Download evidence images</a>}
            <a className="download-link" href={media(exam.id,'timeline.jsonl')+'?download=true'}><ArrowDownToLine size={15}/> Download frame annotations</a>
          </div>
        </section>
        <FullVideoExport exam={exam} request={async options=>{
          const result=await send<{url:string}>(path(exam.id,'/video-export'),options)
          return result.url+'?v='+Date.now()
        }}/>
        <details className="panel limits"><summary>Analysis details</summary>
          <p>{profiles.find(item=>item.id===report.profile)?.name ?? report.profile} · {report.device.toUpperCase()} · {report.stats.detection_frames} detection passes</p>
          <ActivityChart report={report}/>
          <FrameAnalysis exam={exam}/>
          <p>Stable FDI suggestions in {report.stats.numbered_frames} frames.</p>
          <p>{report.cavity_review?`Possible cavity flags in ${report.stats.cavity_frames} frames.`:'Cavity flags off.'}</p>
          <ul>{report.limitations.map(item=><li key={item}>{item}</li>)}</ul>
          <button className="subtle compact-action delete-exam" onClick={()=>void remove()}>Delete examination</button>
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
  const [loading,setLoading]=useState(true)
  async function list() { const data=await api<Exam[]>('/api/exams');setExams(data);return data }
  useEffect(()=>{
    list().then(data=>{if(data[0])setId(data[0].id)}).catch(e=>setError(message(e))).finally(()=>setLoading(false))
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
        if(result.state==='queued'||result.state==='running'||result.state==='canceling')timer=setTimeout(poll,1250)
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
  async function cancel() {
    if (!exam) return
    try {setExam(await send<Exam>(path(exam.id,'/cancel'),{}));setPollVersion(v=>v+1)}
    catch(e) {setError(message(e))}
  }
  async function remove() {
    if (!exam || !window.confirm('Delete this examination, its video, bookmarks, and exports from this Mac?')) return
    try {
      const response=await fetch(path(exam.id),{method:'DELETE'})
      if (!response.ok) throw new Error('Could not delete the examination. Try again.')
      setExam(undefined);setId(null)
      const items=await list()
      setId(items[0]?.id ?? null)
    } catch(e) {setError(message(e))}
  }
  const stopped=exam?.state==='failed'||exam?.state==='cancelled'
  return <div className="app-shell">
    <nav className="sidebar" aria-label="Examination workspace">
      <button className="brand" onClick={()=>setUploading(false)}><span className="brand-icon"><ScanLine size={23}/></span><span>DentiLens <b>XR</b><small>Examination workspace</small></span></button>
      <button className="primary new-exam" onClick={()=>setUploading(true)}><Plus size={18}/> New examination</button>
      <div className="sidebar-caption"><span>Examinations</span><span>{exams.length}</span></div>
      <div className="exam-list">
        {exams.map(row=><button key={row.id} className={!uploading&&id===row.id?'active':''} aria-current={!uploading&&id===row.id?'true':undefined} onClick={()=>choose(row.id)}>
          <Film size={16}/><span><strong title={row.title}>{row.title}</strong><small>{examStates[row.state]} · {row.profile}</small></span>{id===row.id&&!uploading&&<ChevronRight size={15}/>}</button>)}
        {!exams.length&&!loading&&<small>No examinations yet</small>}
      </div>
      <div className="sidebar-bottom"><Activity size={17}/><span>Local workspace<small>Files stay on this Mac</small></span></div>
    </nav>
    <main>
      {error && <div role="alert" className="error page-error">{error}</div>}
      {uploading ? <UploadScreen back={()=>setUploading(false)} submit={created}/> :
        exam?.state==='ready' && exam.report ? <ExamReview key={exam.id} exam={exam} refresh={refresh} remove={remove}/> :
        exam ? <div className="processing panel">
          <h1>{stopped?exam.stage:exam.state==='queued'?'Analysis queued':exam.state==='canceling'?'Stopping analysis':'Analyzing video'}</h1><p>{exam.title}</p>
          <MediaMetrics info={exam.metadata} bytes={exam.bytes}/>
          {stopped ? <>{exam.error && <div className="error"><CircleAlert size={16}/> {exam.error}</div>}<div className="actions"><button className="primary" onClick={()=>void retry()}>Retry analysis</button>
              <button className="subtle" onClick={()=>void remove()}>Delete examination</button></div></>
            : <><div className="progress-track" role="progressbar" aria-label="Analysis progress" aria-valuenow={Math.round(exam.progress*100)}><span style={{width:`${exam.progress*100}%`}}/></div>
              <div className="progress-label"><span><LoaderCircle size={16} className="spin"/> {exam.state==='queued'?'Waiting to start':exam.stage}</span><strong>{Math.round(exam.progress*100)}%</strong></div>
              <button className="subtle stop-analysis" onClick={()=>void cancel()} disabled={exam.state==='canceling'}>Stop analysis</button>
              <video className="processing-preview" src={path(exam.id,'/preview')} controls playsInline preload="metadata"
                aria-label="Uploaded recording preview"/></>}
        </div> : loading || id ? <p role="status" className="quiet">Loading examinations…</p> :
          <div className="welcome"><span className="upload-icon"><ScanLine size={31}/></span><h1>Review a dental examination</h1>
            <p>Upload a video to find teeth and save bookmarks.</p>
            <button className="primary" onClick={()=>setUploading(true)}><Plus size={17}/> New examination</button></div>}
    </main>
  </div>
}
createRoot(document.getElementById('root')!).render(<App/>)
