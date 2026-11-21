import { useEffect, useRef, useState } from 'react'
import { api, clock, send } from './api'

type Segment = {id:string; start:number; end:number; text:string; note?:string; fdi_number?:string | null; status?:'reviewed'|'removed'}
type Transcript = { state:'ready'|'no_audio'|'silent'; segments:Segment[]; language:string; seconds:number }
type Event = {id:string; kind:string; start:number; end:number; label:string; status:string}
type Chapter = {id:string; title:string; start:number; end:number; thumbnail:number; note:string; reviewed:boolean; events?:Event[]}
type Storyboard = { chapters:Chapter[]; original_chapters:Chapter[] }

function useSource(id:string) {
  const video = useRef<HTMLVideoElement>(null)
  const [time,setTime]=useState(0)
  const base=`/api/exams/${id}`
  function seek(at:number) {
    if (video.current) {video.current.currentTime=at;video.current.pause()}
    setTime(at)
  }
  const player=<video ref={video} controls playsInline preload="metadata" src={base+'/media/source.mp4'}
    onTimeUpdate={event=>setTime(event.currentTarget.currentTime)}/>
  return {base,time,seek,player}
}

function save<T>(url:string, body:unknown) {
  return api<T>(url,{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})
}

function VoiceNote({base,time,duration,language}: {base:string;time:number;duration:number;language:string}) {
  const active=useRef<MediaRecorder | null>(null)
  const [recording,setRecording]=useState(false)
  const [busy,setBusy]=useState(false)
  const [draft,setDraft]=useState<{time:number; note:string; fdi_number:string}>()
  const [message,setMessage]=useState('')
  useEffect(()=>()=> {
    const recorder=active.current
    if(!recorder) return
    recorder.onstop=null
    if(recorder.state==='recording') recorder.stop()
    recorder.stream.getTracks().forEach(track=>track.stop())
  },[])
  async function start() {
    setBusy(true);setMessage('')
    let stream:MediaStream | undefined
    try {
      stream=await navigator.mediaDevices.getUserMedia({audio:true})
      const recorder=new MediaRecorder(stream)
      active.current=recorder
      const bookmarkTime=Math.max(0,Math.min(time,duration-.001))
      const chunks:Blob[]=[]
      recorder.ondataavailable=event=>{if(event.data.size) chunks.push(event.data)}
      recorder.onstop=async ()=>{
        recorder.stream.getTracks().forEach(track=>track.stop())
        active.current=null
        setRecording(false);setBusy(true)
        try{
          const body=new FormData()
          body.append('recording',new Blob(chunks,{type:recorder.mimeType}),'note.webm')
          body.append('language',language)
          const {text}=await api<{text:string}>(base+'/voice-note',{method:'POST',body})
          setDraft({time:bookmarkTime,note:text,fdi_number:''})
          if(!text) setMessage('No speech recognized. You can type the note.')
        } catch(e){setMessage(e instanceof Error?e.message:'The voice note could not be transcribed.')}
        finally{setBusy(false)}
      }
      recorder.start()
      setRecording(true)
    } catch(e){
      stream?.getTracks().forEach(track=>track.stop())
      setMessage(e instanceof Error?e.message:'The microphone could not be opened.')
    } finally{setBusy(false)}
  }
  async function saveNote() {
    if(!draft) return
    setBusy(true);setMessage('')
    try {
      await send(base+'/reviews',{...draft,label:'Voice note',fdi_number:draft.fdi_number || null})
      setDraft(undefined);setMessage('Voice note saved to examination bookmarks.')
    } catch(e){setMessage(e instanceof Error?e.message:'The note could not be saved.')}
    finally{setBusy(false)}
  }
  return <div className="event-editor">
    <div className="actions">
      <button type="button" className="subtle compact-action" disabled={busy || !!draft}
        onClick={()=>recording?active.current?.stop():void start()}>{recording?'Stop and transcribe':'Record voice note'}</button>
    </div>
    {message && <p role="status" className="quiet">{message}</p>}
    {draft && <>
      <p className="quiet">Bookmark at {clock(draft.time,true)}</p>
      <label className="field">Note<textarea maxLength={500} value={draft.note} onChange={e=>setDraft({...draft,note:e.target.value})}/></label>
      <label className="field">FDI number (optional)<input maxLength={2} value={draft.fdi_number} onChange={e=>setDraft({...draft,fdi_number:e.target.value})}/></label>
      <div className="actions">
        <button className="primary compact-action" disabled={busy || !draft.note.trim()}
          onClick={()=>void saveNote()}>Save voice note</button>
        <button className="subtle compact-action" disabled={busy} onClick={()=>setDraft(undefined)}>Cancel</button>
      </div>
    </>}
  </div>
}

export function AudioIntelligence({id,duration}: {id:string;duration:number}) {
  const {base,time,seek,player}=useSource(id)
  const [ready,setReady]=useState(false)
  const [transcript,setTranscript]=useState<Transcript | null>(null)
  const [language,setLanguage]=useState('auto')
  const [editing,setEditing]=useState<Segment>()
  const [busy,setBusy]=useState(false)
  const [error,setError]=useState('')
  useEffect(()=>{
    api<{model_ready:boolean;transcript:Transcript | null}>(base+'/transcript').then(r=>{
      setReady(r.model_ready);setTranscript(r.transcript)
    }).catch(e=>setError(e.message))
  },[base])
  async function run() {
    setBusy(true);setError('')
    try {setTranscript(await send<Transcript>(base+'/transcript',{language}))}
    catch(e) {setError(e instanceof Error?e.message:'Transcription failed.')}
    finally {setBusy(false)}
  }
  async function edit(status:'reviewed' | 'removed') {
    if(!editing) return
    setBusy(true);setError('')
    try {
      setTranscript(await save<Transcript>(base+'/transcript/'+editing.id,{...editing,status}))
      setEditing(undefined)
    } catch(e){setError(e instanceof Error?e.message:'The transcript could not be saved.')}
    finally{setBusy(false)}
  }
  return <section className="panel intelligence-panel">
    <h2>Audio transcript</h2>
    <div className="event-workspace">
      {player}
      <div className="event-details">
        {!transcript && <>
          <label className="field">Language<select value={language} disabled={busy} onChange={e=>setLanguage(e.target.value)}>
            <option value="auto">Detect automatically</option><option value="en">English</option><option value="hi">Hindi</option>
            <option value="es">Spanish</option><option value="fr">French</option><option value="de">German</option>
          </select></label>
          <button className="primary compact-action" disabled={busy} onClick={()=>void run()}>{busy?'Transcribing locally…':'Transcribe audio'}</button>
          <p className="quiet">{ready?'Whisper runs on this Mac.':'Speech recognition needs the local Whisper model. Silent recordings still return an audio status.'}</p>
        </>}
        {error && <p className="error" role="alert">{error}</p>}
        {transcript && (transcript.state==='no_audio'?<p className="quiet">This recording has no audio track.</p>:
          transcript.state==='silent'?<p className="quiet">The audio track is silent.</p>:
          <p className="quiet">{transcript.segments.filter(s=>s.status!=='removed').length} segments · local Whisper · {transcript.seconds} s</p>)}
        {transcript?.segments.filter(s=>s.status!=='removed').map(s=><div key={s.id} className="event-row" data-active={time>=s.start && time<s.end}>
          <button className="inline-jump" onClick={()=>seek(s.start)}>{clock(s.start,true)}–{clock(s.end,true)}</button>
          <p>{s.text}</p><small>{s.status ?? 'Unreviewed'}{s.fdi_number && ` · FDI ${s.fdi_number}`}</small>
          {s.note && <p className="quiet">{s.note}</p>}
          <button className="subtle compact-action" onClick={()=>setEditing({...s})}>Edit segment</button>
        </div>)}
        {editing && <div className="event-editor">
          <label className="field">Text<textarea value={editing.text} maxLength={4000} onChange={e=>setEditing({...editing,text:e.target.value})}/></label>
          <div className="event-times">
            <label className="field">Start (seconds)<input type="number" min={0} max={duration} step=".01" value={editing.start} onChange={e=>setEditing({...editing,start:Number(e.target.value)})}/></label>
            <label className="field">End (seconds)<input type="number" min={0} max={duration} step=".01" value={editing.end} onChange={e=>setEditing({...editing,end:Number(e.target.value)})}/></label>
          </div>
          <label className="field">Note<textarea value={editing.note ?? ''} maxLength={500} onChange={e=>setEditing({...editing,note:e.target.value})}/></label>
          <label className="field">FDI number (optional)<input value={editing.fdi_number ?? ''} maxLength={2}
            onChange={e=>setEditing({...editing,fdi_number:e.target.value || null})}/></label>
          <div className="actions">
            <button className="primary compact-action" disabled={busy || !editing.text.trim()} onClick={()=>void edit('reviewed')}>Save reviewed text</button>
            <button className="subtle compact-action" disabled={busy} onClick={()=>void edit('removed')}>Remove segment</button>
            <button className="subtle compact-action" onClick={()=>setEditing(undefined)}>Cancel</button>
          </div>
        </div>}
        {ready && <VoiceNote base={base} time={time} duration={duration} language={language}/>}
      </div>
    </div>
  </section>
}

export function ExaminationStoryboard({id,duration}: {id:string;duration:number}) {
  const {base,time,seek,player}=useSource(id)
  const [story,setStory]=useState<Storyboard | null>(null)
  const [editing,setEditing]=useState<Chapter>()
  const [busy,setBusy]=useState(false)
  const [error,setError]=useState('')
  const root=base+'/storyboard'
  useEffect(()=>{
    api<{storyboard:Storyboard | null}>(root).then(r=>setStory(r.storyboard)).catch(e=>setError(e.message))
  },[root])
  async function build() {
    setBusy(true);setError('')
    try{setStory(await send<Storyboard>(root,{}))}
    catch(e){setError(e instanceof Error?e.message:'The storyboard could not be built.')}
    finally{setBusy(false)}
  }
  async function update(operation:'save'|'split'|'merge'|'remove') {
    if(!story || !editing) return
    setBusy(true);setError('')
    try {
      let parts=story.chapters.filter(row=>row.id!==editing.id)
      let edited={...editing,thumbnail:Math.max(editing.start,Math.min(editing.thumbnail,editing.end-.001))}
      if(operation==='split'){
        const split=(edited.start+edited.end)/2
        parts.push({...edited,end:split,thumbnail:edited.start},
                   {...edited,id:'chapter-'+crypto.randomUUID(),start:split,thumbnail:split,reviewed:false})
      } else if (operation==='merge') {
        const next=story.chapters.find(row=>row.start>=edited.end && row.id!==edited.id)
        if(!next) throw new Error('There is no following chapter to merge.')
        parts=parts.filter(row=>row.id!==next.id)
        parts.push({...edited,end:next.end})
      } else if (operation==='save') parts.push(edited)
      setStory(await save<Storyboard>(root,{chapters:parts}));setEditing(undefined)
    } catch(e){setError(e instanceof Error?e.message:'The chapter could not be saved.')}
    finally{setBusy(false)}
  }
  return <section className="panel intelligence-panel">
    <div className="section-heading"><h2>Examination storyboard</h2>{!story &&
      <button className="primary compact-action" onClick={()=>void build()} disabled={busy}>{busy?'Building…':'Build storyboard'}</button>}</div>
    <div className="story-player">{player}</div>
    {error && <p role="alert" className="error">{error}</p>}
    {story && <div className="actions event-download">
      <a className="subtle compact-action" href={base+'/storyboard-export?format=json'}>Download JSON</a>
      <a className="subtle compact-action" href={base+'/storyboard-export?format=csv'}>Download CSV</a>
    </div>}
    <div className="story-grid">{story?.chapters.map(item=><article key={item.id} data-active={time>=item.start && time<item.end}>
      <button className="story-thumb" onClick={()=>seek(item.start)} aria-label={`Play chapter at ${clock(item.start,true)}`}>
        <img alt="" loading="lazy" src={base+'/still.jpg?time='+item.thumbnail}/><strong>{clock(item.start,true)}–{clock(item.end,true)}</strong>
      </button>
      <h3>{item.title}</h3>
      <p>{item.reviewed?'Reviewed':'Draft'} · {Math.round((item.end-item.start)*10)/10} s</p>
      {!!item.note && <p>{item.note}</p>}
      {item.events?.slice(0,4).map(event=><button key={event.id} className="event-link" onClick={()=>seek(event.start)}>{clock(event.start,true)} · {event.label}</button>)}
      <button className="subtle compact-action" onClick={()=>setEditing({...item})}>Edit chapter</button>
    </article>)}</div>
    {editing && <div className="event-editor">
      <label className="field">Chapter title<input value={editing.title} maxLength={120} onChange={e=>setEditing({...editing,title:e.target.value})}/></label>
      <label className="field">Note<textarea value={editing.note} maxLength={500} onChange={e=>setEditing({...editing,note:e.target.value})}/></label>
      <div className="event-times">
        <label className="field">Start (seconds)<input type="number" min={0} max={duration} step=".01" value={editing.start} onChange={e=>setEditing({...editing,start:Number(e.target.value)})}/></label>
        <label className="field">End (seconds)<input type="number" min={0} max={duration} step=".01" value={editing.end} onChange={e=>setEditing({...editing,end:Number(e.target.value)})}/></label>
      </div>
      <label className="toggle"><input type="checkbox" checked={editing.reviewed} onChange={e=>setEditing({...editing,reviewed:e.target.checked})}/> Reviewed</label>
      <div className="actions">
        <button className="primary compact-action" disabled={busy || !editing.title.trim()} onClick={()=>void update('save')}>Save chapter</button>
        <button className="subtle compact-action" disabled={busy} onClick={()=>void update('split')}>Split at midpoint</button>
        <button className="subtle compact-action" disabled={busy} onClick={()=>void update('merge')}>Merge with next</button>
        <button className="subtle compact-action" disabled={busy} onClick={()=>void update('remove')}>Remove chapter</button>
        <button className="subtle compact-action" onClick={()=>setEditing(undefined)}>Cancel</button>
      </div>
    </div>}
  </section>
}
