import { useEffect, useState } from 'react'
import { api, clock } from './api'
import type { Exam, Finding, Review } from './api'

type Span = {frame:number; last_frame:number; time:number; end:number; count:number}
type Observation = {detection_id:string; frame:number; time:number; label:string; confidence?:number; number_score?:number; type_score?:number}
type History = {
  tracks: {track_id:string; suggested_numbers:string[]; confirmed_numbers:string[]; frames:Observation[];
    views:Span[]; allFrames?:Observation[]; evidence:Finding[]; visible_seconds:number; reviews:Review[]}[];
  other_reviews:Review[];
}
type Result = {kind:string; id:string; label:string; time:number; end?:number; image?:string; confidence?:number; note?:string; status:string}
type Search = {results:Result[]; total?:number; message:string}
type Quality = {frames:number; cues:{label:string; frames:number; appearances:Span[]}[]}

function stamp(time:number) { return clock(time,true) }

export function ToothHistory({id, fdi, jump}: {id:string; fdi:string; jump:(time:number)=>void}) {
  const [history,setHistory]=useState<History>()
  const [error,setError]=useState('')
  const [pages,setPages]=useState<Record<string,number>>({})
  const [showTrack,setShowTrack]=useState('')
  useEffect(()=>{
    const controller=new AbortController()
    api<History>(`/api/exams/${id}/tooth-history?fdi=${fdi}`,{signal:controller.signal})
      .then(setHistory).catch(e=>{if(!controller.signal.aborted) setError(e.message)})
    return ()=>controller.abort()
  },[id,fdi])
  useEffect(()=>{
    if(!showTrack) return
    const controller=new AbortController()
    api<History>(`/api/exams/${id}/tooth-history?track=${showTrack}`,{signal:controller.signal})
      .then(data=>setHistory(current=>current && ({...current,tracks:current.tracks.map(track=>
        track.track_id===showTrack && data.tracks.length ? {...track,allFrames:data.tracks[0].frames} : track)})))
      .catch(e=>{if(!controller.signal.aborted)setError(e.message)})
    return ()=>controller.abort()
  },[id,showTrack])
  if (error) return <p role="alert" className="error">{error}</p>
  if (!history) return <p role="status" className="quiet">Loading tooth history…</p>
  if (!history.tracks.length && !history.other_reviews.length) return <p className="quiet">No numbered observations or reviews tied to this position.</p>
  return <div className="tooth-history">
    {history.tracks.map(track=>{
      const start=(pages[track.track_id] ?? 0)*40
      const allFrames=track.allFrames
      const frames=allFrames ?? track.frames
      return <section key={track.track_id}>
        <h4>Track {track.track_id}</h4>
        {track.confirmed_numbers.length>0 && <p>Reviewed FDI {track.confirmed_numbers.join(', ')}</p>}
        <p>{track.frames.length} numbered frames · {track.visible_seconds} s of numbered observations</p>
        {track.views.length>0 && <p>First {stamp(track.views[0].time)} · Last {stamp(track.views.at(-1)!.end)}</p>}
        <div className="actions">{track.views.map(span=><button className="subtle compact-action" key={span.frame} onClick={()=>jump(span.time)}>
          {stamp(span.time)}{span.count>1 && '–'+stamp(span.end)}</button>)}</div>
        {track.reviews.map(review=><p key={review.id}><button className="inline-jump" onClick={()=>jump(review.time)}>{stamp(review.time)}</button>
          {' · '}{review.decision}{review.note && ' · '+review.note}</p>)}
        <details><summary>Frame observations</summary>
          {!allFrames && <button className="subtle compact-action" onClick={()=>{setShowTrack(track.track_id);setPages(current=>({...current,[track.track_id]:0}))}}>Load the full track</button>}
          <div className="frame-history">
            {frames.slice(start,start+40).map(row=><button className="history-row" key={row.detection_id} onClick={()=>jump(row.time)}>
              <time>{stamp(row.time)}</time><span>{row.label}</span>
              <small>{typeof row.confidence!=='number'?'':`Model ${Math.round(row.confidence*100)}%`}
                {typeof row.number_score==='number' && ` · FDI ${Math.round(row.number_score*100)}%`}
                {typeof row.type_score==='number' && ` · Type ${Math.round(row.type_score*100)}%`}</small></button>)}
          </div>
          {frames.length>40 && <div className="actions">
            <button className="subtle compact-action" disabled={!start} onClick={()=>setPages({...pages,[track.track_id]:start/40-1})}>Previous frames</button>
            <button className="subtle compact-action" disabled={start+40>=frames.length} onClick={()=>setPages({...pages,[track.track_id]:start/40+1})}>Next frames</button>
          </div>}
        </details>
      </section>
    })}
    {history.other_reviews.map(review=><p key={review.id}><button className="inline-jump" onClick={()=>jump(review.time)}>{stamp(review.time)}</button>
      {' · '}{review.decision}{review.note && ' · '+review.note}</p>)}
  </div>
}

export function ExamIntelligence({exam,jump}: {exam:Exam; jump:(time:number)=>void}) {
  const [query,setQuery]=useState('')
  const [answer,setAnswer]=useState<Search>()
  const [quality,setQuality]=useState<Quality>()
  const [busy,setBusy]=useState(false)
  const [scanning,setScanning]=useState(false)
  const [searchError,setSearchError]=useState('')
  const [qualityError,setQualityError]=useState('')
  const endpoint=`/api/exams/${exam.id}`
  async function search(q:string) {
    const term=q.trim()
    if(!term || busy) return
    setQuery(term);setBusy(true);setSearchError('');setAnswer(undefined)
    try {setAnswer(await api<Search>(endpoint+'/intelligence?q='+encodeURIComponent(term)))}
    catch(e) {setSearchError(e instanceof Error?e.message:'Search failed. Try again.')}
    finally {setBusy(false)}
  }
  async function scan() {
    setScanning(true);setQualityError('')
    try {setQuality(await api<Quality>(endpoint+'/quality'))}
    catch(e) {setQualityError(e instanceof Error?e.message:'Scan failed. Try again.')}
    finally {setScanning(false)}
  }
  return <div className="intelligence-page">
    <section className="panel intelligence-panel">
      <div className="section-heading"><h2>Ask DentiLens</h2>
        <a className="subtle compact-action" href={endpoint+'/evidence-index'}>Download evidence index</a></div>
      <form className="command" onSubmit={e=>{e.preventDefault();void search(query)}}>
        <input aria-label="Search this examination" value={query} maxLength={160} onChange={e=>setQuery(e.target.value)} placeholder="Search by tooth, timestamp, or note"/>
        <button className="primary" disabled={busy || !query.trim()}>{busy?'Searching…':'Search'}</button>
      </form>
      <div className="actions">{['Show tooth 21','Show canine appearances','Show notes'].map(q=><button className="subtle compact-action" key={q}
        disabled={busy} onClick={()=>void search(q)}>{q}</button>)}</div>
      {searchError && <p role="alert" className="error">{searchError}</p>}
      {answer?.message && <p role="status" className="quiet">{answer.message}</p>}
      {answer && !!answer.results.length && <p role="status" className="quiet">{answer.results.length} of {answer.total} results</p>}
      {answer?.results.map(item=><button className="chart-finding" key={item.kind+item.id} onClick={()=>jump(item.time)}>
        {item.image && <img src={endpoint+'/images/'+item.image} alt="" loading="lazy"/>}
        <span><strong>{item.label}</strong><small>{stamp(item.time)} · {item.status}
          {typeof item.confidence==='number' && ` · Model ${Math.round(item.confidence*100)}%`}</small>
          {item.note && <small>{item.note}</small>}</span>
      </button>)}
    </section>
    <section className="panel intelligence-panel">
      <div className="section-heading"><h2>Recording quality</h2>
        <button className="subtle compact-action" disabled={scanning} onClick={()=>void scan()}>{scanning?'Scanning…':'Scan recording'}</button></div>
      <p className="quiet">Scan for dark frames, bright frames, low detail, and large frame changes.</p>
      {qualityError && <p role="alert" className="error">{qualityError}</p>}
      {quality && <p role="status" className="quiet">{quality.frames} frames checked. {quality.cues.length===0?'No quality cues flagged.':''}</p>}
      {quality?.cues.map(cue=><details key={cue.label} className="quality-cue">
        <summary>{cue.label} · {cue.frames} frames</summary>
        <div className="actions">{cue.appearances.map(span=><button className="subtle compact-action" key={span.frame} onClick={()=>jump(span.time)}>
          {stamp(span.time)}{span.count>1 && '–'+stamp(span.end)}</button>)}</div>
      </details>)}
    </section>
  </div>
}
