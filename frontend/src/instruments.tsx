import { useEffect, useState } from 'react'
import { api, clock, send } from './api'
import type { Box } from './api'
import { RegionEditor } from './review-tools'

type Observation = {
  detection_id:string; class:string; time:number; confidence:number | null; polygon:number[][] | null; box:Box
}
type Appearance = { id:string; time:number; end:number; best:Observation }
type Review = { detection_id:string; label:string; note:string; decision:'confirmed'|'dismissed'; box?:Box | null }
type InstrumentReport = { device:string | null; samples:number; seconds:number; scanned:boolean; appearances:Appearance[]; reviews:Review[] }
type InstrumentResponse = {model_ready:boolean; report:InstrumentReport | null}

export function Instruments({examId, duration, currentTime, jump, refresh}: {
  examId:string; duration:number; currentTime:number; jump:(time:number)=>void; refresh:()=>Promise<void>
}) {
  const root = `/api/exams/${examId}`
  const [available,setAvailable]=useState<boolean>()
  const [report,setReport]=useState<InstrumentReport | null>(null)
  const [selected,setSelected]=useState<Appearance>()
  const [manual,setManual]=useState(false)
  const [time,setTime]=useState(0)
  const [redraw,setRedraw]=useState(false)
  const [box,setBox]=useState<Box | null>(null)
  const [label,setLabel]=useState('')
  const [note,setNote]=useState('')
  const [busy,setBusy]=useState(false)
  const [saving,setSaving]=useState(false)
  const [error,setError]=useState('')
  const [candidates,setCandidates]=useState(false)
  useEffect(()=>{
    const controller=new AbortController()
    api<InstrumentResponse>(root+'/instruments',{signal:controller.signal})
      .then(data=>{setAvailable(data.model_ready);setReport(data.report)})
      .catch(e=>{if(!controller.signal.aborted)setError(e.message)})
    return ()=>controller.abort()
  },[root])

  const reviews=new Map(report?.reviews.map(item=>[item.detection_id,item]))
  const current=selected && reviews.get(selected.best.detection_id)
  const visible=report?.appearances.filter(item=>reviews.get(item.best.detection_id)?.decision==='confirmed' ||
    (candidates && item.best.confidence!==null)) ?? []
  const sourceTime=selected?.best.time ?? time
  const still=root+'/still.jpg?time='+sourceTime

  function choose(item:Appearance) {
    const record=reviews.get(item.best.detection_id)
    setSelected(item);setManual(false);setRedraw(false);setBox(record?.box ?? null);setError('')
    setLabel(record?.label ?? '');setNote(record?.note ?? '')
  }
  function addManual() {
    setSelected(undefined);setManual(true);setRedraw(true);setBox(null);setError('')
    setTime(Math.min(currentTime,Math.max(0,duration-.01)));setLabel('');setNote('')
  }
  async function scan() {
    setBusy(true);setError('')
    try {setReport(await send<InstrumentReport>(root+'/instruments',{}));await refresh()}
    catch(e) {setError(e instanceof Error?e.message:'Instrument scan failed.')}
    finally {setBusy(false)}
  }
  async function review(decision:Review['decision']) {
    if(!selected && !manual) return
    setSaving(true);setError('')
    try {
      const record=await send<Review>(root+'/instrument-reviews',{
        detection_id:selected?.best.detection_id ?? null,time:manual?time:null,label,note,decision,box,
      })
      const data=await api<InstrumentResponse>(root+'/instruments')
      setReport(data.report)
      setManual(false);setRedraw(false)
      setSelected(data.report?.appearances.find(item=>item.best.detection_id===record.detection_id && record.decision==='confirmed'))
      await refresh()
    } catch(e) {setError(e instanceof Error?e.message:'Review could not be saved.')}
    finally {setSaving(false)}
  }
  return <section className="panel intelligence-panel instruments-page">
    <div className="section-heading"><h2>Instrument review</h2>
      <div className="actions">
        <button className="subtle compact-action" disabled={saving} onClick={addManual}>Mark instrument</button>
        <button className="subtle compact-action" disabled={!available || busy || !!report?.scanned} onClick={()=>void scan()}>
          {busy?'Scanning…':report?.scanned?'Scan complete':'Scan for candidates'}</button>
      </div>
    </div>
    {busy && <p role="status" className="quiet">Checking video frames for possible instrument regions…</p>}
    {error && <p className="error" role="alert">{error}</p>}
    {report?.scanned && <label className="toggle"><input type="checkbox" checked={candidates} disabled={saving}
      onChange={e=>{setCandidates(e.target.checked);if(!e.target.checked && current?.decision!=='confirmed')setSelected(undefined)}}/>
      <span>Show model candidates<small>{report.samples} frames sampled. Inspect each box before confirming.</small></span></label>}
    <div className="instrument-layout">
      {visible.length>0 ? <div>
        <div className="instrument-timeline" aria-label={candidates?'Instrument candidates and reviews':'Confirmed instruments'}>
          {visible.map(item=>{
            const saved=reviews.get(item.best.detection_id)
            const confirmed=saved?.decision==='confirmed'
            const at=confirmed?item.best.time:item.time
            return <button key={item.id} type="button" aria-label={`Open ${confirmed?'review':'candidate'} at ${clock(at,true)}`}
              onClick={()=>choose(item)} disabled={saving} aria-pressed={selected?.id===item.id}
              style={{left:at/duration*100+'%',width:Math.max(.7,confirmed?0:(item.end-item.time)/duration*100)+'%'}}/>
          })}
        </div>
        <div className="instrument-list">
          {visible.map(item=>{
            const saved=reviews.get(item.best.detection_id)
            return <button className="history-row" key={item.id} disabled={saving} onClick={()=>choose(item)} aria-pressed={selected?.id===item.id}>
              <time>{clock(item.best.time,true)}</time>
              <span>{saved?.decision==='confirmed'?saved.label || 'Reviewed instrument':'Model candidate'}</span>
              <small>{saved?.decision ?? 'Unreviewed'}</small>
            </button>
          })}
        </div>
      </div> : <p className="quiet">{candidates?'No model candidates.':'No confirmed instruments. Mark a region or review the model candidates.'}</p>}
      {(selected || manual) && <div className="instrument-evidence" key={selected?.id ?? 'manual'}>
        {manual && <label className="field">Source time (seconds)<input type="number" min={0} max={Math.max(0,duration-.01)} step=".01"
          value={time} disabled={saving} onChange={e=>{setBox(null);setTime(Math.max(0,Math.min(Number(e.target.value),duration-.01)))}}/></label>}
        {redraw ? <RegionEditor key={sourceTime} src={still} value={box} change={setBox}/> : selected && <div className="instrument-image">
          <img src={still} alt={`Examination frame at ${clock(sourceTime,true)}`}/>
          <svg viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
            {box ? <rect x={box[0]*100} y={box[1]*100} width={(box[2]-box[0])*100} height={(box[3]-box[1])*100} vectorEffect="non-scaling-stroke"/> :
              <polygon points={selected.best.polygon?.map(p=>p.map(v=>v*100).join(',')).join(' ')} vectorEffect="non-scaling-stroke"/>}
          </svg>
        </div>}
        <p className="quiet">{clock(sourceTime,true)}
          {selected && current?.decision!=='confirmed' && selected.best.confidence!==null &&
            ` · Model guess: ${selected.best.class} (${Math.round(selected.best.confidence*100)}%)`}</p>
        <div className="actions">
          <button className="subtle compact-action" onClick={()=>jump(sourceTime)}>Open source video</button>
          {!manual && !redraw && <button className="subtle compact-action" onClick={()=>setRedraw(true)}>Redraw box</button>}
        </div>
        <label className="field">Reviewed label<input maxLength={100} value={label} disabled={saving} onChange={e=>setLabel(e.target.value)}/></label>
        <label className="field">Note<textarea maxLength={500} rows={2} value={note} disabled={saving} onChange={e=>setNote(e.target.value)}/></label>
        <div className="actions">
          <button className="primary compact-action" disabled={saving || !label.trim() || (redraw && !box)}
            onClick={()=>void review('confirmed')}>{current?.decision==='confirmed'?'Save changes':'Confirm'}</button>
          {!manual && <button className="subtle compact-action" disabled={saving || current?.decision==='dismissed'}
            onClick={()=>void review('dismissed')}>Dismiss</button>}
        </div>
      </div>}
    </div>
  </section>
}
