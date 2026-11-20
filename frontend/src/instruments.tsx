import { useEffect, useState } from 'react'
import { api, clock, send } from './api'

type Observation = { detection_id:string; track_id:string | null; class:string; time:number; confidence:number; polygon:number[][] }
type Appearance = { id:string; label:string; time:number; end:number; samples:number; best:Observation }
type Review = { detection_id:string; label:string; note:string; decision:'confirmed'|'dismissed' }
type InstrumentReport = { device:string; samples:number; seconds:number; appearances:Appearance[]; reviews:Review[] }

export function Instruments({examId, duration, jump, refresh}: {examId:string; duration:number; jump:(time:number)=>void; refresh:()=>Promise<void>}) {
  const root = `/api/exams/${examId}`
  const [available,setAvailable]=useState<boolean>()
  const [report,setReport]=useState<InstrumentReport | null>(null)
  const [selected,setSelected]=useState<Appearance>()
  const [label,setLabel]=useState('')
  const [note,setNote]=useState('')
  const [busy,setBusy]=useState(false)
  const [saving,setSaving]=useState(false)
  const [error,setError]=useState('')
  const [filter,setFilter]=useState('')
  useEffect(()=>{
    const controller=new AbortController()
    api<{model_ready:boolean;report:InstrumentReport | null}>(root+'/instruments',{signal:controller.signal})
      .then(data=>{setAvailable(data.model_ready);setReport(data.report)})
      .catch(e=>{if(!controller.signal.aborted)setError(e.message)})
    return ()=>controller.abort()
  },[root])

  function choose(item:Appearance) {
    const record=report?.reviews.find(review=>review.detection_id===item.best.detection_id)
    setSelected(item);setLabel(record?.label || item.label);setNote(record?.note ?? '')
  }
  async function scan() {
    setBusy(true);setError('')
    try {setReport(await send<InstrumentReport>(root+'/instruments',{}));await refresh()}
    catch(e) {setError(e instanceof Error?e.message:'Instrument scan failed.')}
    finally {setBusy(false)}
  }
  async function review(decision:Review['decision']) {
    if(!selected || !report) return
    setSaving(true);setError('')
    try {
      const record=await send<Review>(root+'/instrument-reviews',{detection_id:selected.best.detection_id,label,note,decision})
      setReport({...report,reviews:[...report.reviews.filter(item=>item.detection_id!==record.detection_id),record]})
      await refresh()
    } catch(e) {setError(e instanceof Error?e.message:'Review could not be saved.')}
    finally {setSaving(false)}
  }
  const reviews=new Map(report?.reviews.map(item=>[item.detection_id,item]))
  const current=selected && reviews.get(selected.best.detection_id)
  const classes=[...new Set(report?.appearances.map(item=>item.label))]
  const visible=report?.appearances.filter(item=>!filter || item.label===filter) ?? []
  return <section className="panel intelligence-panel instruments-page">
    <div className="section-heading"><h2>Instrument appearances</h2>
      <button className="subtle compact-action" disabled={!available || busy || !!report} onClick={()=>void scan()}>
        {busy?'Scanning…':report?'Scan complete':'Scan recording'}</button></div>
    {available===false && !report && <p className="quiet">Install the evaluated instrument checkpoint to scan this recording.</p>}
    {available===true && !report && !busy && <p className="quiet">Scan for instrument suggestions using the local model.</p>}
    {busy && <p role="status" className="quiet">Checking video frames for instruments…</p>}
    {error && <p className="error" role="alert">{error}</p>}
    {report && <>
      <p className="quiet">{report.samples} sampled frames · {report.seconds} s on {report.device.toUpperCase()} · Experimental model</p>
      {report.appearances.length===0 ? <p className="quiet">No instrument suggestions met the threshold in sampled frames.</p> :
        <div className="instrument-layout">
          <div>
            <label className="field">Filter instruments<select value={filter} onChange={e=>setFilter(e.target.value)}>
              <option value="">All classes</option>{classes.map(name=><option key={name}>{name}</option>)}
            </select></label>
            <div className="instrument-timeline" aria-label="Instrument observations">
              {visible.map(item=><button key={item.id} type="button" title={item.label+' · '+clock(item.time,true)}
                aria-label={item.label+' at '+clock(item.time,true)} onClick={()=>choose(item)} aria-pressed={selected?.id===item.id}
                style={{left:item.time/duration*100+'%',width:Math.max(.7,(item.end-item.time)/duration*100)+'%'}}/>)}
            </div>
            <div className="instrument-list">
            {visible.map(item=><button className="history-row" key={item.id} onClick={()=>choose(item)} aria-pressed={selected?.id===item.id}>
              <time>{clock(item.time,true)}{item.end!==item.time && '–'+clock(item.end,true)}</time>
              <span>{reviews.get(item.best.detection_id)?.label || item.label}</span><small>{reviews.get(item.best.detection_id)?.decision ?? 'Suggested'}</small>
            </button>)}
            </div>
          </div>
          {selected ? <div className="instrument-evidence" key={selected.id}>
            <div className="instrument-image">
              <img src={root+'/still.jpg?time='+selected.best.time} alt={`Examination frame at ${clock(selected.best.time,true)}`}/>
              <svg viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
                <polygon points={selected.best.polygon.map(p=>p.map(v=>v*100).join(',')).join(' ')} vectorEffect="non-scaling-stroke"/>
              </svg>
            </div>
            <p className="quiet">{clock(selected.best.time,true)} · Model {Math.round(selected.best.confidence*100)}% · {current?.decision ?? 'Suggested'}</p>
            <div className="actions"><button className="subtle compact-action" onClick={()=>jump(selected.best.time)}>Open source video</button></div>
            <label className="field">Reviewed label<input maxLength={100} value={label} disabled={saving} onChange={e=>setLabel(e.target.value)}/></label>
            <label className="field">Note<textarea maxLength={500} rows={2} value={note} disabled={saving} onChange={e=>setNote(e.target.value)}/></label>
            <div className="actions">
              <button className="primary compact-action" disabled={saving} onClick={()=>void review('confirmed')}>{current?.decision==='confirmed'?'Save changes':'Confirm'}</button>
              <button className="subtle compact-action" disabled={saving || current?.decision==='dismissed'} onClick={()=>void review('dismissed')}>Dismiss</button>
            </div>
          </div> : <p className="quiet">Select an appearance to inspect its source frame.</p>}
        </div>}
    </>}
  </section>
}
