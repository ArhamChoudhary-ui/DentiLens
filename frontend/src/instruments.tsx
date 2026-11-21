import { useEffect, useState } from 'react'
import { api, clock, send } from './api'
import type { Box } from './api'
import { RegionEditor } from './review-tools'

type Prediction = {label:string;confidence:number;accepted:boolean;model:string;alternatives:{label:string;confidence:number}[]}
type IdentityInfo = {model?:string;specialists?:IdentityInfo[];thresholds?:Record<string,number>;error?:string;scores?:Record<string,{crops:number;abstention:number}>}
type Observation = {identity?:Prediction | null;
  detection_id:string; class:string; time:number; confidence:number | null; polygon:number[][] | null; box:Box
}
type Appearance = { samples?:number; id:string; time:number; end:number; best:Observation }
type Review = { review_action?:string; detection_id:string; label:string; note:string; decision:'confirmed'|'dismissed'; box?:Box | null; fdi_number?:string | null; visibility?:'visible' | 'partial' | null }
type InstrumentReport = {identity_model?:string | null; device:string | null; samples:number; seconds:number; scanned:boolean; experimental_samples?:number | null; appearances:Appearance[]; reviews:Review[] }
type TrackedFrame = {frame:number; time:number; box:Box}
type InstrumentResponse = {identity?:IdentityInfo;model_ready:boolean; experimental_ready?:boolean; report:InstrumentReport | null}

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
  const [fdi,setFdi]=useState('')
  const [visibility,setVisibility]=useState('')
  const [anchor,setAnchor]=useState<string>()
  const [busy,setBusy]=useState(false)
  const [saving,setSaving]=useState(false)
  const [error,setError]=useState('')
  const [candidates,setCandidates]=useState(false)
  const [experimentalReady,setExperimentalReady]=useState(false)
  const [identity,setIdentity]=useState<IdentityInfo>()
  const [identifying,setIdentifying]=useState(false)
  const [tracking,setTracking]=useState(false)
  const [tracked,setTracked]=useState<TrackedFrame[] | null>(null)
  const [trackedFrame,setTrackedFrame]=useState<TrackedFrame>()
  useEffect(()=>{
    const controller=new AbortController()
    api<InstrumentResponse>(root+'/instruments',{signal:controller.signal})
      .then(data=>{setAvailable(data.model_ready);setExperimentalReady(!!data.experimental_ready);setIdentity(data.identity);setReport(data.report);setCandidates(data.report?.experimental_samples != null)})
      .catch(e=>{if(!controller.signal.aborted)setError(e.message)})
    return ()=>controller.abort()
  },[root])

  const reviews=new Map(report?.reviews.map(item=>[item.detection_id,item]))
  const current=selected && reviews.get(selected.best.detection_id)
  const visible=report?.appearances.filter(item=>reviews.get(item.best.detection_id)?.decision==='confirmed' ||
    (candidates && item.best.confidence!==null)) ?? []
  const sourceTime=trackedFrame?.time ?? selected?.best.time ?? time
  const previewBox=trackedFrame?.box ?? box
  const still=root+'/still.jpg?time='+sourceTime

  function clearTrack() {
    setTracked(null)
    setTrackedFrame(undefined)
  }
  function choose(item:Appearance) {
    clearTrack()
    const record=reviews.get(item.best.detection_id)
    setSelected(item);setManual(false);setRedraw(false);setBox(record?.box ?? null);setError('')
    setLabel(record?.label ?? (item.best.identity?.accepted ? item.best.identity.label : ''));setNote(record?.note ?? '')
    setFdi(record?.fdi_number ?? '');setVisibility(record?.visibility ?? '');setAnchor(undefined)
  }
  function addManual() {
    clearTrack()
    setSelected(undefined);setManual(true);setRedraw(true);setBox(null);setError('')
    setTime(Math.min(currentTime,Math.max(0,duration-.01)));setLabel('');setNote('');setFdi('');setVisibility('');setAnchor(undefined)
  }
  async function scan(experimental = false) {
    setBusy(true);setError('')
    try {
      setReport(await send<InstrumentReport>(root+'/instruments'+(experimental?'?experimental=true':''),{}))
      setCandidates(true)
      if(experimental && identity?.model) {
        setIdentifying(true)
        setReport(await send<InstrumentReport>(root+'/instrument-identities',{}))
      }
      await refresh()
    }
    catch(e) {setError(e instanceof Error?e.message:'Instrument scan failed.')}
    finally {setBusy(false);setIdentifying(false)}
  }
  async function identify() {
    setIdentifying(true);setError('')
    try {
      setReport(await send<InstrumentReport>(root+'/instrument-identities',{}))
      setSelected(undefined);setCandidates(true);await refresh()
    } catch(e) {setError(e instanceof Error?e.message:'Instrument identification failed.')}
    finally{setIdentifying(false)}
  }
  async function follow() {
    if(!selected) return
    clearTrack()
    setTracking(true);setError('')
    try {
      const data=await send<{frames:TrackedFrame[]}>(root+'/instrument-reviews/'+selected.best.detection_id+'/track',{})
      setTracked(data.frames)
    } catch(e) {setError(e instanceof Error?e.message:'Could not follow the instrument.')}
    finally {setTracking(false)}
  }
  function useTrackedFrame() {
    if(!trackedFrame || !selected) return
    setTime(trackedFrame.time);setBox(trackedFrame.box);setAnchor(selected.best.detection_id)
    setSelected(undefined);setManual(true);setRedraw(true);setNote('');clearTrack()
  }
  async function review(decision:Review['decision']) {
    if(!selected && !manual) return
    setSaving(true);setError('')
    try {
      const record=await send<Review>(root+'/instrument-reviews',{
        detection_id:selected?.best.detection_id ?? null,time:manual?time:null,label,note,decision,box,
        fdi_number:fdi || null,visibility:visibility || null,tracked_from:anchor ?? null,
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
        <button className="subtle compact-action" disabled={saving || tracking} onClick={addManual}>Mark instrument</button>
        <button className="subtle compact-action" disabled={!available || busy || !!report?.scanned} onClick={()=>void scan()}>
          {busy && !identifying?'Scanning…':report?.scanned?'Still-photo scan complete':'Scan with still-photo model'}</button>
        {experimentalReady && <button className="subtle compact-action" disabled={busy || report?.experimental_samples != null}
          onClick={()=>void scan(true)}>{report?.experimental_samples != null?'Video scan complete':'Scan with experimental video model'}</button>}
      </div>
    </div>
    {report?.experimental_samples != null && <div className="actions">
      <button className="subtle compact-action" disabled={busy || identifying || !identity?.model || saving}
        onClick={()=>void identify()}>{identifying?'Identifying…':report.identity_model?'Check saved identities':'Identify instruments'}</button>
      {report.identity_model && <small className="quiet">Identification complete · {report.identity_model}</small>}
    </div>}
    <details>
      <summary>Instrument model</summary>
      <p className="quiet">{identity?.error ?? identity?.model ?? 'Checking classifier…'}</p>
      {!!identity?.model && <p className="quiet">Local weights · models/instrument-identity.pt · V3 photos and reviewed video crops</p>}
      {identity?.specialists?.map(detail=><p key={detail.model} className="quiet">
        {detail.model} · {Object.keys(detail.thresholds ?? {}).map(key=>key.replaceAll('_',' ')).join(', ') || 'No enabled identities'}
      </p>)}
      {identity?.thresholds && <p className="quiet">{Object.entries(identity.thresholds).map(([key,value])=>key.replaceAll('_',' ')+' '+Math.round(value*100)+'%').join(' · ') || 'No enabled identities'}</p>}
      {identity?.scores?.video_test && <p className="quiet">Held-out video · {identity.scores.video_test.crops} checked crops · {Math.round(identity.scores.video_test.abstention*100)}% uncertain</p>}
    </details>
    {busy && !identifying && <p role="status" className="quiet">Checking video frames for possible instrument regions…</p>}
    {error && <p className="error" role="alert">{error}</p>}
    {(report?.scanned || report?.experimental_samples != null) && <label className="toggle"><input type="checkbox" checked={candidates} disabled={saving}
      onChange={e=>{setCandidates(e.target.checked);if(!e.target.checked && current?.decision!=='confirmed')setSelected(undefined)}}/>
      <span>Show model candidates<small>{report?.experimental_samples != null ? `Experimental video model sampled ${report.experimental_samples} frames. Review each box and supply the tool name.` : `Still-photo model sampled ${report?.samples} frames.`}</small></span></label>}
    <div className="instrument-layout">
      {visible.length>0 ? <div>
        <div className="instrument-timeline" aria-label={candidates?'Instrument candidates and reviews':'Confirmed instruments'}>
          {visible.map(item=>{
            const saved=reviews.get(item.best.detection_id)
            const confirmed=saved?.decision==='confirmed'
            const at=confirmed?item.best.time:item.time
            return <button key={item.id} type="button" aria-label={`Open ${confirmed?'review':'candidate'} at ${clock(at,true)}`}
              onClick={()=>choose(item)} disabled={saving || tracking} aria-pressed={selected?.id===item.id}
              style={{left:at/duration*100+'%',width:Math.max(.7,confirmed?0:(item.end-item.time)/duration*100)+'%'}}/>
          })}
        </div>
        <div className="instrument-list">
          {visible.map(item=>{
            const saved=reviews.get(item.best.detection_id)
            const classification=item.best.identity
            const heading=saved?.decision==='confirmed'?saved.label:classification?.accepted?classification.label:
              classification?'Uncertain instrument':item.best.detection_id.startsWith('instrument-video-')?'Unclassified candidate':item.best.class
            const status=saved?.decision==='dismissed'?'Rejected':saved?.decision==='confirmed'?
              (saved.review_action==='corrected'?'Corrected':'Confirmed'):'Needs review'
            return <button className="history-row instrument-row" key={item.id} disabled={saving || tracking || identifying}
              onClick={()=>choose(item)} aria-pressed={selected?.id===item.id}>
              <img loading="lazy" alt="" src={root+'/still.jpg?instrument='+encodeURIComponent(item.best.detection_id)}/>
              <span><strong>{heading}</strong><small>{clock(item.time,true)}{item.end>item.time && `–${clock(item.end,true)} · ${item.samples} frames`}</small></span>
              <small>{classification?.accepted ? `${Math.round(classification.confidence*100)}% · ` : ''}{status}</small>
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
            {previewBox ? <rect x={previewBox[0]*100} y={previewBox[1]*100} width={(previewBox[2]-previewBox[0])*100} height={(previewBox[3]-previewBox[1])*100} vectorEffect="non-scaling-stroke"/> :
              <polygon points={selected.best.polygon?.map(p=>p.map(v=>v*100).join(',')).join(' ')} vectorEffect="non-scaling-stroke"/>}
          </svg>
        </div>}
        <p className="quiet">{clock(sourceTime,true)}
          {selected && current?.decision!=='confirmed' && selected.best.confidence!==null &&
            ` · Model guess: ${selected.best.class} (${Math.round(selected.best.confidence*100)}%)`}</p>
        {selected?.best.identity && <p className="quiet">
          {selected.best.identity.accepted ? selected.best.identity.label : 'Uncertain instrument'} · {selected.best.identity.model}
          {' · '}{selected.best.identity.alternatives.map(value=>`${value.label} ${Math.round(value.confidence*100)}%`).join(' · ')}
        </p>}
        <div className="actions">
          <button className="subtle compact-action" onClick={()=>jump(sourceTime)}>Open source video</button>
          {!manual && !redraw && !trackedFrame && <button className="subtle compact-action" disabled={tracking} onClick={()=>{clearTrack();setRedraw(true)}}>Redraw box</button>}
          {current?.decision==='confirmed' && !redraw && <button className="subtle compact-action" disabled={tracking || saving} onClick={()=>void follow()}>{tracking?'Following…':'Follow nearby frames'}</button>}
        </div>
        {tracked && <div className="field">
          {tracked.length===0 ? <p className="quiet">No stable match nearby. Mark the instrument on another clear frame.</p> : <>
            <label className="field">Inspect tracked frame
              <select value={trackedFrame?.frame ?? ''} onChange={e=>setTrackedFrame(tracked.find(frame=>String(frame.frame)===e.target.value))}>
                <option value="">Confirmed frame</option>
                {tracked.map(frame=><option key={frame.frame} value={frame.frame}>{clock(frame.time,true)} · Review needed</option>)}
              </select>
            </label>
            {trackedFrame && <button className="subtle compact-action" onClick={useTrackedFrame}>Review this frame</button>}
          </>}
        </div>}
        {!trackedFrame && <><label className="field">Reviewed label<input list="review-instrument-types" maxLength={100} value={label} disabled={saving} onChange={e=>setLabel(e.target.value)}/></label>
        <datalist id="review-instrument-types">
          {['Dental mirror', 'Dental explorer/probe', 'Periodontal probe', 'Scaler', 'Dental suction', 'High-speed handpiece',
            'Low-speed handpiece', 'Dental forceps', 'Orthodontic bracket holder', 'Bracket positioner', 'Ligature applicator',
            'Weingart pliers', 'Wire cutter', 'Other recognizable dental instrument', 'Unknown'].map(type=><option key={type} value={type}/>)}
        </datalist>
        <details>
          <summary>Tooth association and visibility</summary>
          <label className="field">Associated FDI number (optional)<input value={fdi} maxLength={2} onChange={e=>setFdi(e.target.value)}/></label>
          <label className="field">Visibility<select value={visibility} onChange={e=>setVisibility(e.target.value)}>
            <option value="">Unspecified</option><option value="visible">Visible</option><option value="partial">Partially obscured</option>
          </select></label>
        </details>
        <label className="field">Note<textarea maxLength={500} rows={2} value={note} disabled={saving} onChange={e=>setNote(e.target.value)}/></label>
        <div className="actions">
          <button className="primary compact-action" disabled={saving || tracking || !label.trim() || (redraw && !box)}
            onClick={()=>void review('confirmed')}>{current?.decision==='confirmed'?'Save changes':selected?.best.identity?.accepted && label!==selected.best.identity.label?'Save correction':'Confirm'}</button>
          {!manual && <button className="subtle compact-action" disabled={saving || current?.decision==='dismissed'}
            onClick={()=>void review('dismissed')}>Reject</button>}
        </div>
        {current?.decision==='confirmed' && <a className="inline-jump" href={root+'/instrument-annotations'} download="instrument-annotations.json">Download instrument annotations</a>}
        </>}
      </div>}
    </div>
  </section>
}
