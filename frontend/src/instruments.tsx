import { useEffect, useState } from 'react'
import { api, clock, send } from './api'
import type { Box } from './api'
import { RegionEditor } from './review-tools'

type Prediction = {class_id?:string;label:string;confidence:number;accepted:boolean;model:string;supporting_frames?:number;reason?:string | null;alternatives:{label:string;confidence:number}[]}
type IdentityInfo = {weights_sha256?:string;validation_status?:string;model?:string;specialists?:IdentityInfo[];context?:Record<string,number>|null;thresholds?:Record<string,number>;error?:string;scores?:Record<string,{crops:number;abstention:number}>}
type Observation = {identity?:Prediction | null;
  detection_id:string; class:string; time:number; reason_code?:string;
  crop_geometry?:{width:number;height:number;encoded_width:number;encoded_height:number;padding:number}; track_identity?:Prediction | null; confidence:number | null; polygon:number[][] | null; box:Box
}
type TrackEvidence = { detection_id:string;time:number;status:string;reason:string;predicted_class?:string;score?:number }
type Appearance = {
  samples?:number; id:string; time?:number; start?:number; end:number; best:Observation; track_id?:string | null
  duration?:number; status?:string; termination_status?:string; identity?:Prediction | null
  context_identity?:Prediction | null; evidence?:TrackEvidence[]; gaps?:{start:number;end:number}[]
}
type Diagnostics = {raw_proposals:number;duplicate_proposals:number;tracks:number;proposal_statuses:Record<string,number>}
type Review = { review_action?:string; detection_id:string; label:string; note:string; decision:'confirmed'|'dismissed'|'unknown'; box?:Box | null; fdi_number?:string | null; visibility?:'visible' | 'partial' | null }
type InstrumentReport = {observations?:Observation[];diagnostics?:Diagnostics;identity_model?:string | null; device:string | null; samples:number; seconds:number; scanned:boolean; experimental_samples?:number | null; appearances:Appearance[]; reviews:Review[] }
type TrackedFrame = {frame:number; time:number; box:Box}
type ShadowReport = {metadata:{model:string;classes:string[];thresholds:Record<string,number>;weights_sha256:string;
  calibration_version:string;temperature:number;evaluation_date:string;validation_status:string};production_model:string;
  latency_seconds:number;rows:Record<string,{production:Prediction|null;experimental:Prediction|null;disagreement:boolean}>}
type InstrumentResponse = {shadow_enabled?:boolean;identity?:IdentityInfo;model_ready:boolean; experimental_ready?:boolean; report:InstrumentReport | null}

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
  const [showContext,setShowContext]=useState(true)
  const [queue,setQueue]=useState<'all'|'unidentified'|'reviewed'>('all')
  const [experimentalReady,setExperimentalReady]=useState(false)
  const [identity,setIdentity]=useState<IdentityInfo>()
  const [identifying,setIdentifying]=useState(false)
  const [shadowEnabled,setShadowEnabled]=useState(false)
  const [shadow,setShadow]=useState<ShadowReport|null>(null)
  const [shadowBusy,setShadowBusy]=useState(false)
  const [tracking,setTracking]=useState(false)
  const [tracked,setTracked]=useState<TrackedFrame[] | null>(null)
  const [trackedFrame,setTrackedFrame]=useState<TrackedFrame>()
  useEffect(()=>{
    const controller=new AbortController()
    setShadow(null);setShadowEnabled(false)
    api<InstrumentResponse>(root+'/instruments',{signal:controller.signal})
      .then(data=>{
        setShadowEnabled(!!data.shadow_enabled)
        if(data.shadow_enabled) void api<ShadowReport|null>(root+'/instrument-shadow',{signal:controller.signal})
          .then(value=>{if(!controller.signal.aborted)setShadow(value)})
          .catch(e=>{if(!controller.signal.aborted)setError(e.message)})
        setAvailable(data.model_ready);setExperimentalReady(!!data.experimental_ready);setIdentity(data.identity);setReport(data.report);setCandidates(data.report?.experimental_samples != null)})
      .catch(e=>{if(!controller.signal.aborted)setError(e.message)})
    return ()=>controller.abort()
  },[root])

  const reviews=new Map(report?.reviews.map(item=>[item.detection_id,item]))
  const comparison=selected && shadow?.rows[selected.best.detection_id]
  const current=selected && reviews.get(selected.best.detection_id)
  const candidatesAndReviews=report?.appearances.filter(item=>reviews.get(item.best.detection_id)?.decision==='confirmed' ||
    (candidates && item.best.confidence!==null && (showContext || !['non_instrument','context_detected'].includes(item.status ?? '')))) ?? []
  const visible=candidatesAndReviews.filter(item=>{
    const decision=reviews.get(item.best.detection_id)?.decision
    if(queue==='reviewed') return !!decision
    if(queue==='unidentified') return item.best.detection_id.startsWith('instrument-video-') &&
      decision!=='confirmed' && decision!=='dismissed' && !(item.identity ?? item.best.identity)?.accepted
    return true
  })
  const count=new Set(visible.map(item=>item.track_id ?? item.id)).size
  const sourceTime=trackedFrame?.time ?? selected?.best.time ?? time
  const previewBox=trackedFrame?.box ?? box
  const still=root+'/still.jpg?time='+sourceTime

  async function runShadow() {
    setShadowBusy(true);setError('')
    try {setShadow(await send<ShadowReport>(root+'/instrument-shadow',{}))}
    catch(e) {setError(e instanceof Error?e.message:'Shadow comparison failed.')}
    finally {setShadowBusy(false)}
  }
  function clearTrack() {
    setTracked(null)
    setTrackedFrame(undefined)
  }
  function choose(item:Appearance) {
    clearTrack()
    const record=reviews.get(item.best.detection_id)
    setSelected(item);setManual(false);setRedraw(false);setBox(record?.box ?? null);setError('')
    setLabel(record?.label ?? (item.identity?.label ?? item.best.track_identity?.label ?? (item.best.identity?.accepted ? item.best.identity.label : '')));setNote(record?.note ?? '')
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
  async function review(decision:Review['decision'],nonInstrument=false) {
    if(!selected && !manual) return
    setSaving(true);setError('')
    try {
      const record=await send<Review>(root+'/instrument-reviews',{
        detection_id:selected?.best.detection_id ?? null,time:manual?time:null,label,note,decision,box,
        fdi_number:fdi || null,visibility:visibility || null,tracked_from:anchor ?? null,non_instrument:nonInstrument,
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
      {identity?.weights_sha256 && <p className="quiet">Production SHA-256 <code>{identity.weights_sha256}</code></p>}
      {shadowEnabled && <div>
        <button className="subtle compact-action" disabled={shadowBusy || busy || identifying || !report?.identity_model}
          onClick={()=>void runShadow()}>{shadowBusy?'Comparing…':'Run shadow comparison'}</button>
        {shadow && <>
          <button className="subtle compact-action" onClick={()=>setShadow(null)}>Hide comparison</button>
          <p className="quiet">{shadow.metadata.model} · {shadow.metadata.validation_status} · {shadow.latency_seconds.toFixed(1)} s</p>
          <p className="quiet">{shadow.metadata.calibration_version} · {shadow.metadata.evaluation_date}
            {' · '}{shadow.metadata.classes.map(key=>key.replaceAll('_',' ')).join(', ')}
            {' · '}{Object.entries(shadow.metadata.thresholds).map(([key,value])=>key.replaceAll('_',' ')+' '+Math.round(value*100)+'%').join(', ')}</p>
          <p className="quiet">Shadow SHA-256 <code>{shadow.metadata.weights_sha256}</code></p>
        </>}
      </div>}
      {!!identity?.model && <p className="quiet">Local weights · models/instrument-identity.pt · V3 photos and reviewed video crops</p>}
      {identity?.specialists?.map(detail=><p key={detail.model} className="quiet">
        {detail.model} · {Object.keys(detail.thresholds ?? {}).map(key=>key.replaceAll('_',' ')).join(', ') || 'No enabled identities'}
      </p>)}
      {identity?.context && <p className="quiet">Context model · {Object.keys(identity.context).map(label=>label.replaceAll('_',' ')).join(', ')}</p>}
      {identity?.thresholds && <p className="quiet">{Object.entries(identity.thresholds).map(([key,value])=>key.replaceAll('_',' ')+' '+Math.round(value*100)+'%').join(' · ') || 'No enabled identities'}</p>}
      {identity?.scores?.video_test && <p className="quiet">Held-out video · {identity.scores.video_test.crops} checked crops · {Math.round(identity.scores.video_test.abstention*100)}% uncertain</p>}
    </details>
    {busy && !identifying && <p role="status" className="quiet">Checking video frames for possible instrument regions…</p>}
    {error && <p className="error" role="alert">{error}</p>}
    {(report?.scanned || report?.experimental_samples != null) && <label className="toggle"><input type="checkbox" checked={candidates} disabled={saving}
      onChange={e=>{setCandidates(e.target.checked);if(!e.target.checked && current?.decision!=='confirmed')setSelected(undefined)}}/>
      <span>Show model candidates<small>{report?.experimental_samples != null ? `Experimental video model sampled ${report.experimental_samples} frames. Review each box and supply the tool name.` : `Still-photo model sampled ${report?.samples} frames.`}</small></span></label>}
    {report?.diagnostics && <p className="quiet">
      {report.diagnostics.raw_proposals} video proposals · {report.diagnostics.duplicate_proposals} duplicates · {report.diagnostics.tracks} tracks
      {(report.diagnostics.proposal_statuses.context_detected ?? 0)>0 && <> · {report.diagnostics.proposal_statuses.context_detected} proposals with visible hand or mouth context</>}
    </p>}
    {report?.appearances.some(item=>['non_instrument','context_detected'].includes(item.status ?? '')) && <label className="toggle">
      <input type="checkbox" checked={showContext} onChange={e=>setShowContext(e.target.checked)}/> Show hand and mouth context
    </label>}
    {report?.experimental_samples != null && <div className="instrument-queue">
      <label className="field">Review queue<select value={queue}
        onChange={e=>{setQueue(e.target.value as typeof queue);setSelected(undefined);setManual(false)}}>
        <option value="all">All results</option><option value="unidentified">Unidentified tracks</option><option value="reviewed">Reviewed results</option>
      </select></label>
      <small className="quiet">{count} track or result group{count===1?'':'s'}</small>
    </div>}
    <div className="instrument-layout">
      {visible.length>0 ? <div>
        <div className="instrument-timeline" aria-label={candidates?'Instrument candidates and reviews':'Confirmed instruments'}>
          {visible.map(item=>{
            const saved=reviews.get(item.best.detection_id)
            const confirmed=saved?.decision==='confirmed'
            const at=confirmed?item.best.time:item.start ?? item.time ?? item.best.time
            return <button key={item.id} type="button" aria-label={`Open ${confirmed?'review':'candidate'} at ${clock(at,true)}`}
              onClick={()=>choose(item)} disabled={saving || tracking} aria-pressed={selected?.id===item.id}
              style={{left:at/duration*100+'%',width:Math.max(.7,confirmed?0:(item.end-at)/duration*100)+'%'}}/>
          })}
        </div>
        <div className="instrument-list">
          {visible.map(item=>{
            const saved=reviews.get(item.best.detection_id)
            const classification=item.identity ?? item.best.identity
            const start=item.start ?? item.time ?? item.best.time
            const heading=saved?.decision==='confirmed'?saved.label:saved?.decision==='unknown'?'Type uncertain':item.context_identity ? item.context_identity.label+' in crop' :
              (classification?.accepted?classification.label:
               item.status==='track_interrupted'?'Track interrupted':item.status==='type_uncertain'?'Type uncertain':
               item.best.detection_id.startsWith('instrument-video-')?'Unclassified candidate':item.best.class)
            const status=saved?.decision==='unknown'?'Left unknown':saved?.decision==='dismissed'?(saved.review_action==='non_instrument'?'Non-instrument':'Rejected'):saved?.decision==='confirmed'?
              (saved.review_action==='corrected'?'Corrected':'Confirmed'):
              item.status==='non_instrument'?'Non-instrument':item.status==='context_detected'?'Context · Needs review':classification?.accepted?'AI predicted · Needs review':'Needs review'
            return <button className="history-row instrument-row" key={item.id} disabled={saving || tracking || identifying}
              onClick={()=>choose(item)} aria-pressed={selected?.id===item.id}>
              <img loading="lazy" alt="" src={root+'/still.jpg?instrument='+encodeURIComponent(item.best.detection_id)}/>
              <span><strong>{heading}</strong><small>{clock(start,true)}{item.end>start && `–${clock(item.end,true)} · ${item.samples} frames`}{item.track_id && ` · ${item.track_id}`}</small></span>
              <small>{saved?.decision!=='unknown' && (item.context_identity ?? classification)?.accepted ? `${Math.round((item.context_identity ?? classification)!.confidence*100)}% · ` : ''}{status}</small>
            </button>
          })}
        </div>
      </div> : <p className="quiet">{candidates?'No results in this queue.':'No confirmed instruments. Mark a region or review the model candidates.'}</p>}
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
        {selected && !trackedFrame && <details open={queue==='unidentified' || undefined}>
          <summary>Inspect model crop</summary>
          <img className="instrument-audit-crop" loading="lazy" alt="Padded instrument proposal"
            src={root+'/still.jpg?instrument='+encodeURIComponent(selected.best.detection_id)+'&detail=true'}/>
          {selected.best.crop_geometry && <p className="quiet">
            {selected.best.crop_geometry.width} × {selected.best.crop_geometry.height} px source crop ·
            {' '}{selected.best.crop_geometry.encoded_width} × {selected.best.crop_geometry.encoded_height} px before padding to model input ·
            {' '}{Math.round(selected.best.crop_geometry.padding*100)}% crop margin per side
          </p>}
        </details>}
        <p className="quiet">{clock(sourceTime,true)}
          {selected && current?.decision!=='confirmed' && selected.best.confidence!==null &&
            ` · Model guess: ${selected.best.class} (${Math.round(selected.best.confidence*100)}%)`}</p>
        {selected?.best.identity && <p className="quiet">
          {selected.best.identity.accepted ? selected.best.identity.label : 'Uncertain instrument'} · {selected.best.identity.model}
          {' · '}{selected.best.identity.alternatives.map(value=>`${value.label} ${Math.round(value.confidence*100)}%`).join(' · ')}
        </p>}
        {comparison && <details open>
          <summary>Shadow comparison</summary>
          <p className="quiet">Production · {comparison.production ? comparison.production.label+' '+Math.round(comparison.production.confidence*100)+'% · '+comparison.production.model : 'No prediction'}
            {comparison.production && !comparison.production.accepted && ' · Rejected'}</p>
          <p className="quiet">Experimental · {comparison.experimental ? comparison.experimental.label+' '+Math.round(comparison.experimental.confidence*100)+'%' : 'No prediction'}
            {comparison.experimental && !comparison.experimental.accepted && ' · Rejected'}</p>
          <p className="quiet">{comparison.disagreement?'Different top classes':'Same top class'} · {current?.decision==='confirmed'?current.label:'Awaiting reviewer confirmation'}</p>
        </details>}
        {selected?.context_identity && <p className="quiet">
          {selected.context_identity.label} visible in this crop · {Math.round(selected.context_identity.confidence*100)}% model score
        </p>}
        {selected?.evidence && <details>
          <summary>Track evidence</summary>
          <p className="quiet">{selected.track_id} · {(selected.duration ?? 0).toFixed(2)} s · {selected.samples} detections
            {selected.identity && ` · ${selected.identity.supporting_frames} supporting frames`}
            {selected.termination_status==='track_lost'?' · Track lost':selected.termination_status==='track_interrupted'?' · Track interrupted':''}
          </p>
          {!!selected.gaps?.length && <p className="quiet">{selected.gaps.length} short tracking gaps</p>}
          <label className="field">Evidence frame<select value={selected.best.detection_id}
            onChange={e=>{
              const row=report?.observations?.find(item=>item.detection_id===e.target.value)
              if(row) choose({...selected,best:row})
            }}>
            {!selected.evidence.some(row=>row.detection_id===selected.best.detection_id) &&
              <option value={selected.best.detection_id}>{clock(selected.best.time,true)} · Reviewed proposal</option>}
            {selected.evidence.map(row=><option key={row.detection_id} value={row.detection_id}>
              {clock(row.time,true)} · {row.status.replaceAll('_',' ')}{row.score!=null?` · ${Math.round(row.score*100)}%`:''}
            </option>)}
          </select></label>
          {selected.best.reason_code && <p className="quiet">{selected.best.reason_code.replaceAll('_',' ')}</p>}
        </details>}
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
          {!manual && <button className="subtle compact-action" disabled={saving || current?.decision==='unknown'}
            onClick={()=>void review('unknown')}>Leave unknown</button>}
          {!manual && <button className="subtle compact-action" disabled={saving || current?.review_action==='non_instrument'}
            onClick={()=>void review('dismissed',true)}>Mark non-instrument</button>}
        </div>
        {current?.decision==='confirmed' && <a className="inline-jump" href={root+'/instrument-annotations'} download="instrument-annotations.json">Download instrument annotations</a>}
        </>}
      </div>}
    </div>
  </section>
}
