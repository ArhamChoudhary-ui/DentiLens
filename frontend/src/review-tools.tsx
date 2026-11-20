import { useState } from 'react'
import type { PointerEvent } from 'react'
import { ArrowDownToLine, Clock, Download } from 'lucide-react'
import { clock } from './api'
import { ToothHistory } from './intelligence'
import type { Box, Exam, Finding, Review } from './api'

const subtypes = ['central incisor', 'lateral incisor', 'canine', 'first premolar', 'second premolar', 'first molar', 'second molar', 'third molar']
const upper = ['18','17','16','15','14','13','12','11','21','22','23','24','25','26','27','28']
const lower = ['48','47','46','45','44','43','42','41','31','32','33','34','35','36','37','38']

function suggestionNumber(item: Finding) {
  return item.fdi_number ?? item.label.match(/^FDI (\d{2})\b/)?.[1]
}

export function ToothChart({exam, jump, jumpToTime, initial, openAnatomy}: {exam: Exam; jump: (item: Finding) => void;
  jumpToTime: (time:number)=>void; initial:string; openAnatomy:(fdi:string)=>void}) {
  const [tooth, setTooth] = useState(initial)
  const reviews = exam.reviews ?? []
  const report = exam.report!
  function associations(code: string) {
    return report.findings.flatMap(item => {
      const review = reviews.find(value => value.finding_id === item.id)
      const number = review?.fdi_number ?? suggestionNumber(item)
      return number === code ? [{item, review}] : []
    })
  }
  const selected = associations(tooth)
  const confirmed = reviews.filter(item => !item.finding_id && item.fdi_number === tooth && item.decision === 'confirmed')
  function arch(codes: string[], name: string) {
    return <div className="arch"><h3>{name}</h3><div className="arch-teeth">
      {codes.map(code => {
        const items = associations(code)
        const status = items.some(({review}) => review?.decision==='confirmed' && review.fdi_number===code)
          || reviews.some(review => !review.finding_id && review.fdi_number===code && review.decision==='confirmed') ? 'confirmed'
          : items.some(({review}) => review?.decision!=='dismissed') ? 'suggested'
          : items.length ? 'dismissed' : report.tracks.some(track=>track.fdi_number===code) ? 'suggested' : 'unobserved'
        return <button type="button" key={code} className={`tooth-cell ${status}`} aria-pressed={tooth===code}
          aria-label={`Tooth ${code}, ${status}`} onClick={()=>setTooth(code)}>{code}</button>
      })}
    </div></div>
  }
  return <section className="panel tooth-chart">
    <h2>Permanent teeth</h2>
    <div className="chart-orientation"><span>Patient’s right</span><span>Patient’s left</span></div>
    {arch(upper,'Upper arch')}
    {arch(lower,'Lower arch')}
    <div className="arch-legend"><span className="confirmed">Confirmed</span><span className="suggested">Suggested</span>
      <span className="dismissed">Dismissed</span><span>Unobserved</span></div>
    {tooth ? <div className="chart-detail">
      <div className="section-heading"><h3>{tooth} · {subtypes[Number(tooth[1])-1]}</h3>
        <button className="subtle compact-action" onClick={()=>openAnatomy(tooth)}>View anatomy</button></div>
      {selected.map(({item, review})=><button className="chart-finding" key={item.id} onClick={()=>jump(item)}>
        <img src={`/api/exams/${exam.id}/images/${item.image}`} alt="" loading="lazy"/>
        <span><strong>{review?.label || item.label}</strong><small>{review?.decision ?? 'Suggested'} · {clock(item.time,true)}</small></span>
      </button>)}
      {confirmed.map(review=><p key={review.id}>{review.label || 'Confirmed bookmark'} · {clock(review.time,true)}</p>)}
      <ToothHistory key={tooth} id={exam.id} fdi={tooth} jump={jumpToTime}/>
    </div> : <p className="quiet">Select a tooth to see its bookmarks.</p>}
  </section>
}

export function RegionEditor({src, value, change}: {src: string; value: Box | null; change: (box: Box | null) => void}) {
  const [start, setStart] = useState<[number, number] | null>(null)
  const [draft, setDraft] = useState<Box | null>(null)
  function point(e: PointerEvent<HTMLDivElement>): [number, number] {
    const rect = e.currentTarget.getBoundingClientRect()
    return [Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width)),
            Math.max(0, Math.min(1, (e.clientY - rect.top) / rect.height))]
  }
  function bounds(a: [number,number], b: [number,number]): Box {
    return [Math.min(a[0],b[0]), Math.min(a[1],b[1]), Math.max(a[0],b[0]), Math.max(a[1],b[1])]
  }
  const box = draft ?? value
  return <div className="region-editor">
    <p>Drag on the frame to mark a region.</p>
    <div className="region-image" onPointerDown={e=>{e.currentTarget.setPointerCapture(e.pointerId); setStart(point(e))}}
      onPointerMove={e=>{if(start) setDraft(bounds(start,point(e)))}}
      onPointerCancel={()=>{setStart(null);setDraft(null)}}
      onPointerUp={e=>{
        if(start) {
          const next=bounds(start,point(e))
          if(next[2]-next[0]>.005 && next[3]-next[1]>.005) change(next)
        }
        setStart(null);setDraft(null)
      }}>
      <img src={src} alt="Original frame for marking a region" draggable={false}/>
      {box && <span className="region-box" style={{left:box[0]*100+'%',top:box[1]*100+'%',
        width:(box[2]-box[0])*100+'%',height:(box[3]-box[1])*100+'%'}}/>}
    </div>
    <div className="region-coordinates">
      {['Left','Top','Right','Bottom'].map((name,index)=><label className="field" key={name}>{name} %
        <input type="number" min={0} max={100} step={.1} value={value ? Math.round(value[index]*1000)/10 : ''}
          onChange={e=>{
            const next: Box = value ? [...value] : [.1,.1,.9,.9]
            next[index]=Math.min(100,Math.max(0,Number(e.target.value)))/100
            change(next)
          }}/></label>)}
    </div>
    <button type="button" className="subtle compact-action" disabled={!value} onClick={()=>change(null)}>Reset box</button>
  </div>
}

export function FullVideoExport({exam, request}: {exam: Exam; request: (options: object) => Promise<string>}) {
  const [version,setVersion]=useState('confirmed')
  const [includeInstruments,setIncludeInstruments]=useState(false)
  const [preset,setPreset]=useState('labels')
  const [url,setUrl]=useState('')
  const [status,setStatus]=useState('')
  const [working,setWorking]=useState(false)
  const [custom,setCustom]=useState<{boxes:boolean;labels:boolean;fdi:boolean;timestamp:boolean;scores:boolean;track_ids:boolean}>()
  const defaults={boxes:true,labels:preset!=='minimal',fdi:preset!=='minimal',
    timestamp:preset!=='minimal',scores:preset==='detailed',track_ids:preset==='detailed'}
  const overlay=custom ?? defaults
  const controls={boxes:'Boxes',labels:'Tooth names',fdi:'FDI numbers',timestamp:'Timestamp',scores:'Model scores',track_ids:'Track IDs'}
  return <details className="panel limits export-options">
    <summary>Export full video</summary>
    <form onSubmit={async e=>{
      e.preventDefault();setWorking(true);setStatus('');setUrl('')
      try {
        const result=await request({version,...overlay,instruments:includeInstruments})
        setUrl(result);setStatus('Video ready.')
      } catch(e) {setStatus(e instanceof Error?e.message:'Export failed. Try again.')}
      finally {setWorking(false)}
    }}>
      <label className="field">Markings<select value={version} disabled={working} onChange={e=>setVersion(e.target.value)}>
        <option value="confirmed">Confirmed bookmarks</option><option value="suggestions">AI suggestions</option>
      </select></label>
      <label className="field">Overlay<select value={preset} disabled={working} onChange={e=>{setPreset(e.target.value);setCustom(undefined)}}>
        <option value="labels">Standard</option><option value="minimal">Minimal</option><option value="detailed">Detailed</option>
      </select></label>
      <details className="overlay-options"><summary>Choose what to show</summary>
        {(Object.keys(controls) as Array<keyof typeof controls>).map(key=><label className="toggle" key={key}>
          <input type="checkbox" checked={overlay[key]} disabled={working} onChange={e=>setCustom({...overlay,[key]:e.target.checked})}/>
          <span>{controls[key]}</span>
        </label>)}
      </details>
      {exam.instruments_available && <label className="toggle"><input type="checkbox" checked={includeInstruments}
        disabled={working} onChange={e=>setIncludeInstruments(e.target.checked)}/><span>Include reviewed instruments</span></label>}
      <button className="subtle" disabled={working || (version==='confirmed' &&
        !(includeInstruments && exam.confirmed_instruments) &&
        !exam.reviews?.some(item=>item.decision==='confirmed' && (item.finding_id || item.box)))}>
        <Download size={15}/>{working?'Building video…':'Build video'}
      </button>
      {status && <div role="status">{status}</div>}
      {url && <><video src={url} controls playsInline preload="metadata"/><a className="download-link" href={url+'&download=true'}><ArrowDownToLine size={15}/> Download MP4</a></>}
    </form>
  </details>
}

export function FrameAnalysis({exam}: {exam: Exam}) {
  const report=exam.report!
  const labels: Record<string,string> = {overlay:'Frames with overlays',region_below_search_threshold:'No region passed the search threshold',
    no_accepted_detection:'No detection passed review thresholds',association_rejected:'Association rejected',
    tracking_lost:'Tracking lost',between_detection_passes:'Awaiting detection'}
  const confirmed=exam.reviews?.filter(item=>item.decision==='confirmed').length ?? 0
  const dismissed=exam.reviews?.filter(item=>item.decision==='dismissed').length ?? 0
  return <dl className="analysis-data">
    <div><dt>Individual tooth overlays</dt><dd>{report.stats.individual_tooth_frames ?? '—'}</dd></div>
    <div><dt>Confirmed bookmarks</dt><dd>{confirmed}</dd></div>
    <div><dt>Dismissed suggestions</dt><dd>{dismissed}</dd></div>
    {Object.entries(report.frame_reasons ?? {}).map(([key, value])=><div key={key}><dt>{labels[key] ?? key}</dt><dd>{value}</dd></div>)}
    <div><dt>Eligible-frame coverage</dt><dd>Requires visibility labels</dd></div>
    {report.performance && <div><dt><Clock size={13}/> Analysis time</dt><dd>{report.performance.total_seconds} s</dd></div>}
  </dl>
}
