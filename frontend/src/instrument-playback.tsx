import type { Box } from './api'

type Identity = { label:string; accepted:boolean }
type Observation = {
  detection_id:string
  time:number
  box:Box
  identity?:Identity | null
  track_identity?:Identity | null
  context_identity?:Identity | null
  track_id?:string | null
  duplicate_of?:string | null
  instrument_status?:string
}
type Decision = {detection_id:string; decision:string; label:string; box?:Box | null}
type Sample = {time:number; box:Box; label:string; suggestion:string; context:string}
export type InstrumentScan = {
  observations:Observation[]
  reviews:Decision[]
  experimental_samples?:number | null
  maximum_track_gap?:number
}

function firstAfter(samples:Sample[], time:number) {
  let first=0, last=samples.length
  while(first<last) {
    const middle=(first+last)>>>1
    if(samples[middle].time<=time) first=middle+1
    else last=middle
  }
  return first
}

export function instrumentPlayback(scan:InstrumentScan) {
  const decisions=new Map(scan.reviews.map(row=>[row.detection_id,row]))
  const reviewedDuplicates=new Map<string,Decision[]>()
  for(const row of scan.observations) {
    const decision=decisions.get(row.detection_id)
    if(row.duplicate_of && decision?.decision==='confirmed') {
      const list=reviewedDuplicates.get(row.duplicate_of) ?? []
      list.push(decision)
      reviewedDuplicates.set(row.duplicate_of,list)
    }
  }
  const tracks=new Map<string,Sample[]>()
  const rejections=new Map<string,number[]>()
  const gap=scan.maximum_track_gap ?? .24
  for(const row of scan.observations) {
    if(!row.detection_id.startsWith('instrument-video-') || row.duplicate_of) continue
    const track=row.track_id ?? row.detection_id
    const reviewed=reviewedDuplicates.get(row.detection_id) ?? []
    const decision=decisions.get(row.detection_id) ??
      (new Set(reviewed.map(item=>item.label)).size===1?reviewed[0]:undefined)
    if(decision?.decision==='dismissed' || (row.instrument_status==='non_instrument' && decision?.decision!=='confirmed')) {
      const times=rejections.get(track) ?? []
      times.push(row.time)
      rejections.set(track,times)
      continue
    }
    const samples=tracks.get(track) ?? []
    const suggestion=row.track_identity ?? row.identity
    samples.push({time:row.time,box:decision?.box ?? row.box,
      label:decision?.decision==='confirmed'?decision.label:'',
      suggestion:suggestion?.accepted?suggestion.label:'',context:row.context_identity?.label ?? ''})
    tracks.set(track,samples)
  }
  const entries=[...tracks].map(([id,samples])=>{
    samples.sort((a,b)=>a.time-b.time)
    const labels=new Set(samples.map(sample=>sample.label).filter(Boolean))
    return {id,samples,reviewedLabel:labels.size===1?[...labels][0]:'',rejections:rejections.get(id) ?? []}
  })

  return (time:number)=>{
    const result=[]
    for(const {id,samples,reviewedLabel,rejections} of entries) {
      if(time<samples[0].time-.11 || time>samples[samples.length-1].time+.11 ||
         rejections.some(at=>Math.abs(at-time)<.10)) continue
      const index=firstAfter(samples,time)
      const before=samples[index-1], after=samples[index]
      const closest=before && (!after || time-before.time<=after.time-time)?before:after
      const bridged=before && after && after.time-before.time<=gap &&
        !rejections.some(at=>at>before.time && at<after.time)
      if(!closest || (!bridged && Math.abs(closest.time-time)>.11)) continue
      const fraction=bridged?(time-before.time)/(after.time-before.time):0
      const box=bridged?before.box.map((p,i)=>p+(after.box[i]-p)*fraction) as Box:closest.box
      const reviewed=!!closest.label && Math.abs(time-closest.time)<.025
      const prediction=closest.suggestion ||
        (bridged && before.suggestion===after.suggestion?before.suggestion:'')
      result.push({box,track:id,label:reviewed?closest.label:reviewedLabel ||
        (prediction?prediction+' · AI':closest.context?closest.context+' in crop':'Instrument candidate'),reviewed,named:!!reviewedLabel})
    }
    return result
  }
}
