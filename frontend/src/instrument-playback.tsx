import type { Box } from './api'

type Observation = {detection_id:string; time:number; box:Box}
type Decision = {detection_id:string; decision:string; label:string; box?:Box | null}
export type InstrumentScan = {observations:Observation[]; reviews:Decision[]; experimental_samples?:number | null}

function overlap(a:Box,b:Box) {
  const area=(box:Box)=>(box[2]-box[0])*(box[3]-box[1])
  const intersection=Math.max(0,Math.min(a[2],b[2])-Math.max(a[0],b[0]))*
    Math.max(0,Math.min(a[3],b[3])-Math.max(a[1],b[1]))
  return intersection/Math.max(.000001,area(a)+area(b)-intersection)
}

export function instrumentPlayback(scan:InstrumentScan) {
  const decisions=new Map(scan.reviews.map(row=>[row.detection_id,row]))
  const groups=new Map<number, {box:Box; label:string; track:number}[]>()
  let nextTrack=0
  const names=new Map<number,Set<string>>()
  for(const row of scan.observations) {
    if(!row.detection_id.startsWith('instrument-video-')) continue
    const decision=decisions.get(row.detection_id)
    if(decision?.decision==='dismissed') continue
    const group=groups.get(row.time) ?? []
    group.push({box:decision?.box ?? row.box,label:decision?.decision==='confirmed'?decision.label:'',track:-1})
    groups.set(row.time,group)
  }
  const frames=[...groups].sort(([a],[b])=>a-b)
  frames.forEach(([time,rows],index)=>{
    const previous=frames[index-1]
    const prior=previous && time-previous[0]<=.24?previous[1]:[]
    for(const row of rows) {
      const matches=prior.filter(other=>overlap(other.box,row.box)>=.45)
      const [match]=matches
      row.track=matches.length===1 && rows.filter(other=>overlap(other.box,match.box)>=.45).length===1
        ?match.track:nextTrack++
      if(row.label) {
        const labels=names.get(row.track) ?? new Set<string>()
        labels.add(row.label)
        names.set(row.track,labels)
      }
    }
  })
  return (time:number)=>{
    const index=frames.findIndex(([at])=>at>time)
    const before=frames[index<0?frames.length-1:index-1]
    const after=frames[index]
    const closest=before && (!after || time-before[0]<=after[0]-time)?before:after
    if(!closest || Math.abs(time-closest[0])>.11) return []
    return closest[1].map(row=>{
      const from=before?.[1].find(other=>other.track===row.track)
      const to=after?.[1].find(other=>other.track===row.track)
      const fraction=from && to && after[0]-before[0]<=.24?(time-before[0])/(after[0]-before[0]):0
      const box=from && to && fraction>0?from.box.map((value,i)=>value+(to.box[i]-value)*fraction) as Box:row.box
      const labels=names.get(row.track)
      const trackedName=labels?.size===1?[...labels][0]:''
      const reviewed=!!row.label && Math.abs(time-closest[0])<.025
      return {box,track:row.track,label:reviewed?row.label:trackedName || 'Instrument',reviewed,named:!!trackedName}
    })
  }
}
