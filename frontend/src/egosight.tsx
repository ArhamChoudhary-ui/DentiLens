import { useEffect, useRef, useState } from 'react'
import { Activity, AlertCircle, ArrowDownToLine, ArrowRight, CheckCircle2, Clock3, Eye, FileJson, Film, LayoutGrid, Loader2, Settings2, Sparkles, UploadCloud, X } from 'lucide-react'
import { api, uploadWithProgress, UploadProgress } from './api'
import './egosight.css'

const repo = 'https://github.com/ArhamChoudhary-ui/EgoSight-XR'
const guide = repo + '/tree/xr-4-video-intelligence/track4'
type Span = [number, number]
type Result = {
  duration_sec: number
  label: { name: string; score: number }
  actions: { segment: Span; verb: string; noun: string; score: number }[]
  highlights: { segment: Span }[]
}
type HighlightEvidence = { verb?: string; noun?: string; confidence?: number; fallback?: boolean }

export type ActionMetadata = {
  start: number
  end: number
  category: 'Recognized action' | 'Potential activity interval' | 'Activity transition' | 'Visually unclear interval'
  quality_rank: number
  quality_score: number
  rationale: string
  d_score?: number
  d_motion?: number
}

export type HighlightDetail = {
  start: number
  end: number
  duration: number
  importance: number
  related_proposal_index?: number | null
  related_action?: string | null
  rationale: string
}

export type ProposalStats = {
  raw_candidates?: number
  retained?: number
  suppressed_nms?: number
  suppressed_low_conf?: number
  suppressed_low_info?: number
}

export type SceneItem = {
  index: number
  timestamp: number
  timestamp_formatted: string
  caption: string
  quality?: string
  context?: string
  highlight_index?: number | null
  thumbnail?: string
  model?: string
}

export type SceneUnderstanding = {
  status: 'ready' | 'running' | 'failed' | 'pending'
  version?: string
  model: string
  device?: string
  frame_count?: number
  scenes: SceneItem[]
  error?: string
}

export type Run = {
  id: string; name: string; bytes: number; duration: number; created: string
  status: 'running' | 'validating' | 'ready' | 'failed'; error?: string
  data?: Result; model?: string; label_source?: string; warnings?: string[]; manifest_match?: boolean; highlight_evidence?: HighlightEvidence[]
  action_metadata?: ActionMetadata[]
  highlight_details?: HighlightDetail[]
  proposal_stats?: ProposalStats
  scene_understanding?: SceneUnderstanding
}
type Settings = { models: Record<string, boolean>; labels: string[]; benchmark?: { valid: number; eligible: number; epic: number; fpv: number; date: string } }
type View = 'Overview' | 'Video Intelligence' | 'Analysis History' | 'Benchmark & Validation' | 'Settings'
const tabs = [
  { title: 'Overview', Icon: LayoutGrid }, { title: 'Video Intelligence', Icon: Film },
  { title: 'Analysis History', Icon: Clock3 }, { title: 'Benchmark & Validation', Icon: Activity },
  { title: 'Settings', Icon: Settings2 },
] as const
const endpoint = (id: string) => '/api/action/runs/' + id
const clock = (t: number) => `${Math.floor(t / 60)}:${(t % 60).toFixed(2).padStart(5, '0')}`
const errorText = (error: unknown) => error instanceof Error ? error.message : String(error)

const activityName = (label: string) => ({ rockclimbing: 'Rock climbing', skate: 'Skateboarding', ski: 'Skiing' }[label] ?? label)

function highlightExplanation(evidence?: HighlightEvidence, span?: Span, scenes?: SceneItem[], activity?: string) {
  if (!evidence || !span) return { rationale: 'Run this video again to see the saved ranking signals.', score: '', visual: null }

  const associatedScene = scenes?.find(s => s.timestamp >= span[0] - 1.0 && s.timestamp <= span[1] + 1.0)

  if (evidence.fallback) {
    return {
      rationale: 'Initial 4.0-second fallback window selected to satisfy minimum highlight duration requirement.',
      score: 'Fallback',
      visual: associatedScene?.caption ?? null,
    }
  }

  const isRecognized = evidence.verb && evidence.verb !== 'other'
  const actionLabel = isRecognized ? `${evidence.verb} ${evidence.noun}` : null
  const scorePct = evidence.confidence ? `${Math.round(evidence.confidence * 100)}%` : null

  let rationale = ''
  if (actionLabel) {
    rationale = `Selected based on high-salience recognized interaction “${actionLabel}”${scorePct ? ` (${scorePct} confidence)` : ''}, prioritized under the 15% budget.`
  } else {
    rationale = `Ranked as a salient ${activityName(activity || 'activity')} moment with high spatiotemporal feature energy${scorePct ? ` (${scorePct} candidate score)` : ''}, satisfying non-overlapping budget constraints.`
  }

  return {
    rationale,
    score: scorePct || 'High salience',
    visual: associatedScene?.caption ?? null,
  }
}

function getProposalClass(category?: string, isRecognized?: boolean): string {
  if (isRecognized || category === 'Recognized action') return 'eg-cat-recognized'
  if (category === 'Activity transition') return 'eg-cat-transition'
  if (category === 'Visually unclear interval') return 'eg-cat-unclear'
  return 'eg-cat-potential'
}

function getCategoryBadgeLabel(category?: string, isRecognized?: boolean): string {
  if (isRecognized || category === 'Recognized action') return 'Recognized Action'
  if (category === 'Activity transition') return 'Activity Transition'
  if (category === 'Visually unclear interval') return 'Visually Unclear'
  return 'Salient Activity'
}

function Review({ run }: { run: Run }) {
  const player = useRef<HTMLVideoElement>(null)
  const stopAt = useRef<number | null>(null)
  const [playhead, setPlayhead] = useState(0)
  const [highlight, setHighlight] = useState(0)
  const [copied, setCopied] = useState('')
  const [scenes, setScenes] = useState<SceneUnderstanding | undefined>(run.scene_understanding)
  const [scenesLoading, setScenesLoading] = useState(false)

  const data = run.data!
  const actions = [...data.actions].sort((a, b) => a.segment[0] - b.segment[0])
  const isRecognizedAction = (verb: string, noun: string) => verb !== 'other' || noun !== 'other'
  const hasRecognizedActions = actions.some(a => isRecognizedAction(a.verb, a.noun))

  const [selectedMoment, setSelectedMoment] = useState<{
    start: number
    end: number
    highlightIndex?: number
    actionIndex?: number
  }>(() => {
    if (data.highlights.length > 0) {
      return { start: data.highlights[0].segment[0], end: data.highlights[0].segment[1], highlightIndex: 0 }
    }
    if (actions.length > 0) {
      return { start: actions[0].segment[0], end: actions[0].segment[1], actionIndex: 0 }
    }
    return { start: 0, end: Math.min(4.0, data.duration_sec) }
  })

  const laneEnds: number[] = []
  const actionLanes = actions.map(action => {
    const openLane = laneEnds.findIndex(end => end <= action.segment[0])
    const lane = openLane >= 0 ? openLane : laneEnds.length
    laneEnds[lane] = action.segment[1]
    return lane
  })

  useEffect(() => {
    setScenes(run.scene_understanding)
    if (!run.scene_understanding || run.scene_understanding.status === 'pending' || run.scene_understanding.version !== '2.0') {
      setScenesLoading(true)
      api<SceneUnderstanding>(endpoint(run.id) + '/scenes')
        .then(res => setScenes(res))
        .catch(() => {})
        .finally(() => setScenesLoading(false))
    }
  }, [run.id, run.scene_understanding])

  function seek(start: number, end?: number) {
    if (!player.current) return
    stopAt.current = end ?? null
    player.current.currentTime = start
    if (end !== undefined) void player.current.play().catch(() => {})
  }
  const spanStyle = ([start, end]: Span) => ({
    left: Math.min(100, 100 * start / data.duration_sec) + '%',
    width: Math.max(0, Math.min(100 * (end - start) / data.duration_sec, 100 - 100 * start / data.duration_sec)) + '%',
  })
  return <div className="eg-review">
    <div>
      {/* 1. Video Player */}
      <video className="eg-player" ref={player} controls playsInline preload="metadata" src={endpoint(run.id)+'/video'}
        onTimeUpdate={()=>{const v=player.current!; setPlayhead(v.currentTime); if(stopAt.current!==null&&v.currentTime>=stopAt.current){const end=stopAt.current;stopAt.current=null;v.pause();v.currentTime=end;setPlayhead(end)}}}/>
      
      {/* 2. Proposal / Highlight Timeline */}
      <div className="eg-timeline" aria-label="Video timeline">
        {actions.length > 0 && <div className="eg-track"><span className="eg-track-name">{hasRecognizedActions ? 'Actions' : 'Proposals'}</span><div style={{height: Math.max(13, laneEnds.length * 11 + 2)}}>
          {actions.map(({segment,verb,noun,score},i)=>{
            const meta = run.action_metadata?.[i]
            const isRec = isRecognizedAction(verb, noun)
            const catClass = getProposalClass(meta?.category, isRec)
            const catLabel = getCategoryBadgeLabel(meta?.category, isRec)
            return (
              <button className={`eg-action-mark ${catClass}`} key={i} style={{...spanStyle(segment), top: 1 + actionLanes[i] * 11, height: 10}}
                onClick={()=>{ setSelectedMoment({ start: segment[0], end: segment[1], actionIndex: i }); seek(segment[0], segment[1]); }}
                title={isRec ? `${verb} ${noun} · ${clock(segment[0])}–${clock(segment[1])}` : `Proposal ${i+1} [${catLabel}] (${Math.round(score*100)}%) · ${clock(segment[0])}–${clock(segment[1])}`}
                aria-label={isRec ? `Seek to ${verb} ${noun} at ${clock(segment[0])}` : `Seek to proposal ${i+1} at ${clock(segment[0])}`}/>
            )
          })}
        </div></div>}
        <div className="eg-track"><span className="eg-track-name">Highlights</span><div>
          {data.highlights.map(({segment},i)=><button className="eg-highlight-mark" key={i} style={spanStyle(segment)}
            onClick={()=>{ setHighlight(i); setSelectedMoment({ start: segment[0], end: segment[1], highlightIndex: i }); seek(...segment); }}
            title={`Play highlight ${i+1}`} aria-label={`Play highlight ${i+1}`}/>)}
        </div></div>
        <div className="eg-axis"><span>0:00</span><span>{clock(playhead)} / {clock(data.duration_sec)}</span></div>
        <div className="eg-timeline-legend">
          <span className="eg-leg-item"><span className="eg-dot eg-dot-rec" /> Recognized Action</span>
          <span className="eg-leg-item"><span className="eg-dot eg-dot-act" /> Salient Activity</span>
          <span className="eg-leg-item"><span className="eg-dot eg-dot-trn" /> Activity Transition</span>
          <span className="eg-leg-item"><span className="eg-dot eg-dot-hl" /> Selected Highlight</span>
        </div>
        <input className="eg-scrub" type="range" min={0} max={data.duration_sec} step={0.01} value={Math.min(playhead, data.duration_sec)}
          aria-label="Scrub video" onChange={e=>seek(e.currentTarget.valueAsNumber)}/>
      </div>

      {/* 3. AI Scene Understanding */}
      <section className="eg-section eg-scene-understanding" aria-label="AI Scene Understanding">
        <div className="eg-scene-header">
          <div>
            <h2>AI Scene Understanding {scenes?.scenes?.length ? <span>{scenes.scenes.length} moments</span> : null}</h2>
            <p className="eg-scene-subtitle">Explore what is visually happening throughout your video.</p>
          </div>
          <span className="eg-scene-tag" title="Pretrained Salesforce/blip-image-captioning-base visual perception">
            <Sparkles size={13}/> BLIP Vision-Language v2.0
          </span>
        </div>

        {scenesLoading && (
          <div className="eg-scene-status" role="status">
            <Loader2 size={16} className="eg-spin"/>
            <span>Examining representative frames from your video...</span>
          </div>
        )}

        {!scenesLoading && scenes?.status === 'failed' && (
          <div className="eg-scene-fallback" role="alert">
            <AlertCircle size={15}/>
            <span>Scene descriptions could not be generated. Your video analysis and highlights are still available.</span>
          </div>
        )}

        {scenes?.scenes && scenes.scenes.length > 0 && (
          <div className="eg-scene-grid">
            {scenes.scenes.map((scene, idx) => {
              const isActive = Math.abs(playhead - scene.timestamp) < 2.5
              return (
                <div
                  key={idx}
                  className={`eg-scene-card ${isActive ? 'eg-active' : ''}`}
                  onClick={() => {
                    seek(scene.timestamp)
                    const hlI = data.highlights.findIndex(h => scene.timestamp >= h.segment[0] - 0.5 && scene.timestamp <= h.segment[1] + 0.5)
                    const actI = actions.findIndex(a => scene.timestamp >= a.segment[0] - 0.5 && scene.timestamp <= a.segment[1] + 0.5)
                    if (hlI >= 0) {
                      setHighlight(hlI)
                      setSelectedMoment({ start: data.highlights[hlI].segment[0], end: data.highlights[hlI].segment[1], highlightIndex: hlI })
                    } else if (actI >= 0) {
                      setSelectedMoment({ start: actions[actI].segment[0], end: actions[actI].segment[1], actionIndex: actI })
                    } else {
                      setSelectedMoment({ start: Math.max(0, scene.timestamp - 2.0), end: Math.min(data.duration_sec, scene.timestamp + 2.0) })
                    }
                  }}
                  role="button"
                  tabIndex={0}
                  onKeyDown={e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); seek(scene.timestamp); } }}
                  title={`Jump to ${scene.timestamp_formatted}`}
                >
                  <div className="eg-scene-thumb-wrap">
                    {scene.thumbnail ? (
                      <img src={scene.thumbnail} alt={`Visual frame at ${scene.timestamp_formatted}`} loading="lazy" />
                    ) : (
                      <div className="eg-scene-thumb-placeholder"><Film size={18}/></div>
                    )}
                    <span className="eg-scene-time">{scene.timestamp_formatted}</span>
                    {scene.quality && scene.quality !== 'Clear visual description' && (
                      <span className={`eg-scene-quality ${
                        scene.quality === 'Title / intro screen' ? 'eg-q-title' :
                        scene.quality === 'Uncertain interpretation' ? 'eg-q-uncertain' : 'eg-q-limited'
                      }`}>
                        {scene.quality === 'Title / intro screen' ? 'Intro Screen' :
                         scene.quality === 'Uncertain interpretation' ? 'Uncertain' : 'Low Visibility'}
                      </span>
                    )}
                    {scene.highlight_index !== null && scene.highlight_index !== undefined ? (
                      <span className="eg-scene-badge">Highlight #{scene.highlight_index + 1}</span>
                    ) : scene.context ? (
                      <span className="eg-scene-badge">{scene.context}</span>
                    ) : null}
                  </div>
                  <div className="eg-scene-body">
                    <p className="eg-scene-caption">“{scene.caption}”</p>
                  </div>
                </div>
              )
            })}
          </div>
        )}
      </section>

      {/* 4. Smart Highlights */}
      <div className="eg-section">
        <h2>Smart Highlights <span>{data.highlights.length}</span></h2>
        <div className="eg-highlight-list">
          {data.highlights.map(({ segment }, i) => {
            const ev = run.highlight_evidence?.[i]
            const hlDetail = run.highlight_details?.[i]
            const expl = highlightExplanation(ev, segment, scenes?.scenes, data.label.name)
            const isCurrent = highlight === i
            const rationaleText = hlDetail?.rationale || expl.rationale
            return (
              <div key={i} className={`eg-highlight-card ${isCurrent ? 'eg-active' : ''}`} onClick={() => { setHighlight(i); setSelectedMoment({ start: segment[0], end: segment[1], highlightIndex: i }); seek(...segment); }}>
                <div className="eg-highlight-top">
                  <div className="eg-highlight-meta">
                    <Film size={16} />
                    <strong>Highlight {i + 1}</strong>
                    <time>{clock(segment[0])}–{clock(segment[1])}</time>
                    <span className="eg-hl-dur">{(segment[1] - segment[0]).toFixed(1)}s</span>
                    {hlDetail?.related_proposal_index && (
                      <span className="eg-hl-rel-pill" title="Linked to temporal proposal">
                        Proposal #{hlDetail.related_proposal_index}
                      </span>
                    )}
                  </div>
                  {expl.score && <span className="eg-hl-score">{expl.score}</span>}
                </div>
                {expl.visual && (
                  <p className="eg-hl-visual">
                    <Sparkles size={13} /> “{expl.visual}”
                  </p>
                )}
                <p className="eg-hl-rationale">{rationaleText}</p>
              </div>
            )
          })}
        </div>
      </div>

      {/* 5. Unified Moment Intelligence */}
      <section className="eg-moment-panel" aria-label="Unified Moment Intelligence">
        {(() => {
          const mStart = selectedMoment.start
          const mEnd = selectedMoment.end
          const mDur = mEnd - mStart
          const midSec = (mStart + mEnd) / 2

          // Highlight matching
          const matchedHlIdx = selectedMoment.highlightIndex !== undefined
            ? selectedMoment.highlightIndex
            : data.highlights.findIndex(h => Math.max(h.segment[0], mStart) < Math.min(h.segment[1], mEnd))
          const isHl = matchedHlIdx >= 0 && matchedHlIdx < data.highlights.length
          const matchedHl = isHl ? data.highlights[matchedHlIdx] : null
          const hlDetail = isHl ? run.highlight_details?.[matchedHlIdx] : null
          const hlEvidence = isHl ? run.highlight_evidence?.[matchedHlIdx] : null
          const hlExpl = isHl && matchedHl ? highlightExplanation(hlEvidence || undefined, matchedHl.segment, scenes?.scenes, data.label.name) : null

          // Proposal matching
          const matchedActIdx = selectedMoment.actionIndex !== undefined
            ? selectedMoment.actionIndex
            : actions.findIndex(a => Math.max(a.segment[0], mStart) < Math.min(a.segment[1], mEnd))
          const isAct = matchedActIdx >= 0 && matchedActIdx < actions.length
          const matchedAct = isAct ? actions[matchedActIdx] : null
          const actMeta = isAct ? run.action_metadata?.[matchedActIdx] : null
          const isActRec = matchedAct ? isRecognizedAction(matchedAct.verb, matchedAct.noun) : false
          const actCatLabel = isAct ? getCategoryBadgeLabel(actMeta?.category, isActRec) : null
          const actCatClass = isAct ? getProposalClass(actMeta?.category, isActRec) : ''

          // Scene matching
          const matchedScene = scenes?.scenes?.find(s => s.timestamp >= mStart - 0.75 && s.timestamp <= mEnd + 0.75)
            || scenes?.scenes?.slice().sort((a, b) => Math.abs(a.timestamp - midSec) - Math.abs(b.timestamp - midSec))[0]

          return (
            <>
              <div className="eg-moment-header">
                <div className="eg-moment-title">
                  <Eye className="eg-moment-icon" size={20} />
                  <h3>Unified Moment Intelligence</h3>
                  <span className="eg-moment-span">{clock(mStart)}–{clock(mEnd)} ({mDur.toFixed(1)}s)</span>
                </div>
                <div className="eg-moment-badges">
                  {isHl ? (
                    <span className="eg-moment-hl-badge">
                      Highlight #{matchedHlIdx + 1} of {data.highlights.length}
                    </span>
                  ) : (
                    <span className="eg-moment-nonhl-badge">Timeline Interval</span>
                  )}
                  {isAct && (
                    <span className={`eg-cat-badge ${actCatClass}`}>
                      {actCatLabel}
                    </span>
                  )}
                </div>
              </div>

              <div className="eg-moment-grid">
                {/* 1. Activity Recognition */}
                <div className="eg-moment-cell">
                  <span className="eg-moment-label">1. Activity Recognition</span>
                  <div className="eg-moment-val">
                    <strong>{data.label.name === 'other' && !data.label.score ? 'Activity uncertain' : activityName(data.label.name)}</strong>
                    <span className="eg-moment-sub">
                      {data.label.score ? `${Math.round(data.label.score * 100)}% video-level score` : 'Model abstained'} · {run.model || 'R3D-18 Transfer'}
                    </span>
                  </div>
                </div>

                {/* 2. Action Recognition */}
                <div className="eg-moment-cell">
                  <span className="eg-moment-label">2. Action Recognition</span>
                  <div className="eg-moment-val">
                    {hasRecognizedActions && isActRec && matchedAct ? (
                      <>
                        <strong className="eg-text-emerald">“{matchedAct.verb} {matchedAct.noun}”</strong>
                        <span className="eg-moment-sub">
                          {Math.round(matchedAct.score * 100)}% confidence · Trained EPIC-KITCHENS interaction head
                        </span>
                      </>
                    ) : (
                      <>
                        <strong className="eg-text-muted">Domain interaction head inactive</strong>
                        <span className="eg-moment-sub">
                          Fine-grained verb–noun heads are trained on EPIC-KITCHENS. Video actions default to other/other to avoid false claims.
                        </span>
                      </>
                    )}
                  </div>
                </div>

                {/* 3. Temporal Action Proposal */}
                <div className="eg-moment-cell">
                  <span className="eg-moment-label">3. Temporal Action Proposal</span>
                  <div className="eg-moment-val">
                    {isAct && matchedAct ? (
                      <>
                        <div className="eg-action-header-row">
                          <strong>{isActRec ? `${matchedAct.verb} ${matchedAct.noun}` : `Proposal ${matchedActIdx + 1}`}</strong>
                          <span className={`eg-cat-badge ${actCatClass}`}>{actCatLabel}</span>
                          {actMeta?.quality_rank && <span className="eg-rank-badge">Rank #{actMeta.quality_rank}</span>}
                        </div>
                        <span className="eg-moment-sub">
                          {actMeta?.rationale || 'Spatiotemporal feature activation detected during this interval.'}
                        </span>
                      </>
                    ) : (
                      <>
                        <strong className="eg-text-muted">No proposal peak</strong>
                        <span className="eg-moment-sub">
                          Interval falls outside top non-maximum-suppressed proposal clusters.
                        </span>
                      </>
                    )}
                  </div>
                </div>

                {/* 4. Visual Scene Evidence */}
                <div className="eg-moment-cell eg-moment-cell-scene">
                  <span className="eg-moment-label">4. Visual Scene Evidence</span>
                  {matchedScene ? (
                    <div className="eg-moment-scene-content">
                      {matchedScene.thumbnail ? (
                        <img className="eg-moment-thumb" src={matchedScene.thumbnail} alt={`Visual frame at ${matchedScene.timestamp_formatted}`} />
                      ) : (
                        <div className="eg-scene-thumb-placeholder" style={{ width: 120, height: 68 }}><Film size={18} /></div>
                      )}
                      <div className="eg-moment-scene-text">
                        <div className="eg-action-header-row" style={{ marginBottom: 4 }}>
                          <span className="eg-moment-span" style={{ marginLeft: 0 }}>Sampled frame at {matchedScene.timestamp_formatted}</span>
                          {matchedScene.quality && matchedScene.quality !== 'Clear visual description' && (
                            <span className="eg-scene-quality eg-q-uncertain" style={{ position: 'static' }}>
                              {matchedScene.quality}
                            </span>
                          )}
                          <span className="eg-scene-tag" style={{ fontSize: '0.68rem', padding: '1px 6px' }}>
                            <Sparkles size={11} /> BLIP v2.0
                          </span>
                        </div>
                        <p className="eg-moment-caption">“{matchedScene.caption}”</p>
                        {matchedScene.context && (
                          <span className="eg-moment-sub" style={{ marginTop: 2, display: 'block' }}>
                            Visual Context: {matchedScene.context}
                          </span>
                        )}
                      </div>
                    </div>
                  ) : (
                    <p className="eg-moment-sub">Visual frame observation pending or outside sampled intervals.</p>
                  )}
                </div>

                {/* 5 & 6. Highlight Selection & Transparent Rationale */}
                <div className="eg-moment-cell eg-moment-cell-hl">
                  <span className="eg-moment-label">5 &amp; 6. Highlight Selection &amp; Transparent Rationale</span>
                  <div className="eg-moment-val">
                    {isHl && matchedHl ? (
                      <>
                        <div className="eg-action-header-row">
                          <strong className="eg-text-mint">Selected for Track 4 Highlight Submission (Highlight #{matchedHlIdx + 1})</strong>
                          <span className="eg-hl-dur">{(matchedHl.segment[1] - matchedHl.segment[0]).toFixed(1)}s</span>
                          {hlDetail?.related_proposal_index && (
                            <span className="eg-hl-rel-pill">Linked to Proposal #{hlDetail.related_proposal_index}</span>
                          )}
                        </div>
                        <p style={{ margin: '4px 0 0', fontSize: '0.84rem', color: 'var(--eg-text)' }}>
                          <strong>Selection Rationale:</strong> {hlDetail?.rationale || hlExpl?.rationale || 'Ranked in top highlights based on feature salience, motion dynamics, and temporal diversity.'}
                        </p>
                        <span className="eg-reason-constraint" style={{ marginTop: 2 }}>
                          Satisfies official Track 4 non-overlapping constraint within {Math.round(data.duration_sec * 0.15)}s maximum duration budget (&le;15%).
                        </span>
                      </>
                    ) : (
                      <>
                        <strong className="eg-text-muted">Not selected for highlight summary</strong>
                        <p style={{ margin: '4px 0 0', fontSize: '0.82rem', color: 'var(--eg-muted)' }}>
                          <strong>Selection Rationale:</strong> The highlight selector prioritized intervals with higher multi-signal utility (feature energy, motion magnitude, and temporal diversity across the full timeline) under the strict 15% duration ceiling.
                        </p>
                      </>
                    )}
                  </div>
                </div>

                {/* 7. Model Attribution & Evidence Limitations */}
                <div className="eg-moment-cell eg-moment-cell-attr">
                  <span className="eg-moment-label">7. Model Attribution &amp; System Transparency</span>
                  <p className="eg-limits" style={{ margin: '4px 0 0' }}>
                    Activity: R3D-18 Backbone · Action Proposals: Dense Temporal Windowing &amp; Temporal NMS (IoU 0.30) · Captions: Salesforce BLIP Base v2.0 · Selection: Multi-Signal Utility Optimization.
                    <br />
                    <em>Technical note: Egocentric wearable cameras experience rapid ego-motion. Proposals indicate high-salience intervals rather than segmented bounding boxes. No hidden ground-truth annotations are accessed.</em>
                  </p>
                </div>
              </div>
            </>
          )
        })()}
      </section>

      {/* 6. Output JSON */}
      <details className="eg-section"><summary><FileJson size={17}/> Output JSON</summary>
        <div className="eg-actions"><button className="eg-button" onClick={async()=>{
          try{const r=await fetch(endpoint(run.id)+'/result');if(!r.ok)throw Error();await navigator.clipboard.writeText(await r.text());setCopied('Copied')}
          catch {setCopied('Copy unavailable')}
        }}>{copied||'Copy JSON'}</button><a className="eg-button" href={endpoint(run.id)+'/result'}><ArrowDownToLine size={15}/> Download JSON</a></div>
        <pre className="eg-json">{JSON.stringify(data,null,2).split(/("(?:[^"\\]|\\.)*"|\b\d+(?:\.\d+)?\b)/g).map((token,i)=><span key={i} className={token.startsWith('"')?'eg-string':/^\d/.test(token)?'eg-number':undefined}>{token}</span>)}</pre>
      </details>
    </div>

    {/* Sidebar Evidence */}
    <aside className="eg-evidence">
      <h2>Activity</h2><p className="eg-label">{run.label_source === 'Model prediction' && data.label.name === 'other' && !data.label.score ? 'Activity uncertain' : activityName(data.label.name)}</p>
      <p>{run.label_source === 'Model prediction' ? data.label.name==='other'&&data.label.score===0?'The full-video model abstained.':`${Math.round(data.label.score*100)}% full-video score` : 'Category supplied by the benchmark guide'}</p>
      <p className="eg-model">{run.model}</p>
      <p className="eg-valid"><CheckCircle2 size={16}/> Local validation passed</p>
      <p>Checked with the team-created Track 4 validator.</p>
      {!run.manifest_match && <p>Development clip. This output is outside the Samsung manifest.</p>}
      {run.warnings?.map(w=><p className="eg-warning" key={w}>{w}</p>)}
      
      <h2>{hasRecognizedActions ? 'Recognized Actions' : 'Action Proposals'} <span>{actions.length}</span></h2>
      {!actions.length ? <p>This model scores activities and selects highlights. Timed verb–noun actions are available on the EPIC-KITCHENS path.</p> :
      <>
        {!hasRecognizedActions && <p style={{fontSize: '.78rem', color: 'var(--eg-muted)', margin: '4px 0 8px'}}>
          Localized temporal intervals ranked by feature energy and motion dynamics. Redundant overlapping windows are suppressed.
        </p>}
        {run.proposal_stats && (
          <div className="eg-proposal-stats-pill">
            <span>{run.proposal_stats.retained ?? actions.length} retained</span>
            {run.proposal_stats.suppressed_nms ? <span> · {run.proposal_stats.suppressed_nms} suppressed redundant</span> : null}
            {run.proposal_stats.suppressed_low_info ? <span> · {run.proposal_stats.suppressed_low_info} low info</span> : null}
          </div>
        )}
        <div className="eg-action-list">{actions.map(({segment,verb,noun,score},i)=>{
          const isRec = isRecognizedAction(verb, noun)
          const meta = run.action_metadata?.[i]
          const catLabel = getCategoryBadgeLabel(meta?.category, isRec)
          const catClass = getProposalClass(meta?.category, isRec)
          const sceneContext = scenes?.scenes?.find(s => s.timestamp >= segment[0] && s.timestamp <= segment[1])
          return (
            <button className="eg-row eg-action-row" key={i} onClick={() => { setSelectedMoment({ start: segment[0], end: segment[1], actionIndex: i }); seek(segment[0], segment[1]); }}>
              <div className="eg-action-main">
                <span className="eg-action-idx">{i+1}</span>
                <div className="eg-action-text">
                  <div className="eg-action-header-row">
                    <strong>{isRec ? `${verb} ${noun}` : `Proposal ${i+1}`}</strong>
                    <span className={`eg-cat-badge ${catClass}`}>{catLabel}</span>
                    {meta?.quality_rank && <span className="eg-rank-badge">Rank #{meta.quality_rank}</span>}
                  </div>
                  <span className="eg-action-sub">{isRec ? 'Trained Interaction Head' : 'Spatiotemporal Interval'}</span>
                  {meta?.rationale && <span className="eg-action-rationale">{meta.rationale}</span>}
                  {sceneContext && (
                    <span className="eg-action-ctx" title={sceneContext.caption}>
                      <Sparkles size={11} /> {sceneContext.caption}
                    </span>
                  )}
                </div>
              </div>
              <div className="eg-action-meta">
                <time>{clock(segment[0])}–{clock(segment[1])}</time>
                <span className="eg-action-score">{Math.round(score*100)}%</span>
              </div>
            </button>
          )
        })}</div>
      </>}
    </aside>
  </div>
}

export function EgoSight() {
  const [view, setView] = useState<View>('Overview')
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState('')
  const [duration, setDuration] = useState<number | null>(null)
  const [error, setError] = useState('')
  const [uploading, setUploading] = useState(false)
  const [uploadProgress, setUploadProgress] = useState<UploadProgress>({ percent: 0, loaded: 0, total: 0 })
  const [uploadStatus, setUploadStatus] = useState('Preparing video')
  const [run, setRun] = useState<Run>()
  const [history, setHistory] = useState<Run[]>([])
  const [settings, setSettings] = useState<Settings>()
  const [offline, setOffline] = useState(false)

  const input = useRef<HTMLInputElement>(null)
  const abortControllerRef = useRef<AbortController | null>(null)
  const uploadingRef = useRef(false)
  const active = run?.status === 'running' || run?.status === 'validating'

  function loadHistory() {
    return api<Run[]>('/api/action/runs').then(setHistory).catch(() => setOffline(true))
  }

  useEffect(() => {
    api<Settings>('/api/action/settings').then(setSettings).catch(() => setOffline(true))
    void loadHistory()

    const parseHash = () => {
      const match = window.location.hash.match(/run=([a-f0-9]+)/i)
      if (match) {
        api<Run>(endpoint(match[1]))
          .then(openRun)
          .catch(() => {})
      }
    }
    parseHash()
    window.addEventListener('hashchange', parseHash)
    return () => window.removeEventListener('hashchange', parseHash)
  }, [])

  useEffect(() => {
    setDuration(null)
    if (!file) {
      setPreview('')
      return
    }
    const url = URL.createObjectURL(file)
    setPreview(url)
    return () => URL.revokeObjectURL(url)
  }, [file])

  useEffect(() => {
    if (!run || !active) return
    let live = true
    const timer = setInterval(async () => {
      try {
        const next = await api<Run>(endpoint(run.id))
        if (live) {
          setRun(next)
          setError('')
          if (next.status === 'ready' || next.status === 'failed') void loadHistory()
        }
      } catch (e) {
        if (live) setError(errorText(e))
      }
    }, 900)
    return () => {
      live = false
      clearInterval(timer)
    }
  }, [run?.id, active])

  function choose(f?: File) {
    if (!f || active || uploading) return
    if (!/\.mp4$/i.test(f.name) || f.size > 500 * 1024 * 1024) {
      setError('Choose an MP4 smaller than 500 MB.')
      return
    }
    setError('')
    setFile(f)
    setRun(undefined)
    setView('Video Intelligence')
    void startUpload(f)
  }

  async function startUpload(f: File) {
    if (uploadingRef.current) return
    uploadingRef.current = true
    setUploading(true)
    setError('')
    setUploadProgress({ percent: 0, loaded: 0, total: f.size })
    setUploadStatus('Preparing video')

    const controller = new AbortController()
    abortControllerRef.current = controller

    const body = new FormData()
    body.append('video', f)

    try {
      setUploadStatus('Uploading video')
      const newRun = await uploadWithProgress<Run>(
        '/api/action/runs',
        body,
        p => {
          setUploadProgress(p)
          if (p.percent < 100) {
            setUploadStatus(`Uploading video (${p.percent}%)`)
          } else {
            setUploadStatus('Upload complete · Starting AI analysis')
          }
        },
        controller.signal
      )
      setRun(newRun)
      setUploadStatus('Analysis started')
      void loadHistory()
    } catch (err) {
      if (err instanceof DOMException && err.name === 'AbortError') {
        setError('Upload cancelled.')
      } else {
        setError(errorText(err))
      }
      setFile(null)
    } finally {
      setUploading(false)
      uploadingRef.current = false
      abortControllerRef.current = null
    }
  }

  function cancelUpload() {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
    }
    setUploading(false)
    uploadingRef.current = false
    setFile(null)
    setUploadProgress({ percent: 0, loaded: 0, total: 0 })
  }

  function openRun(r: Run) {
    setRun(r)
    setFile(null)
    setError('')
    setView('Video Intelligence')
    try {
      if (window.location.hash !== '#run=' + r.id) {
        window.location.hash = 'run=' + r.id
      }
    } catch {}
  }

  function browse() {
    setView('Video Intelligence')
    input.current?.click()
  }

  return <div className="egosight">
    <nav className="eg-nav" aria-label="EgoSight workspace">
      <button className="eg-brand" onClick={() => setView('Overview')}>
        <img src="/egosight.svg" alt=""/>
        <span>EgoSight <b>XR</b></span>
      </button>
      <div>{tabs.map(({ title, Icon }) => <button key={title} aria-current={view === title ? 'page' : undefined} onClick={() => setView(title)}><Icon size={17}/>{title}</button>)}</div>
      <small>Team Rajasthani · Track 4</small>
    </nav>
    <main className="eg-main">
      <input hidden ref={input} type="file" accept=".mp4,video/mp4" onChange={e => { choose(e.target.files?.[0]); e.target.value = '' }} aria-label="Choose an MP4 video"/>

      {view === 'Overview' && <>
        <section className="eg-hero">
          <div>
            <h1>See the Actions.<br/>Understand the Moment.</h1>
            <p>Review first-person footage by action. Jump to the moment you need, replay highlights, explore AI scene descriptions, and export the model’s predictions.</p>
            <button className="eg-button eg-primary" onClick={browse}><UploadCloud size={18}/> Upload video <ArrowRight size={16}/></button>
          </div>
          <figure><img src="/review-cover.jpg" alt="Illustration of hands setting an oven timer from a first-person perspective"/><figcaption>Concept illustration</figcaption></figure>
        </section>
        <section className="eg-overview">
          <h2>Recent analyses</h2>
          {history.length ? history.slice(0, 4).map(r => <button className="eg-row" key={r.id} onClick={() => openRun(r)}>
            <Film size={17}/><strong>{r.name}</strong><time>{r.data?.label.name ?? r.status}</time><ArrowRight size={16}/></button>) : <p>Upload your first MP4 to start a review.</p>}
        </section>
      </>}

      {view === 'Video Intelligence' && <>
        <header className="eg-heading">
          <div>
            <h1>Video Intelligence</h1>
            {run && <p>{run.name}</p>}
          </div>
          {(file || run) && <button className="eg-button" disabled={active || uploading} onClick={browse}>Replace video</button>}
        </header>

        {error && <p className="eg-warning" role="alert">{error}</p>}

        {/* Real-time upload progress card */}
        {uploading && (
          <div className="eg-upload-card" role="status">
            <div className="eg-upload-header">
              <div className="eg-upload-title">
                <Film size={20}/>
                <strong>{file?.name || 'Selected video'}</strong>
              </div>
              <span className="eg-upload-meta">
                {file ? `${(file.size / (1024 * 1024)).toFixed(1)} MB` : ''}
              </span>
            </div>
            <div className="eg-progress-track">
              <div className="eg-progress-bar" style={{ width: `${uploadProgress.percent}%` }} />
            </div>
            <div className="eg-progress-footer">
              <span className="eg-status-badge">
                <Loader2 size={13} className="eg-spin"/> {uploadStatus}
              </span>
              <span>
                {(uploadProgress.loaded / (1024 * 1024)).toFixed(1)} MB of {(uploadProgress.total / (1024 * 1024)).toFixed(1)} MB ({uploadProgress.percent}%)
              </span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: 4 }}>
              <button className="eg-cancel-button" onClick={cancelUpload}>Cancel upload</button>
            </div>
          </div>
        )}

        {/* Drop zone when idle */}
        {!file && !run && !uploading && (
          <div className="eg-drop" onDragOver={e => e.preventDefault()} onDrop={e => { e.preventDefault(); choose(e.dataTransfer.files[0]) }}>
            <UploadCloud size={36}/>
            <h2>Drop an MP4 here</h2>
            <p>MP4 up to 500 MB. At least 4 seconds.</p>
            <button className="eg-button eg-primary" onClick={browse}>Browse files</button>
          </div>
        )}

        {/* Active Analysis Processing Screen */}
        {run && active && (
          <div className="eg-processing-card" role="status">
            {preview && (
              <video className="eg-player" controls playsInline muted src={preview} />
            )}
            <div className="eg-processing-box">
              <div className="eg-processing-header">
                <Activity size={24} className="eg-spin" style={{ color: 'var(--eg-accent)' }} />
                <div>
                  <h2>Analyzing Video with EgoSight XR</h2>
                  <p>{run.name} · {duration !== null ? `${clock(duration)} · ` : ''}{(run.bytes / (1024 * 1024)).toFixed(1)} MB</p>
                </div>
              </div>
              <div className="eg-progress-track">
                <div className="eg-progress-bar eg-indeterminate" />
              </div>
              <div className="eg-stepper">
                <div className="eg-step-item eg-done">
                  <CheckCircle2 size={15}/> <span>Upload & Video Decoding completed</span>
                </div>
                <div className={`eg-step-item ${run.status === 'running' ? 'eg-current' : 'eg-done'}`}>
                  {run.status === 'running' ? <Loader2 size={15} className="eg-spin"/> : <CheckCircle2 size={15}/>}
                  <span>Spatiotemporal Feature Extraction (R3D-18 / 3D CNN)</span>
                </div>
                <div className={`eg-step-item ${run.status === 'validating' ? 'eg-current' : ''}`}>
                  {run.status === 'validating' ? <Loader2 size={15} className="eg-spin"/> : <Clock3 size={15}/>}
                  <span>Benchmark Contract Validation & Highlight Candidate Scoring</span>
                </div>
                <div className="eg-step-item">
                  <Sparkles size={15}/> <span>BLIP Visual Scene Understanding</span>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Failed Run Screen */}
        {run?.status === 'failed' && (
          <div className="eg-upload-card" style={{ borderColor: '#683a3a', background: '#221515' }} role="alert">
            <div className="eg-upload-header">
              <div className="eg-upload-title" style={{ color: '#ffb3b3' }}>
                <AlertCircle size={20}/>
                <strong>Analysis Encountered an Error</strong>
              </div>
            </div>
            <p style={{ margin: 0, color: 'var(--eg-muted)', fontSize: '0.88rem' }}>{run.error || 'The model could not process this video.'}</p>
            <div style={{ display: 'flex', gap: 10, marginTop: 8 }}>
              <button className="eg-button eg-primary" onClick={browse}>Try another video</button>
            </div>
          </div>
        )}

        {/* Ready Run Review */}
        {run?.status === 'ready' && <Review key={run.id} run={run} />}
      </>}

      {view === 'Analysis History' && <>
        <h1>Analysis History</h1>
        {history.length ? history.map(r => <button className="eg-row" key={r.id} onClick={() => openRun(r)}>
          <Film size={17}/><strong>{r.name}</strong><time>{new Date(r.created).toLocaleDateString()}</time><span>{r.data ? `${r.data.label.name} · ${r.data.actions.length} ${r.data.actions.some(a => a.verb !== 'other' || a.noun !== 'other') ? 'actions' : 'proposals'} · ${r.data.highlights.length} highlights` : r.status}</span>
        </button>) : <p>No analyses saved yet.</p>}
      </>}

      {view === 'Benchmark & Validation' && <>
        <h1>Benchmark & Validation</h1>
        <p>Samsung’s manifest lists 138 EPIC-KITCHENS and 49 FPVSum clips. Two unavailable FPVSum clips were excluded, leaving 185 eligible videos.</p>
        {settings?.benchmark && <p><strong>{settings.benchmark.valid} of {settings.benchmark.eligible}</strong> outputs passed the local checker on {settings.benchmark.date}. The staging ZIP includes {settings.benchmark.epic} EPIC-KITCHENS and {settings.benchmark.fpv} FPVSum outputs.</p>}
        <p>EPIC-KITCHENS tests timed verb–noun predictions. FPVSum tests video categories and highlights. Ground-truth actions count toward coverage when at least half their duration falls inside the selected highlights.</p>
        <h2>Supported video labels</h2>
        <p>{settings?.labels.join(', ') || 'Connect the local backend to read the vocabulary.'}</p>
        <a href={guide}>Read the model and validation documentation</a>
      </>}

      {view === 'Settings' && <>
        <h1>Settings</h1>
        <p>Local backend · {offline ? 'Connection unavailable' : settings ? 'Connected' : 'Checking connection'}</p>
        <h2>Model files</h2>
        {settings && Object.entries(settings.models).map(([name, ready]) => <p key={name}>{name} · {ready ? 'Available locally' : 'Weights missing'}</p>)}
        <p>BLIP Scene Understanding · Available (Salesforce/blip-image-captioning-base v2.0)</p>
        <a href={guide}>Checkpoint setup</a>
        <p>Team Rajasthani · Vellore Institute of Technology, Vellore</p>
      </>}
    </main>
  </div>
}
