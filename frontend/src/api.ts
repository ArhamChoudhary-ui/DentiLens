export type State = 'queued' | 'running' | 'ready' | 'failed'
export interface MediaInfo {
  width: number
  height: number
  fps: number
  frames: number
  duration: number
}
export interface Finding {
  id: string
  time: number
  frame: number
  category: string
  label: string
  track_id: string | null
  confidence: number
  image: string
}
export interface Track {
  id: string
  start: number
  end: number
  label: string
  frames: number
  anchors: number
  fdi_number?: string | null
  tooth_type?: string | null
}
export interface Report {
  input: MediaInfo
  video: MediaInfo
  profile: string
  device: string
  cavity_review: boolean
  stats: {
    frames_decoded: number
    detection_frames: number
    annotated_frames: number
    tracked_frames: number
    numbered_frames: number
    cavity_frames: number
  }
  tracks: Track[]
  findings: Finding[]
  limitations: string[]
  tracking: string
}
export interface Review {
  id: string
  time: number
  finding_id: string | null
  label: string
  note: string
  decision: 'confirmed' | 'dismissed'
  flagged: boolean
}
export interface Exam {
  id: string
  title: string
  bytes: number
  created: string
  state: State
  profile: string
  stage: string
  progress: number
  metadata?: MediaInfo | null
  error?: string | null
  reel_url?: string | null
  reviews?: Review[]
  report?: Report
}

export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, init)
  if (!response.ok) {
    const error = await response.json().catch(() => null)
    throw new Error(typeof error?.detail === 'string' ? error.detail : `Request failed (${response.status}).`)
  }
  return response.json() as Promise<T>
}
export function send<T>(path: string, body: unknown): Promise<T> {
  return api<T>(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })
}
export function clock(seconds: number) {
  const whole = Math.floor(seconds)
  return `${Math.floor(whole / 60).toString().padStart(2, '0')}:${(whole % 60).toString().padStart(2, '0')}`
}
