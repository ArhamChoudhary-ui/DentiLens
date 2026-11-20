import { useEffect, useRef, useState } from 'react'
import {
  AmbientLight, Box3, Color, DirectionalLight, Mesh, MeshStandardMaterial, PerspectiveCamera,
  Raycaster, Scene, SRGBColorSpace, Vector2, Vector3, WebGLRenderer,
} from 'three'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js'
import { ToothHistory } from './intelligence'

const quadrants = ['upper-right', 'upper-left', 'lower-left', 'lower-right']
const types = ['central-incisor', 'lateral-incisor', 'canine', 'first-premolar', 'second-premolar', 'first-molar', 'second-molar']
const descriptions = {
  incisor: 'Incisors are the front teeth used to cut food.',
  canine: 'Canines sit beside the incisors and have a pointed cusp.',
  premolar: 'Premolars sit between the canines and molars and help crush food.',
  molar: 'Molars are posterior teeth with broad surfaces for chewing.',
}
const teeth = quadrants.flatMap((quadrant, q) => types.map((type, t) => ({
  id: quadrant+'-'+type, fdi: String(q+1)+String(t+1), label: quadrant.replace('-', ' ')+' '+type.replace('-', ' '),
})))
const structures = [
  {id:'upper-gum',label:'Upper gum'}, {id:'lower-gum',label:'Lower gum'},
  {id:'left-maxilla',label:'Left upper jaw'}, {id:'right-maxilla',label:'Right upper jaw'},
  {id:'mandible',label:'Lower jaw'},
]

export default function Anatomy({examId, initial, jump, openChart}: {
  examId: string; initial: string; jump: (time:number)=>void; openChart: (fdi:string)=>void;
}) {
  const host = useRef<HTMLDivElement>(null)
  const [selected,setSelected] = useState(() => teeth.find(tooth=>tooth.fdi===initial)?.id ?? '')
  const [isolate,setIsolate] = useState(false)
  const [hideTissue,setHideTissue] = useState(false)
  const [status,setStatus] = useState('Loading anatomy…')
  const [ready,setReady] = useState(false)
  const control = useRef<{apply:(id:string, isolate:boolean, hide:boolean)=>void; reset:()=>void; zoom:(amount:number)=>void; rotate:(radians:number)=>void} | undefined>(undefined)
  const tooth = teeth.find(item=>item.id===selected)
  const structure = structures.find(item=>item.id===selected)
  useEffect(()=>{
    const node = host.current
    if (!node) return
    let renderer:WebGLRenderer
    try {renderer=new WebGLRenderer({antialias:true,alpha:false})}
    catch {setStatus('3D needs WebGL. Use the tooth chart to browse the examination.');return}
    renderer.outputColorSpace=SRGBColorSpace
    renderer.setPixelRatio(Math.min(2,window.devicePixelRatio))
    node.append(renderer.domElement)
    const scene = new Scene()
    scene.background = new Color('#12191f')
    scene.add(new AmbientLight(0xffffff,2))
    const key = new DirectionalLight(0xfff1db,2)
    key.position.set(3,5,6);scene.add(key)
    const fill = new DirectionalLight(0xcbdfee,1.5)
    fill.position.set(-4,0,-2);scene.add(fill)
    const camera = new PerspectiveCamera(35,1,.05,100)
    camera.position.set(0,.45,7)
    const orbit = new OrbitControls(camera,renderer.domElement)
    orbit.enablePan=false
    orbit.minDistance=2; orbit.maxDistance=15
    const observer = new ResizeObserver(()=>{
      if (!node.clientWidth || !node.clientHeight) return
      renderer.setSize(node.clientWidth,node.clientHeight)
      camera.aspect=node.clientWidth/node.clientHeight;camera.updateProjectionMatrix()
      renderer.render(scene,camera)
    })
    observer.observe(node)
    const draw = ()=>renderer.render(scene,camera)
    orbit.addEventListener('change',draw)
    const meshes: {mesh:Mesh; material:MeshStandardMaterial; color:Color; id:string; soft:boolean}[] = []
    const focus = new Vector3()
    const home = camera.position.clone()
    let disposed=false
    const release = () => {
      for(const item of meshes) {item.mesh.geometry.dispose();item.material.dispose()}
    }
    new GLTFLoader().loadAsync('/anatomy/teeth.glb').then(asset=>{
      asset.scene.traverse(child=>{
        if (!(child instanceof Mesh) || !(child.material instanceof MeshStandardMaterial)) return
        const id=child.userData.structureId ?? child.name
        meshes.push({mesh:child,material:child.material,color:child.material.color.clone(),id,
          soft:structures.some(item=>item.id===id)})
      })
      if(disposed) {release();return}
      if(meshes.length!==33) throw new Error('Incomplete anatomy model')
      scene.add(asset.scene)
      const bounds=new Box3().setFromObject(asset.scene)
      bounds.getCenter(focus)
      home.copy(focus).add(new Vector3(0,.4,Math.max(5.6,bounds.getSize(new Vector3()).length()*1.5)))
      control.current={
        apply:(id,only,hide)=>{
          for(const item of meshes) {
            const active=item.id===id
            item.mesh.visible=(!only || !id || active) && (!hide || !item.soft || active)
            item.material.color.copy(active?new Color('#8ee0bb'):item.color)
            item.material.emissive.set(active?'#174b3c':'#000000')
            item.material.opacity=(id && !active)? .23 : item.soft && !active ? .53 : 1
            item.material.transparent=item.material.opacity<1
            item.material.depthWrite=!item.material.transparent
          }
          draw()
        },
        reset:()=>{ camera.position.copy(home);orbit.target.copy(focus);orbit.update();draw() },
        rotate:radians=>{
          const offset=camera.position.clone().sub(orbit.target).applyAxisAngle(new Vector3(0,1,0),radians)
          camera.position.copy(orbit.target).add(offset);orbit.update();draw()
        },
        zoom:amount=>{
          const direction=camera.position.clone().sub(orbit.target)
          direction.setLength(Math.min(orbit.maxDistance,Math.max(orbit.minDistance,direction.length()*amount)))
          camera.position.copy(orbit.target).add(direction);orbit.update();draw()
        },
      }
      control.current.reset()
      setReady(true);setStatus('')
    }).catch(()=>{if(!disposed)setStatus('Could not load the anatomy asset. Install it with scripts/install_anatomy.py.')})
    const ray = new Raycaster()
    let origin:Vector2
    const down = (e:PointerEvent)=>{origin=new Vector2(e.clientX,e.clientY)}
    const up = (e:PointerEvent)=>{
      if(!origin || origin.distanceTo(new Vector2(e.clientX,e.clientY))>5) return
      const rect=renderer.domElement.getBoundingClientRect()
      ray.setFromCamera(new Vector2(2*(e.clientX-rect.left)/rect.width-1,1-2*(e.clientY-rect.top)/rect.height),camera)
      const match=ray.intersectObjects(meshes.filter(item=>item.mesh.visible).map(item=>item.mesh),false)
      const hit=match.find(h=>teeth.some(t=>t.id===h.object.userData.structureId)) ?? match[0]
      if(hit) setSelected(hit.object.userData.structureId ?? hit.object.name)
    }
    renderer.domElement.addEventListener('pointerdown',down)
    renderer.domElement.addEventListener('pointerup',up)
    return ()=>{
      disposed=true;control.current=undefined;observer.disconnect();orbit.dispose()
      release();renderer.dispose();renderer.domElement.remove()
    }
  },[])
  useEffect(()=>{if(ready)control.current?.apply(selected,isolate,hideTissue)},[ready,selected,isolate,hideTissue])
  return <section className="panel anatomy-page">
    <div className="section-heading"><h2>3D dental anatomy</h2></div>
    <div className="anatomy-layout">
      <div>
        <div className="anatomy-canvas" ref={host} tabIndex={ready?0:-1} aria-label="3D anatomy. Use the left and right arrows to rotate."
          onKeyDown={e=>{
            if(e.key==='ArrowLeft' || e.key==='ArrowRight') {
              e.preventDefault(); control.current?.rotate(e.key==='ArrowLeft'?-.14:.14)
            }
          }}>{status && <p role="status">{status}</p>}</div>
        <div className="actions">
          <button className="subtle compact-action" disabled={!ready} onClick={()=>control.current?.zoom(.8)}>Zoom in</button>
          <button className="subtle compact-action" disabled={!ready} onClick={()=>control.current?.zoom(1.25)}>Zoom out</button>
          <button className="subtle compact-action" disabled={!ready} onClick={()=>control.current?.reset()}>Reset view</button>
        </div>
      </div>
      <div className="anatomy-controls">
        <label className="field">Select a structure<select value={selected} onChange={e=>setSelected(e.target.value)}>
          <option value="">Whole model</option>
          <optgroup label="Permanent teeth">{teeth.map(item=><option key={item.id} value={item.id}>{item.fdi} · {item.label}</option>)}</optgroup>
          <optgroup label="Gums and jaws">{structures.map(item=><option key={item.id} value={item.id}>{item.label}</option>)}</optgroup>
        </select></label>
        <label className="toggle"><input type="checkbox" disabled={!ready || !selected} checked={isolate && !!selected} onChange={e=>setIsolate(e.target.checked)}/><span>Isolate the selection</span></label>
        <label className="toggle"><input type="checkbox" disabled={!ready} checked={hideTissue} onChange={e=>setHideTissue(e.target.checked)}/><span>Hide gums and jaw bones</span></label>
        {tooth && <>
          <h3>FDI {tooth.fdi} · {tooth.label}</h3>
          <p className="quiet">{descriptions[tooth.label.split(' ').at(-1) as keyof typeof descriptions]}</p>
          <button className="subtle compact-action" onClick={()=>openChart(tooth.fdi)}>Open tooth chart</button>
          <ToothHistory key={tooth.fdi} id={examId} fdi={tooth.fdi} jump={jump}/>
        </>}
        {structure && <h3>{structure.label}</h3>}
        {initial && !teeth.some(item=>item.fdi===initial) && <p className="quiet">Tooth {initial} is outside this model.</p>}
        <p className="quiet">Educational adult anatomy. This model includes 28 teeth.</p>
      </div>
    </div>
    <p className="anatomy-credit">BodyParts3D / Anatomography © DBCLS. Adapted by OrganUI under <a href="https://creativecommons.org/licenses/by-sa/2.1/jp/deed.en">CC BY-SA 2.1 Japan</a>.
      {' '}<a href="/anatomy/ATTRIBUTION.md">Model attribution</a></p>
  </section>
}
