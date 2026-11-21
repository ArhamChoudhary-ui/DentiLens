"""Generate a local Dataset Explorer with de-identified region previews and filters."""
import argparse
import base64
import json
from collections import Counter
from zipfile import ZipFile

import cv2
import numpy as np

from instrument_dataset import Path, checksum, image_from_archive, load_dataset, validate


UI = """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Instrument Dataset Explorer</title>
<style>
body{font:15px system-ui;color:#202923;background:#f5f6f2;margin:2rem}main{max-width:1080px;margin:auto}
section{background:white;border:1px solid #d9dfd8;padding:1rem;margin:1rem 0}h1,h2{margin:0 0 .7rem}
label{display:inline-grid;gap:.25rem;margin:.35rem}select{font:inherit;padding:.4rem}
table{border-collapse:collapse;width:100%}td,th{text-align:left;padding:.35rem;border-bottom:1px solid #ddd}
#cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:1rem}
figure{margin:0}img{display:block;width:100%;height:120px;object-fit:contain;background:#ecefe9}
</style><main><h1>Instrument Dataset Explorer</h1><p id="source"></p>
<section><h2>Split summary</h2><table id="stats"></table><p id="warnings"></p></section>
<section><h2>Annotation previews</h2><p>Frames show geometry only. Crop thumbnails are included for tools with clearly assigned labels.</p>
<div id="filters"></div><p id="count" aria-live="polite"></p><div id="cards"></div></section></main>
<script>
const data=DATA;
const $=id=>document.getElementById(id);
const el=(tag,text)=>{const node=document.createElement(tag);node.textContent=text;return node};
$('source').textContent=data.title+' · '+data.version+' · '+data.license+' · '+data.url;
$('warnings').textContent=data.warning;
for(const row of [['Split','Recordings','Detection frames','Detection annotations','Classification crops'],...data.stats]){
 const tr=$('stats').insertRow();row.forEach(v=>tr.insertCell().textContent=v);
}
const details=document.createElement('details');details.append(el('summary','Class counts by split'));
for (const [kind,splits] of Object.entries(data.distribution)) {
  details.append(el('h3',kind));
  for(const [split,counts] of Object.entries(splits)) details.append(el('p',split+' · '+Object.entries(counts).map(([k,v])=>k+' '+v).join(', ')));
}
$('stats').after(details);
const keys={dataset:'Dataset',label:'Class',recording:'Recording',split:'Split',kind:'Dataset type',
  occlusion:'Occlusion',quality:'Image quality',status:'Annotation status'};
const controls={};
function render(){
 const rows=data.samples.filter(row=>Object.keys(keys).every(key=>!controls[key].value||controls[key].value===row[key]));
 $('count').textContent=rows.length+' representative examples';
 $('cards').replaceChildren(...rows.map(row=>{
  const card=document.createElement('figure');
  const image=document.createElement('img');image.src=row.image;image.alt=row.label+' annotation';
  card.append(image,el('figcaption',row.label+' · '+row.recording+' · frame '+row.frame));
  return card;
 }));
}
for(const [key,title] of Object.entries(keys)){
 const label=el('label',title);const select=document.createElement('select');
 const all=el('option','All');all.value='';select.append(all);
 for(const v of [...new Set(data.samples.map(row=>row[key]))].sort()){const option=el('option',v);option.value=v;select.append(option);}
 select.addEventListener('change',render);label.append(select);$('filters').append(label);controls[key]=select;
}
render();
</script>"""


def preview(image, box, kind, label):
    if kind=='Crop' and label not in {'non_instrument','unknown','other_instrument'}:
        height,width = image.shape[:2]
        x,y,r,b = np.round(np.array(box)*[width,height,width,height]).astype(int)
        pixels = image[y:b,x:r]
        if pixels.size:
            h,w = pixels.shape[:2]
            pixels = cv2.resize(pixels,(max(1,round(w*150/max(h,w))),max(1,round(h*150/max(h,w)))))
        else:
            pixels = np.full((125,180,3),235,np.uint8)
    else:
        pixels = np.full((135,200,3),235,np.uint8)
        x,y,r,b = np.round(np.array(box)*[199,134,199,134]).astype(int)
        cv2.rectangle(pixels,(x,y),(r,b),(70,130,40),2)
    ok,encoded = cv2.imencode('.jpg',pixels)
    if not ok:
        raise ValueError('Could not encode preview.')
    return 'data:image/jpeg;base64,'+base64.b64encode(encoded).decode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','archive','output'):
        parser.add_argument(name,type=Path)
    args = parser.parse_args()
    metadata,frames,crops = load_dataset(args.dataset)
    validate(metadata,frames,crops)
    if checksum(args.archive) != metadata['archive_sha256']:
        raise ValueError('Archive checksum differs from the manifest.')
    parents = {row['id']:row for row in frames}
    examples = [('Frame',frame,annotation) for frame in frames for annotation in frame['annotations']]
    examples.extend(('Crop',parents[crop['parent']],crop) for crop in crops)
    samples = []
    limits = Counter()
    with ZipFile(args.archive) as archive:
        for kind, frame, annotated in examples:
            label = annotated['label']
            key = (kind,frame['recording'],label)
            if limits[key]>=1:
                continue
            limits[key] += 1
            image = None
            if kind=='Crop' and label not in {'non_instrument','unknown','other_instrument'}:
                _, image = image_from_archive(archive,frame['member'])
            samples.append(dict(kind=kind, recording=frame['recording'], split=frame['split'], label=label,
                                frame=frame['frame'], image=preview(image,annotated['box'],kind,label),
                                occlusion='Unspecified',quality='Unrated',status='Source labeled',
                                dataset=metadata['provenance']['title']))
    stats = []
    for split in ('train','valid','test'):
        frame_rows = [f for f in frames if f['split']==split]
        stats.append([split,len({f['recording'] for f in frame_rows}),len(frame_rows),
                      sum(len(f['annotations']) for f in frame_rows),sum(c['split']==split for c in crops)])
    source = metadata['provenance']
    data = dict(stats=stats,samples=samples, title=source['title'],url=source['url'],license=source['license'],
                version=metadata['version'], distribution={'Detection':metadata['detection_counts'],
                                                          'Classification':metadata['classification_counts']},
                warning=f'{len(metadata["exclusions"])} numbered inputs excluded. '+source['limitations'])
    args.output.write_text(UI.replace('=DATA;', '='+json.dumps(data).replace('<','\\u003c')+';'))


if __name__=='__main__':
    main()
