"""Register, annotate and evaluate locally authorized DentiLens videos."""
import argparse
import json
from collections import defaultdict
from pathlib import Path
import resource
import sys
from time import perf_counter

from instrument_dataset import ROOT, read_json, write_json
from video_benchmark_data import (check_isolation, digest, extract, iter_frames, register,
                                 reviewed_annotations, source_path)
from video_benchmark_metrics import annotation_disagreements, matched_objects, stratified_detection, track_metrics


def evaluate(record, annotation, output, sizes, shadow_path, methods):
    import torch
    from ultralytics import YOLO
    from detector_tiles import predict_windows
    from evaluate_classifier import classification_metrics, COMMON
    from instrument_identity import (classify_pipeline, context_model, describe_tracks,
                                     crop_geometry, identity_model, identity_pipeline, predict_video)

    path=source_path(record,'evaluate','extract','annotate','retain')
    video=record['video']
    if (sizes!=[512] or methods!=['full']) and record['split']!='valid':
        raise ValueError('Resolution comparisons require the validation split.')
    if record['split'] not in {'valid','test'}:
        raise ValueError('Benchmark scoring needs a validation or test recording.')
    torch.set_num_threads(4)
    metadata=read_json(ROOT/'reference/instrument-video-v3-results.json')
    weights=ROOT/'models/video-instruments.pt'
    if digest(weights)!=metadata['checkpoint_sha256']:
        raise ValueError('Detector checkpoint hash mismatch.')
    cutoff=metadata['comparison']['pilot']['score_threshold']
    detector=YOLO(str(weights),task='detect')
    if len(detector.names)!=1:
        raise ValueError('Unexpected detector class mapping.')
    models=identity_pipeline(ROOT/'models')
    context=context_model(ROOT/'models')
    shadow=identity_model(shadow_path,'model') if shadow_path else None
    if shadow and not shadow[1].get('validation_status','').startswith('Experimental'):
        raise ValueError('Shadow checkpoint is not marked experimental.')
    output.mkdir(parents=True,exist_ok=False)
    truth=defaultdict(list)
    for row in annotation['objects']:
        if row['box'] is not None:
            truth[row['frame']].append(row)
    reports={}
    geometries={}
    for size,method in [(size,method) for size in sizes for method in methods]:
        key=str(size) if method=='full' else f'{size}-{method}'
        observations=[]
        samples=[]
        start=perf_counter()
        with (output/f'{key}-detector.jsonl').open('w') as ledger:
            for index,image in iter_frames(path,video):
                predictions,raw=predict_windows(detector,image,method,size)
                time=index/video['fps']
                ledger.write(json.dumps(dict(frame=index,time=time,ranked=predictions,pre_merge=raw))+'\n')
                samples.append((truth[index],predictions,[]))
                for number,(score,box) in enumerate(predictions):
                    if score>=cutoff:
                        observations.append(dict(detection_id=f'instrument-video-{index}-{number}',frame=index,
                            time=time,confidence=score,box=box,crop_geometry=crop_geometry(box,video['width'],video['height'])))
        detection_seconds=perf_counter()-start
        grouped=defaultdict(list)
        for row in observations:
            grouped[row['frame']].append(row)
        geometries[key]=grouped
        metrics=stratified_detection(samples,cutoff,video['width'],video['height'])
        metrics['unmatched_proposals_per_minute']=metrics['fp']*60/video['duration']
        start=perf_counter()
        production=classify_pipeline(path,observations,models,context)
        production_seconds=perf_counter()-start
        heads={'released':(production,production_seconds)}
        if shadow:
            start=perf_counter()
            experimental,=predict_video(path,observations,[shadow])
            heads['shadow']=(experimental,perf_counter()-start)
            disagreement=0
            for identifier,current in production.items():
                alternate=experimental[identifier]
                disagreement+=int((current or {}).get('class_id')!=(alternate or {}).get('class_id'))
        else:
            disagreement=None
        run={}
        for name,(predictions,seconds) in heads.items():
            rows=[dict(row,identity=predictions[row['detection_id']]) for row in observations]
            tracks,_=describe_tracks(rows,video,[])
            run[name]=dict(seconds=seconds, tracking=track_metrics(annotation,rows,video))
            write_json(output/f'{key}-{name}-predictions.json',dict(observations=rows,tracks=tracks))
        reports[key]=dict(detector=metrics, detector_seconds=detection_seconds, comparison=run, shadow_top_class_disagreements=disagreement)
    verified=[dict(row,detection_id=f"{row['track_id']}:{row['frame']}")
              for rows in truth.values() for row in rows
              if row['label'] in COMMON[:2] and row['visibility']!='ambiguous']
    classification={}
    if verified:
        heads=[*models,*([shadow] if shadow else [])]
        results=predict_video(path,verified,heads)
        for (_,meta),predictions in zip(heads,results):
            ordered=[(row,predictions[row['detection_id']]) for row in verified
                     if predictions[row['detection_id']] is not None]
            if ordered:
                classification[meta['model']]=classification_metrics([row['label'] for row,_ in ordered],
                                                                    [value for _,value in ordered])
    differences={}
    if '512' in geometries:
        baseline=geometries['512']
        for size,current in geometries.items():
            if size=='512':
                continue
            common=sum(len(matched_objects(baseline[frame],current[frame])) for frame in annotation['frames'])
            differences[str(size)]=dict(matched_against_512=common,
                                        only_512=sum(map(len,baseline.values()))-common,
                                        only_alternate=sum(map(len,current.values()))-common)
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_bytes=peak if sys.platform=='darwin' else peak*1024
    write_json(output/'report.json',dict(recording=record['id'],source_sha256=record['sha256'],
        detector_sha256=metadata['checkpoint_sha256'],video=video,annotation_version=annotation['version'],
        research_heads={info['model']:info['weights_sha256'] for _,info in [*models,*([shadow] if shadow else [])]},
        reviewed_samples=len(annotation['frames']),ground_truth_tracks=len(annotation['events']),
        runs=reports,detector_geometry_comparison=differences,
        process_peak_rss_bytes=peak_bytes,manual_crop_classification=classification))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    enroll=commands.add_parser('register')
    enroll.add_argument('source',type=Path)
    enroll.add_argument('output',type=Path)
    permissions=commands.add_parser('check')
    permissions.add_argument('registry',type=Path)
    permissions.add_argument('purpose',choices=('annotate','train','evaluate','publish'))
    frames=commands.add_parser('extract')
    frames.add_argument('source',type=Path)
    frames.add_argument('output',type=Path)
    for name in ('validate','evaluate','compare-annotations'):
        command=commands.add_parser(name)
        for field in ('source','xml','review','output'):
            command.add_argument(field,type=Path)
        if name=='compare-annotations':
            command.add_argument('--second-xml',type=Path,required=True)
            command.add_argument('--second-review',type=Path,required=True)
        if name=='evaluate':
            command.add_argument('--sizes',type=int,nargs='+',choices=(512,640,768),default=[512])
            command.add_argument('--shadow',type=Path)
            command.add_argument('--methods',nargs='+',choices=('full','tiles','roi'),default=['full'])
            command.add_argument('--registry',type=Path,required=True)
    args=parser.parse_args()
    if args.command=='register':
        if args.output.exists():
            parser.error('Use a new registry record path.')
        write_json(args.output,register(read_json(args.source)))
    elif args.command=='check':
        registry=[read_json(p) for p in sorted(args.registry.glob('*.json'))]
        check_isolation(registry)
        eligible=[]
        excluded={}
        for row in registry:
            try:
                actions=('extract','annotate','retain',args.purpose) if args.purpose in {'train','evaluate'} else ('extract','retain',args.purpose)
                source_path(row,*actions)
                if args.purpose=='train' and row['split']!='train':
                    raise ValueError('Only the training split can enter a training export.')
                eligible.append(row['id'])
            except (OSError,ValueError) as exc:
                excluded[row['id']]=str(exc)
        print(f'Eligible: {eligible} | Excluded: {excluded}')
        if not eligible:
            raise SystemExit(1)
    elif args.command=='extract':
        extract(read_json(args.source),args.output)
    else:
        record=read_json(args.source)
        annotation=reviewed_annotations(record,args.xml,read_json(args.review))
        if args.command=='validate':
            if args.output.exists():
                parser.error('Use a new annotation output path.')
            write_json(args.output,annotation)
        elif args.command=='compare-annotations':
            second=reviewed_annotations(record,args.second_xml,read_json(args.second_review))
            write_json(args.output,annotation_disagreements(annotation,second))
        else:
            registry=[read_json(p) for p in sorted(args.registry.glob('*.json'))]
            check_isolation(registry)
            if record not in registry:
                raise ValueError('Benchmark source does not belong to the checked registry.')
            evaluate(record,annotation,args.output,args.sizes,args.shadow,args.methods)


if __name__=='__main__':
    main()
