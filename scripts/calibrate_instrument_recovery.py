"""Fit confidence calibration on validation crops, then score the locked test."""
import argparse
from collections import Counter
from zipfile import ZipFile

import numpy as np
import torch

from instrument_dataset import Path, ROOT, checksum, image_from_archive, load_dataset, read_json, validate, write_json
from instrument_identity import encoded_crop, network, score_crops

NAMES = ('dental_mirror','dental_explorer')


def fit_temperature(logits, labels, *, split):
    if split != 'valid':
        raise ValueError('Calibration requires the validation split.')
    logits, labels = torch.tensor(logits,dtype=torch.float64), torch.tensor(labels)
    log_temp = torch.zeros((),dtype=torch.float64,requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temp],max_iter=50,line_search_fn='strong_wolfe')
    def closure():
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(logits/log_temp.exp().clamp(.05,20),labels)
        loss.backward()
        return loss
    optimizer.step(closure)
    return float(log_temp.detach().exp().clamp(.05,20))


def evaluate(rows, classes, temperature):
    scores = torch.tensor([r['logits'] for r in rows],dtype=torch.float64)
    probs = (scores/temperature).softmax(1).numpy()
    truth = np.array([classes.index(row['label']) for row in rows])
    order=np.argsort(-probs,axis=1)
    maximum=probs[np.arange(len(rows)),order[:,0]]
    correct=order[:,0]==truth
    bins=np.minimum((maximum*10).astype(int),9)
    reliability=[]
    for i in range(10):
        mask=bins==i
        reliability.append(dict(low=i/10,count=int(mask.sum()),
                                confidence=float(maximum[mask].mean()) if mask.any() else None,
                                accuracy=float(correct[mask].mean()) if mask.any() else None))
    per_class={}
    for name in NAMES:
        index=classes.index(name)
        own = order[:,0]==index
        count=int(own.sum())
        per_class[name]=dict(correct_top_one=int((own & correct).sum()),top_one_predictions=count,
                             target_count=int((truth==index).sum()),
                             max_output=float(maximum[own].max()) if count else None,
                             correct_median=float(np.median(maximum[own & correct])) if (own & correct).any() else None,
                             incorrect_max=float(maximum[own & ~correct].max()) if (own & ~correct).any() else None)
    sweep=[]
    # Fixed grid for diagnostic curves. The .95 research gate remains unchanged.
    for threshold in (.25,.35,.5,.6,.7,.8,.9,.95,.99):
        named=np.isin(order[:,0],[classes.index(name) for name in NAMES])
        accepted=named & (maximum>=threshold)
        hits=int((accepted & correct).sum())
        n=int(accepted.sum())
        negative=truth==classes.index('non_instrument')
        sweep.append(dict(threshold=threshold,accepted=n,coverage=n/len(rows),
                          precision=hits/n if n else None,risk=1-hits/n if n else None,
                          false_acceptance=float((accepted & negative).sum()/negative.sum()) if negative.any() else None,
                          per_class={name:dict(accepted=int((accepted & (order[:,0]==classes.index(name))).sum()),
                                               correct=int((accepted & correct & (order[:,0]==classes.index(name))).sum()))
                                     for name in NAMES}))
    brier=np.square(probs-np.eye(len(classes))[truth]).sum(axis=1).mean()
    nll=-np.log(probs[np.arange(len(rows)),truth]).mean()
    samples=[dict(id=row['id'],group=row['group'],truth=row['label'],logits=row['logits'],
                  predicted=classes[ranking[0]],correct=bool(ok),
                  top_three=[dict(class_id=classes[index],probability=float(probability[index])) for index in ranking[:3]],
                  margin=float(probability[ranking[0]]-probability[ranking[1]]),
                  accepted_at=[item['threshold'] for item in sweep
                               if classes[ranking[0]] in NAMES and probability[ranking[0]]>=item['threshold']])
             for row,ranking,probability,ok in zip(rows,order,probs,correct)]
    return dict(crops=len(rows),nll=float(nll),brier=float(brier),
                ece=sum(item['count']/len(rows)*abs(item['confidence']-item['accuracy'])
                        for item in reliability if item['count']),
                top1=float(correct.mean()),reliability=reliability,classes=per_class,sweep=sweep,samples=samples)


def plot_report(report, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1,3,figsize=(12,3.7),layout='constrained')
    for name, metrics in report['test'].items():
        if name=='groups_per_class':
            continue
        bins=[row for row in metrics['reliability'] if row['count']]
        axes[0].plot([r['confidence'] for r in bins],[r['accuracy'] for r in bins],marker='o',label=name)
        axes[1].step([r['low'] for r in metrics['reliability']],[r['count'] for r in metrics['reliability']],
                     where='post',label=name)
        sweep=[r for r in metrics['sweep'] if r['precision'] is not None]
        axes[2].plot([r['coverage'] for r in sweep],[r['risk'] for r in sweep],marker='o',label=name)
    axes[0].plot([0,1],[0,1],linestyle=':',color='gray')
    for axis,labels in zip(axes,[('Mean confidence','Observed accuracy'),('Confidence bin','Examples'),
                                 ('Named coverage','Selective risk')]):
        axis.set(xlabel=labels[0],ylabel=labels[1])
        axis.legend()
        axis.grid(alpha=.2)
    figure.suptitle('Locked test diagnostics')
    figure.savefig(output/'calibration.png',dpi=160)
    plt.close(figure)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','archive','experiment','output'):
        parser.add_argument(name,type=Path)
    args=parser.parse_args()
    metadata,frames,crops=load_dataset(args.dataset)
    if any(v for v in validate(metadata,frames,crops).values() if isinstance(v,list)):
        raise ValueError('Resolve dataset leakage flags.')
    if checksum(args.archive)!=metadata['archive_sha256']:
        raise ValueError('Archive does not match the manifest.')
    parents={row['id']:row for row in frames}
    training=read_json(args.experiment/'training.json')
    info=training['model']
    checkpoint=args.experiment/'best.pt'
    if checksum(checkpoint)!=training['checkpoint_sha256']:
        raise ValueError('Experimental checkpoint checksum mismatch.')
    model=network(len(info['classes']),info['backbone'])
    model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True))
    model.eval()
    torch.set_num_threads(4)
    args.output.mkdir(parents=True,exist_ok=False)
    with ZipFile(args.archive) as archive:
        def get_rows(split):
            selected=[row for row in crops if row['split']==split and row['label'] in info['classes']]
            rows=[]
            for start in range(0,len(selected),32):
                batch=selected[start:start+32]
                inputs=[encoded_crop(image_from_archive(archive,parents[row['parent']]['member'])[1],
                                     row['box'],info) for row in batch]
                for row,result in zip(batch,score_crops(model,info,inputs,diagnostics=True)):
                    rows.append(dict(id=row['id'],group=parents[row['parent']]['recording'],
                                     label=row['label'],logits=result['logits']))
            return rows
        valid=get_rows('valid')
        temperature=fit_temperature([row['logits'] for row in valid],
                                    [info['classes'].index(row['label']) for row in valid],split='valid')
        test=get_rows('test')
    report={}
    for split,rows in [('valid',valid),('test',test)]:
        report[split]={}
        for name,value in [('raw',1.),('temperature',temperature)]:
            metrics=evaluate(rows,info['classes'],value)
            write_json(args.output/f'{split}-{name}-examples.json',metrics.pop('samples'))
            report[split][name]=metrics
        report[split]['groups_per_class']={key:len({row['group'] for row in rows if row['label']==key})
                                           for key in info['classes']}
    write_json(args.output/'report.json',report)
    plot_report(report,args.output)
    write_json(args.output/'model.json',dict(**info,weights_sha256=training['checkpoint_sha256'],
               temperature=temperature,calibration_version='dentalguard-validation-temperature-v1',
               evaluation_date='2026-11-21',validation_status='Experimental · review only'))
    print(temperature)


if __name__=="__main__":
    main()
