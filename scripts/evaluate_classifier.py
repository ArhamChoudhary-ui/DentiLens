"""Compare released identity checkpoints on identical source-labelled test crops."""
import argparse
from collections import Counter, defaultdict
from zipfile import ZipFile

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import torch

from instrument_dataset import ROOT, Path, checksum, image_from_archive, load_dataset, validate, write_json
from instrument_identity import encoded_crop, identity_model, score_crops

# The released specialist cannot distinguish other tools from negatives.
# Keep source labels in the manifest, while scoring the comparable common task.
COMMON = ['dental_mirror','dental_explorer','unknown']


def common_label(label):
    return label if label in COMMON[:2] else 'unknown'


def classification_metrics(truth, predictions):
    actual = [common_label(label) for label in truth]
    predicted = [common_label(row['class_id']) for row in predictions]
    scores = np.array([row['confidence'] for row in predictions])
    correct = np.equal(actual,predicted)
    ranked = sorted(range(len(truth)),key=lambda key:-scores[key])
    bins = np.minimum((scores*10).astype(int),9)
    ece = sum(mask.mean()*abs(correct[mask].mean()-scores[mask].mean())
              for value in range(10) if (mask:=bins==value).any())
    accepted = [i for i,row in enumerate(predictions) if row['accepted']]
    negatives = [i for i,value in enumerate(truth) if value=='non_instrument']
    report = classification_report(actual,predicted,labels=COMMON,target_names=COMMON,output_dict=True,zero_division=0)
    top_three = [list(dict.fromkeys(common_label(item['class_id']) for item in row['alternatives']))
                 for row in predictions]
    return dict(crops=len(truth),truth_counts=dict(Counter(truth)),raw_accuracy=float(correct.mean()),
                top3_accuracy=float(np.mean([key in top for key,top in zip(actual,top_three)])),
                macro_f1=report['macro avg']['f1-score'],per_class={key:report[key] for key in COMMON},
                class_order=COMMON,confusion_matrix=confusion_matrix(actual,predicted,labels=COMMON).tolist(),
                raw_expected_calibration_error=float(ece),confidence_p10_p50_p90=np.percentile(scores,[10,50,90]).tolist(),
                unknown_top1_rate=predicted.count('unknown')/len(truth),
                rejected=1-len(accepted)/len(truth),
                accepted_accuracy=sum(predictions[i]['class_id']==truth[i] for i in accepted)/len(accepted) if accepted else None,
                negative_false_acceptance=sum(predictions[i]['accepted'] for i in negatives)/len(negatives) if negatives else None,
                coverage_curve=[dict(coverage=n/len(truth),accuracy=float(correct[ranked[:n]].mean()))
                                for part in (.1,.25,.5,.75,1.) if (n:=max(1,round(len(truth)*part)))])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','archive','output'):
        parser.add_argument(name,type=Path)
    args = parser.parse_args()
    metadata,frames,crops = load_dataset(args.dataset)
    issues = validate(metadata,frames,crops)
    if any(value for value in issues.values() if isinstance(value,list)):
        raise ValueError('Resolve split leakage flags before benchmarking.')
    if checksum(args.archive) != metadata['archive_sha256']:
        raise ValueError('Archive checksum differs from the manifest.')
    per_frame = defaultdict(list)
    for crop in crops:
        if crop['split']=='test' and crop['label'] in {*COMMON,'non_instrument'}:
            per_frame[crop['parent']].append(crop)
    torch.set_num_threads(4)
    models = [identity_model(ROOT/'models',stem) for stem in ('instrument-identity','instrument-detail')]
    truth,pending = [],[]
    predictions = [[] for _ in models]

    def flush():
        if pending:
            for index,(model,info) in enumerate(models):
                predictions[index].extend(score_crops(model,info,[row[index] for row in pending]))
            pending.clear()

    with ZipFile(args.archive) as archive:
        for frame in frames:
            if frame['id'] not in per_frame:
                continue
            _,image = image_from_archive(archive,frame['member'])
            for row in per_frame[frame['id']]:
                pending.append([encoded_crop(image,row['box'],info) for _,info in models])
                truth.append(row['label'])
                if len(pending)==32:
                    flush()
    flush()
    result = dict(dataset=metadata['version'],archive_sha256=metadata['archive_sha256'],heads={
        info['model']:dict(checkpoint_sha256=info['weights_sha256'], **classification_metrics(truth, rows))
        for (_,info),rows in zip(models,predictions)
    })
    write_json(args.output,result)
    print(result)


if __name__=='__main__':
    main()
