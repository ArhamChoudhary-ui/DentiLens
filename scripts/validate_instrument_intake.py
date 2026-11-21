"""Validate independently reviewed future instrument intake records."""
import argparse
from collections import defaultdict
import re

from instrument_dataset import LABELS, Path, read_json, valid_box


def validate_intake(rows, locked_groups=()):
    groups = defaultdict(set)
    seen = set()
    fingerprints = []
    eligible = []
    for row in rows:
        for key in ('id','source','recording','filename','parent_frame','reviewer','dataset_version','permission_evidence'):
            if not row.get(key):
                raise ValueError(f'Missing intake field: {key}')
        if row['label'] not in LABELS or not valid_box(row['box']) or min(row['width'],row['height'])<1:
            raise ValueError(f'Invalid crop: {row["id"]}')
        if row['split'] not in {'train','valid','test'} or row['status'] not in {'verified','quarantine'}:
            raise ValueError('Invalid split or review status.')
        groups[row['recording']].add(row['split'])
        if row['id'] in seen:
            raise ValueError('Duplicate intake ID.')
        seen.add(row['id'])
        if row['status']=='quarantine':
            continue
        rights=row['permissions']
        if any(rights.get(key) is not True for key in ('extract','annotate','train','evaluate')):
            raise ValueError(f'Incomplete permissions: {row["id"]}')
        if row.get('identifiable') is not False and rights.get('identifiable_training') is not True:
            raise ValueError(f'Missing consent for potentially identifiable media: {row["id"]}')
        if row['recording'] in locked_groups:
            raise ValueError('A locked benchmark recording cannot be added to intake.')
        digest=row['sha256']
        signature=row['dhash']
        if not re.fullmatch('[0-9a-f]{64}',digest) or not re.fullmatch('[0-9a-f]{16}',signature):
            raise ValueError('Invalid intake fingerprints.')
        number=int(signature,16)
        for prior_digest, prior, split in fingerprints:
            if digest==prior_digest or (row['split']!=split and (number^prior).bit_count()<=4):
                raise ValueError('Duplicate or cross-split similar crop requires review.')
        fingerprints.append((digest,number,row['split']))
        eligible.append(row['id'])
    if any(len(splits)>1 for splits in groups.values()):
        raise ValueError('A source recording crosses splits.')
    return eligible


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('intake',type=Path)
    args=parser.parse_args()
    from instrument_dataset import ROOT
    locked=read_json(ROOT/'datasets/manifests/dentalguard-splits.json')
    print(validate_intake(read_json(args.intake),{key for key,value in locked.items() if value=='test'}))
