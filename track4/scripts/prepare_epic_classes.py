"""Extract ordered class keys and observed pairs from official training annotations."""
import argparse
from collections import Counter
import csv
from hashlib import file_digest
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from egosight.contract import StudentKit


def read(path):
    with path.open(newline='',encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def build(verbs,nouns,train,kit):
    def keys(path,size):
        rows=sorted(read(path),key=lambda item:int(item['id']))
        if [int(row['id']) for row in rows]!=list(range(size)):
            raise ValueError('Unexpected EPIC class ID ordering.')
        return [row['key'] for row in rows]
    labels=dict(verb=keys(verbs,97),noun=keys(nouns,300),label=['cooking'])
    kit.validate_classes(labels)
    rows=read(train)
    if set(kit.entries).intersection(row['video_id'] for row in rows):
        raise ValueError('Training pairs include an evaluation video ID.')
    pairs=Counter((int(row['verb_class']),int(row['noun_class'])) for row in rows)
    if not pairs or any(not (0<=v<97 and 0<=n<300) for v,n in pairs):
        raise ValueError('Invalid training action pairs.')
    hashes={}
    for path in (verbs,nouns,train):
        with path.open('rb') as stream:
            hashes[path.name]=file_digest(stream,'sha256').hexdigest()
    return dict(classes=labels,pairs=[list(pair) for pair in sorted(pairs)],sources_sha256=hashes)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for field in ('verbs','nouns','train','output'):
        parser.add_argument(field,type=Path)
    args=parser.parse_args()
    args.output.write_text(json.dumps(build(args.verbs,args.nouns,args.train,StudentKit()),indent=2)+'\n')
