"""Generate the local clearance worksheet and unsent source request drafts."""
import argparse
import re
from pathlib import Path

from instrument_dataset import ROOT, read_json, write_json
from video_benchmark_data import require

ACTIONS=('view','store','extract','annotate','train','evaluate','retain','publish')
KNOWN={
    'local-6502766':('https://www.pexels.com/video/extreme-close-up-of-a-dentist-examining-a-patient-6502766/',
                     'cottonbro studio','Confirm the exact clip and obtain written scoped ML and subject authorization.'),
    'local-4489972':('https://www.pexels.com/video/4489972/',None,
                     'Resolve original listing and contributor from download provenance; the original page was not verified.'),
    'local-getty-1471154103':(None,None,'Keep the preview quarantined; independently verify asset, rights contact and permissible uses.'),
}


def assess(record):
    if not re.fullmatch('[a-zA-Z0-9_-]+',record['id']):
        raise ValueError('Invalid recording ID.')
    url,creator,action=KNOWN.get(record['id'],(None,None,
        'Trace the supplied file to its original listing or authorization record before outreach.'))
    missing={}
    states={}
    for purpose in ACTIONS:
        rights=record.get('permissions',{})
        try:
            require(record,purpose)
            states[purpose]='Verified'
        except ValueError as exc:
            states[purpose]='Denied or restricted' if rights.get(purpose) is False else 'Unverified'
            missing[purpose]=str(exc)
    return dict(id=record['id'],name=record['name'],status=record['status'],content_type=record['category'],
                original_url=record.get('original_url') or url,creator=creator,local_reference=record['reference'],
                identifiable=record.get('identifiable'),consent_documented=bool((record.get('consent') or {}).get('reference')),
                permission_evidence_documented=bool(record.get('rights_reference')),expires=record.get('expires'),
                revocation='Recorded revoked' if record['status']=='revoked' else 'Unverified',
                permissions=states,missing=missing,next_action=action)


def request_draft(row):
    url=row.get('original_url') or row.get('url')
    asset=row.get('name')
    subject=f'Request for scoped dental video permission — {asset}'
    identified=f' The source reference is {url}.' if url else ''
    return f"""# {subject}

Hello,

I am working on DentiLens XR, a student research prototype that studies how
well computer vision can locate and identify dental instruments. I am writing
about {asset}.{identified}

If you manage the recording or can direct me to the authorized contact, could
you confirm which of the following uses you can permit in writing?

- View and retain a local source copy.
- Extract frames and annotate instrument boxes, names and tracks.
- Train and evaluate local instrument-recognition models, with separate approval for each use.
- Retain derived crops and annotations for a defined period.
- Use aggregate results in a student research report or demonstration.
- Publish aggregate metrics and, only if separately approved, redistribute any derived examples.
- Confirm any commercial-use restrictions and required attribution.

The footage may contain identifiable people. Where applicable, please
confirm what subject consent and institutional approval cover the proposed
uses. We would restrict access, avoid unnecessary identifying details, use
pseudonymous recording IDs, and keep raw media out of public repositories.
Please specify any expiry, revocation process, retention limit, privacy
conditions, or uses that must remain excluded.

Thank you,
Arham Choudhary
DentiLens XR student project, Vellore Institute of Technology
"""


def package(registry,output,opportunities):
    records=[assess(read_json(path)) for path in sorted(registry.glob('*.json'))]
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/'worksheet.json',records)
    (output/'worksheet.md').write_text('# Private clearance worksheet\n\n'+'\n'.join(
        f"## {row['id']} · {row['name']}\n\n{row['local_reference']}\n\n"
        f"Source: {row['original_url'] or 'Unverified'} · Creator: {row['creator'] or 'Unverified'}\n\n"
        f"{', '.join(key+' '+value.lower() for key,value in row['permissions'].items())}.\n\n"
        f"Consent, expiry and revocation require review. {row['next_action']}\n"
        for row in records))
    for row in records:
        (output/f"draft-{row['id']}.md").write_text(request_draft(row))
    for index,row in enumerate(opportunities):
        (output/f'draft-opportunity-{index+1}.md').write_text(request_draft(row))
    (output/'request-template.md').write_text(request_draft({'name':'[recording or proposed capture]'}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('registry',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    package(args.registry,args.output,read_json(ROOT/'datasets/video_sources/clearance/opportunities.json'))
