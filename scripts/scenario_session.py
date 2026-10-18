"""Prepare consented adult/doll sessions, with subject-disjoint dev/test metadata."""
import argparse
from childsafe.config import load_config
from childsafe.evaluation.sessions import create_session

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',default='outputs/scenarios'); p.add_argument('--id',required=True)
    p.add_argument('--split',required=True,choices=['dev','test'])
    p.add_argument('--subject',action='append',required=True,help='anonymous stable participant ID')
    p.add_argument('--condition',required=True); p.add_argument('--config')
    p.add_argument('--ckpt',required=True); p.add_argument('--kind',choices=['adult','doll'],default='adult')
    p.add_argument('--consent-confirmed',action='store_true')
    a = p.parse_args()
    print(create_session(a.root,a.id,a.split,a.subject,a.condition,load_config(a.config),a.ckpt,
                         a.kind,a.consent_confirmed))
