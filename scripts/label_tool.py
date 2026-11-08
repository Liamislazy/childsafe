"""Local photo capture and box labelling for the hazard object detector (consented adults only).

Open http://127.0.0.1:8003 . Saves still photos and YOLO box labels under data/detector/own,
reviews data/detector/openimages_manifest.csv and own_manifest.csv. Listens on localhost only.
"""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


def main():
    import uvicorn
    from childsafe.web.label_tool import create_label_app
    p = argparse.ArgumentParser()
    p.add_argument('--port', type=int, default=8003)
    p.add_argument('--data', type=Path, default=Path(__file__).resolve().parents[1] / 'data' / 'detector')
    args = p.parse_args()
    args.data.mkdir(parents=True, exist_ok=True)
    print(f'ChildSafe label tool: http://127.0.0.1:{args.port}', flush=True)
    uvicorn.run(create_label_app(args.data), host='127.0.0.1', port=args.port, access_log=False, log_level='warning')


if __name__ == '__main__':
    main()
