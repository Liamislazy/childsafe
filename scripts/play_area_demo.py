"""Run the fixed-table prevention prototype; no classifier/landmarks/weight download."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


def main():
    from childsafe.config import load_config
    from childsafe.runtime import configure_runtime
    from childsafe.web.play_area import create_play_area_app
    import uvicorn
    p = argparse.ArgumentParser()
    p.add_argument('--port', type=int, default=8002)
    p.add_argument('--config')
    p.add_argument('--weights', help='Existing local detection/segmentation .pt only')
    args = p.parse_args()
    cfg = load_config(args.config)
    if args.weights:
        if not Path(args.weights).is_file():
            p.error('Local detector weights not found')
        cfg['detector']['weights'] = str(Path(args.weights).resolve())
        cfg['detector']['weights_gpu'] = None
    configure_runtime(cfg)
    print(f'ChildSafe play-area prototype: http://127.0.0.1:{args.port}/play-area', flush=True)
    uvicorn.run(create_play_area_app(cfg), host='127.0.0.1', port=args.port, access_log=False, log_level='warning')


if __name__ == '__main__':
    main()
