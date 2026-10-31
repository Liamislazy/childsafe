"""Run the event-only relay. Non-loopback listeners require explicit TLS files."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from childsafe.companion.relay import RelayStore, create_relay


def main():
    import uvicorn
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8443)
    p.add_argument("--db", type=Path, default=Path("outputs/companion/relay.sqlite3"))
    p.add_argument("--cert", type=Path)
    p.add_argument("--key", type=Path)
    args = p.parse_args()
    if bool(args.cert) != bool(args.key) or (args.host not in {"127.0.0.1", "::1", "localhost"} and not args.cert):
        p.error("Network listeners require both --cert and --key; HTTP is loopback-only")
    if any(path and not path.is_file() for path in [args.cert, args.key]):
        p.error("TLS file missing")
    args.db.parent.mkdir(parents=True, exist_ok=True)
    uvicorn.run(create_relay(RelayStore(args.db)), host=args.host, port=args.port,
                ssl_certfile=str(args.cert) if args.cert else None,
                ssl_keyfile=str(args.key) if args.key else None,
                access_log=False, log_level="warning", proxy_headers=False)


if __name__ == "__main__":
    main()
