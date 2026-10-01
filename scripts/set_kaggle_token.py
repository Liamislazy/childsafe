"""Save a Kaggle API token (KGAT_...) to %USERPROFILE%\\.kaggle\\access_token, then test it.
You paste the token yourself when asked; it is not echoed and not stored anywhere else.

    python scripts/set_kaggle_token.py
"""
import getpass
import subprocess
import sys
from pathlib import Path


def main():
    token = getpass.getpass("Paste your Kaggle token (hidden) and press Enter: ").strip()
    if not token:
        raise SystemExit("no token entered")
    path = Path.home() / ".kaggle" / "access_token"
    path.parent.mkdir(exist_ok=True)
    path.write_text(token, encoding="ascii")  # no BOM, no newline
    print(f"saved to {path} ({len(token)} chars). Testing...")
    r = subprocess.run([sys.executable, "-m", "kaggle", "datasets", "files", "fairuzkhan002/datasetvgg19"],
                       capture_output=True, text=True)
    print("OK, Kaggle accepted the token." if r.returncode == 0 else f"Kaggle rejected it:\n{r.stderr or r.stdout}")


if __name__ == "__main__":
    main()
