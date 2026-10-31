"""Install only the SDK packages used to build this app, after explicit consent.

Requires --accept-android-sdk-license. Does not accept every repository license,
install emulators, change system settings, or add firewall rules.
"""
import argparse
import os
from pathlib import Path
import subprocess
import zipfile

from android_toolchain import download, TOOLS, ROOT

URL = "https://dl.google.com/android/repository/commandlinetools-win-15859902_latest.zip"
SHA256 = "90ae805d20434428bffcb699c290860f19bb5f66a67e6b330067e3de801fb04a"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--accept-android-sdk-license", action="store_true")
    p.add_argument("--java-home", type=Path, required=True)
    args = p.parse_args()
    if not args.accept_android_sdk_license:
        p.error("Explicit user agreement to Google's Android SDK license is required before downloading")
    if not (args.java_home / "bin/java.exe").is_file():
        p.error("Java executable missing")
    TOOLS.mkdir(parents=True, exist_ok=True)
    sdk = TOOLS / "sdk"
    tools = sdk / "cmdline-tools/latest"
    archive = TOOLS / "android-commandline.zip"
    download(URL, archive, SHA256)
    with zipfile.ZipFile(archive) as z:
        for entry in z.infolist():
            if not entry.filename.startswith("cmdline-tools/"):
                raise ValueError("Unexpected SDK archive structure")
            relative = entry.filename.removeprefix("cmdline-tools/")
            destination = tools / relative
            if not destination.resolve().is_relative_to(tools.resolve()):
                raise ValueError("SDK archive path escapes install directory")
            if entry.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
            elif relative:
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(z.read(entry))
    environment = dict(os.environ, JAVA_HOME=str(args.java_home))
    manager = tools / "bin/sdkmanager.bat"
    # Only these packages share the Android SDK license the user approved.
    # A single response cannot silently accept other, unrelated licenses.
    result = subprocess.run([str(manager), f"--sdk_root={sdk}", "--install", "platform-tools",
                             "platforms;android-35", "build-tools;35.0.0"],
                            input="y\n", text=True, env=environment)
    if result.returncode:
        raise SystemExit(result.returncode)
    (ROOT / "android/local.properties").write_text("sdk.dir=" + sdk.as_posix() + "\n", encoding="utf-8")
    print("Android SDK configured locally; no emulator installed.")


if __name__ == "__main__":
    main()
