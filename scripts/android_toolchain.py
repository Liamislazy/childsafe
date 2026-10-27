"""Download checksum-verified local JDK/Gradle and run Android Gradle tasks.

Does not install/accept Android SDK licenses, change PATH, or use a system JDK.
Large toolchain files stay under ignored outputs/android-tools/.
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "outputs" / "android-tools"
JDK_URL = "https://github.com/adoptium/temurin17-binaries/releases/download/jdk-17.0.20.1%2B1/OpenJDK17U-jdk_x64_windows_hotspot_17.0.20.1_1.zip"
JDK_SHA256 = "e53a79c3c3d86865bd7e787903884331068e71321714ffd44f145785affc7cb0"
GRADLE_URL = "https://services.gradle.org/distributions/gradle-8.11.1-bin.zip"


def download(url: str, target: Path, checksum: str):
    if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == checksum:
        return
    part = target.with_suffix(target.suffix + ".part")
    with urllib.request.urlopen(url, timeout=60) as response, part.open("wb") as out:
        while block := response.read(1024 * 1024):
            out.write(block)
    if hashlib.sha256(part.read_bytes()).hexdigest() != checksum:
        raise RuntimeError("Download checksum mismatch")
    part.replace(target)


def extract(archive: Path):
    with zipfile.ZipFile(archive) as z:
        for entry in z.infolist():
            if not (TOOLS / entry.filename).resolve().is_relative_to(TOOLS.resolve()):
                raise ValueError("Archive path escapes toolchain directory")
        z.extractall(TOOLS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("tasks", nargs="*")
    args, extra = parser.parse_known_args()
    TOOLS.mkdir(parents=True, exist_ok=True)
    jdk = TOOLS / "jdk-17.0.20.1+1"
    gradle = TOOLS / "gradle-8.11.1"
    if not (jdk / "bin/java.exe").is_file():
        print("Downloading verified JDK 17 (~191 MB)", flush=True)
        archive = TOOLS / "jdk17.zip"
        download(JDK_URL, archive, JDK_SHA256)
        extract(archive)
    if not (gradle / "bin/gradle.bat").is_file():
        print("Downloading verified Gradle 8.11.1", flush=True)
        with urllib.request.urlopen(GRADLE_URL + ".sha256", timeout=60) as response:
            checksum = response.read().decode().strip()
        if len(checksum) != 64 or any(c not in "0123456789abcdef" for c in checksum):
            raise RuntimeError("Invalid Gradle checksum response")
        archive = TOOLS / "gradle.zip"
        download(GRADLE_URL, archive, checksum)
        extract(archive)
    if not args.prepare:
        env = dict(os.environ, JAVA_HOME=str(jdk), GRADLE_USER_HOME=str(TOOLS / "gradle-home"))
        command = [str(gradle / "bin/gradle.bat"), "--no-daemon", "--console=plain", *args.tasks, *extra]
        raise SystemExit(subprocess.call(command, cwd=ROOT / "android", env=env))


if __name__ == "__main__":
    main()
