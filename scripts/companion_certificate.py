"""Generate a local TLS certificate with an explicit LAN hostname/IP and QR pin."""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import ipaddress
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", required=True, help="Exact LAN IP or DNS name entered on the phone")
    p.add_argument("--out", type=Path, default=Path("outputs/companion/tls"))
    args = p.parse_args()
    if any(c in args.host for c in "/@?#\r\n") or len(args.host) > 253:
        p.error("Provide only a hostname or IP address")
    args.out.mkdir(parents=True, exist_ok=True)
    private, public = args.out / "key.pem", args.out / "cert.pem"
    if private.exists() or public.exists():
        p.error("TLS files already exist; preserve their fingerprint or choose a new output directory")
    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "ChildSafe LAN relay")])
    try:
        alternative = x509.IPAddress(ipaddress.ip_address(args.host))
    except ValueError:
        alternative = x509.DNSName(args.host.encode("idna").decode())
    now = datetime.now(timezone.utc)
    certificate = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(key.public_key())
                   .serial_number(x509.random_serial_number()).not_valid_before(now-timedelta(minutes=5))
                   .not_valid_after(now+timedelta(days=365)).add_extension(x509.SubjectAlternativeName([alternative]),False)
                   .add_extension(x509.BasicConstraints(ca=True,path_length=0),True).sign(key,hashes.SHA256()))
    with private.open("xb") as out:
        out.write(key.private_bytes(serialization.Encoding.PEM,serialization.PrivateFormat.PKCS8,serialization.NoEncryption()))
    public.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    print("LAN certificate fingerprint (SHA-256):", hashlib.sha256(certificate.public_bytes(serialization.Encoding.DER)).hexdigest())
    print("Private key stays on this machine; pair only using the printed fingerprint.")


if __name__ == "__main__":
    main()
