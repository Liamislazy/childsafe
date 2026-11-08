"""Generate the secrets for the phone-alert service (services/notify, docs/phone_alerts.md).

Prints VAPID_PUBLIC_KEY / VAPID_PRIVATE_KEY (Web Push), ALERT_TOKEN (laptop -> service) and
MONITOR_CODE (typed once on each phone). Paste them into Render's environment settings; set
CHILDSAFE_NOTIFY_TOKEN on the laptop to the same ALERT_TOKEN. Nothing is written to disk.
Needs the 'cryptography' package (services/notify/requirements.txt installs it).
"""
import base64
import secrets

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec


def b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b'=').decode()


def main():
    key = ec.generate_private_key(ec.SECP256R1())
    private = key.private_numbers().private_value.to_bytes(32, 'big')
    public = key.public_key().public_bytes(serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint)
    print('VAPID_PUBLIC_KEY=' + b64url(public))
    print('VAPID_PRIVATE_KEY=' + b64url(private))
    print('ALERT_TOKEN=' + secrets.token_urlsafe(32))
    print('MONITOR_CODE=' + secrets.token_hex(3))


if __name__ == '__main__':
    main()
