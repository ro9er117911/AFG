# HTTPS for phone-browser mic access

Browsers only grant `getUserMedia()` (microphone access) on a **secure context**:
`https://` or literally `http://localhost`. The realistic use case for this tool is opening
it in a **phone's** browser during an actual call, reaching the server over the LAN or
internet — not as `localhost` — so without HTTPS the mic permission prompt silently never
appears (or the call to `getUserMedia()` rejects) and `frontend/app.js`'s `startListening()`
fails before a WebSocket connection is ever made.

There's no way to provision a real, browser-trusted certificate for *your* specific home
network without knowing your domain/network setup in advance, so this document gives you two
concrete, verified paths instead of one turnkey deployment.

## uvicorn already supports TLS natively — no code changes needed

`server/main.py` doesn't need any changes for either path below: `uvicorn` accepts
`--ssl-keyfile`/`--ssl-certfile` directly on the command line, which is all either path uses.

## Path A — self-signed certificate for LAN use (verified, no extra dependency)

Works entirely offline, uses only `openssl` (already on the system — check with
`openssl version`). The phone browser will show a security warning ("connection is not
private") on first visit that you click through once per device; that's expected for a
self-signed cert and is fine for personal use.

```bash
# From the antifraud_v3/ directory (or anywhere — adjust the -out paths):
mkdir -p certs
openssl req -x509 -newkey rsa:2048 -nodes -keyout certs/key.pem -out certs/cert.pem -days 365 \
  -subj "/CN=afg-local" \
  -addext "subjectAltName=DNS:localhost,IP:127.0.0.1,IP:<YOUR-LAN-IP>"
```

Replace `<YOUR-LAN-IP>` with the machine's LAN IP (`ip addr` / `ifconfig`, usually something
like `192.168.1.x`) — the phone will connect to that IP, and the cert's Subject Alternative
Name has to include it or most browsers will still refuse the connection even after you
"accept the risk."

Run the server with TLS:

```bash
cd /home/tommy/Project/AFG
uvicorn antifraud_v3.server.main:app --host 0.0.0.0 \
  --ssl-keyfile antifraud_v3/certs/key.pem --ssl-certfile antifraud_v3/certs/cert.pem
```

`--host 0.0.0.0` is required too — the default `127.0.0.1` only accepts connections from the
same machine, which defeats the purpose of reaching it from a phone on the LAN.

From your phone's browser: `https://<YOUR-LAN-IP>:8000`, accept the certificate warning, then
grant mic permission when prompted.

**Verified in this environment**: generated a cert this way, started uvicorn with
`--ssl-keyfile`/`--ssl-certfile`, and confirmed `curl -k https://127.0.0.1:<port>/` returns
`200` over a real TLS 1.3 handshake (`openssl` reports `subject: CN=afg-local` on the served
cert, matching what was generated). Testing actual mic-permission behavior on a real phone
browser is out of scope for this environment — that last-mile step needs your own device.

If `mkcert` is available on your machine (`which mkcert`) it produces a cert your OS already
trusts (no browser warning at all) instead of a bare self-signed one — not installed in this
dev environment (no `sudo` available to install its local CA), so not verified here, but it's
a drop-in nicer alternative to the two `openssl` commands above if you have it.

## Path B — a tunnel service (real trusted HTTPS, no cert management)

For a certificate your phone trusts with zero warnings, and no need to be on the same LAN
(useful for testing this while genuinely on a call elsewhere), a tunnel service exposes your
locally-running server at a real `https://` URL.

**Verified in this environment** with Cloudflare's `cloudflared` quick tunnel — no Cloudflare
account needed for this mode:

```bash
# Download the static binary (no sudo/install needed, ~40MB):
curl -sL -o cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
chmod +x cloudflared

# With antifraud_v3 already running on port 8000:
./cloudflared tunnel --url http://127.0.0.1:8000
```

This prints a random `https://<random-words>.trycloudflare.com` URL within a few seconds.
Confirmed working end-to-end: `curl https://<the-printed-url>/api/settings` returned a real
`200` with valid JSON, over a certificate that `curl` accepted with no `-k` flag needed (i.e.
a browser wouldn't show any warning either). Open that URL directly on your phone — no LAN
requirement, no IP addresses to find, no cert warning.

Caveats: it's an "account-less" tunnel per Cloudflare's own CLI warning — no uptime guarantee,
the URL changes every time you restart `cloudflared`, and traffic transits Cloudflare's
network (fine for personal testing; if that's a concern for real call audio, prefer Path A on
your own LAN instead, or set up a named/authenticated tunnel with a Cloudflare account, which
this doc doesn't cover). Kill the tunnel process when done — it stops immediately and the URL
stops resolving.

**Not verified in this environment** (not installed, no `sudo` to install them, and some need
an account/login this environment can't complete interactively):

- **`ngrok`** — same idea as `cloudflared`, requires a free account + auth token for the
  tunnel to stay up longer than a few minutes on the free tier.
- **Tailscale Funnel** — if you already use Tailscale for your own devices, `tailscale funnel`
  exposes a local port at a real `https://your-machine.your-tailnet.ts.net` HTTPS URL with a
  certificate Tailscale manages for you. Likely the smoothest option if you're already a
  Tailscale user, since your phone would already be on the tailnet — not verified here since
  it requires an existing Tailscale account/network this environment doesn't have.

## Recommendation

For a quick one-off test: Path B (`cloudflared` quick tunnel) — no LAN/IP juggling, no
browser warning, verified working in ~10 seconds. For a stable setup you'll use repeatedly
from your phone on the same home network: Path A (self-signed cert, or `mkcert` if you
install it) so you're not dependent on an internet tunnel just to talk to a machine on your
own LAN.
