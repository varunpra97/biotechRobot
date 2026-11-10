"""Password gate in front of the localhost tool, for sharing it through a tunnel.
Listens on 127.0.0.1:8766, asks the browser for a user name and password (HTTP Basic), and forwards everything to
the tool on 127.0.0.1:8765. The tool itself is not changed or restarted.

    python share_proxy.py <password file>      (user name: lab)
"""
import base64
import hmac
import sys
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

UPSTREAM = "http://127.0.0.1:8765"
WANT = "Basic " + base64.b64encode(("lab:" + Path(sys.argv[1]).read_text().strip()).encode()).decode()
MAX_BODY = 20000


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def go(self):
        if not hmac.compare_digest(self.headers.get("Authorization", ""), WANT):
            self.send_response(401)
            self.send_header("WWW-Authenticate", "Basic realm=\"Lab bench\"")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        n = int(self.headers.get("Content-Length") or 0)
        if n > MAX_BODY:
            self.send_error(413)
            return
        body = self.rfile.read(n) if n else None
        req = urllib.request.Request(UPSTREAM + self.path, data=body, method=self.command)
        for k in ("Content-Type", "Range"):
            if self.headers.get(k):
                req.add_header(k, self.headers[k])
        try:
            r = urllib.request.urlopen(req, timeout=120)
            code, hdrs, data = r.status, r.headers, r.read()
        except urllib.error.HTTPError as e:
            code, hdrs, data = e.code, e.headers, e.read()
        except Exception:
            self.send_error(502, "the tool is not running")
            return
        self.send_response(code)
        for k in ("Content-Type", "Content-Range", "Accept-Ranges", "Cache-Control"):
            if hdrs.get(k):
                self.send_header(k, hdrs[k])
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    do_GET = do_POST = go


ThreadingHTTPServer(("127.0.0.1", 8766), H).serve_forever()
