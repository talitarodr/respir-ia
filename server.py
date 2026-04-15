from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.request, urllib.error, json, os

API_KEY      = os.environ.get("ANTHROPIC_API_KEY", "")
SHEETS_URL   = os.environ.get("GOOGLE_SHEETS_URL", "")
PORT         = int(os.environ.get("PORT", 8000))

class Handler(BaseHTTPRequestHandler):

    def log_message(self, *a): pass

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")

    def do_OPTIONS(self):
        self.send_response(200)
        self._cors()
        self.end_headers()

    def do_GET(self):
        self.send_response(200)
        self._cors()
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(b"RespirIA backend ativo!")

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body   = self.rfile.read(length)

        if self.path == "/api/log":
            try:
                payload = json.loads(body)
                self._save_to_sheets(payload)
                self.send_response(200)
                self._cors()
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"ok":true}')
            except Exception as e:
                self.send_response(500)
                self._cors()
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode())
            return

        if self.path != "/api/chat":
            self.send_response(404)
            self._cors()
            self.end_headers()
            return

        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=body,
            headers={
                "Content-Type": "application/json",
                "x-api-key": API_KEY,
                "anthropic-version": "2023-06-01",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
            self.send_response(200)
            self._cors()
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(data)

        except urllib.error.HTTPError as e:
            err = e.read()
            self.send_response(e.code)
            self._cors()
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(err)

        except Exception as e:
            msg = json.dumps({"error": {"message": str(e)}}).encode()
            self.send_response(500)
            self._cors()
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(msg)

    def _save_to_sheets(self, payload):
        if not SHEETS_URL:
            print("AVISO: GOOGLE_SHEETS_URL nao configurada", flush=True)
            return
        data = json.dumps(payload).encode()
        req  = urllib.request.Request(
            SHEETS_URL,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                print(f"Sheets OK: {r.status}", flush=True)
        except Exception as e:
            print(f"Sheets ERRO: {e}", flush=True)

if __name__ == "__main__":
    print(f"RespirIA backend rodando na porta {PORT}", flush=True)
    HTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
