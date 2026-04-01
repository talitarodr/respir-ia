from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.request, urllib.error, json, os

API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
FRONTEND_URL = os.environ.get("FRONTEND_URL", "*")

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a): pass  # silencia logs verbose

    def do_OPTIONS(self):
        self._cors()
        self.end_headers()

    def do_POST(self):
        if self.path != "/api/chat":
            self.send_response(404); self.end_headers(); return

        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)

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
            with urllib.request.urlopen(req) as r:
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

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", FRONTEND_URL)
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"RespirIA backend rodando na porta {port}")
    HTTPServer(("0.0.0.0", port), Handler).serve_forever()
