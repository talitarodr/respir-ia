"""
RespirIA Backend com RAG leve
==============================
Usa TF-IDF simples para busca nos PDFs — sem GPU, sem modelos pesados.
Funciona no plano gratuito do Render (512MB RAM).

requirements.txt:
    chromadb==0.4.22
    pypdf==4.1.0
"""

from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.request, urllib.error, json, os, re, math
from collections import defaultdict

API_KEY    = os.environ.get("ANTHROPIC_API_KEY", "")
SHEETS_URL = os.environ.get("GOOGLE_SHEETS_URL", "")
PORT       = int(os.environ.get("PORT", 8000))

# ── Base de conhecimento (carregada uma vez na inicialização) ──────
_chunks  = []   # lista de {"texto": ..., "fonte": ...}
_idf     = {}   # IDF por termo
_tf_vecs = []   # TF por chunk
_rag_ok  = False

def init_rag():
    global _chunks, _idf, _tf_vecs, _rag_ok
    try:
        from pypdf import PdfReader
        pdf_dir = os.path.dirname(os.path.abspath(__file__))
        pdfs = [f for f in os.listdir(pdf_dir) if f.endswith(".pdf")]
        if not pdfs:
            print("RAG: nenhum PDF encontrado", flush=True)
            _rag_ok = True
            return

        for fname in pdfs:
            try:
                reader = PdfReader(os.path.join(pdf_dir, fname))
                texto = " ".join(p.extract_text() or "" for p in reader.pages)
                for chunk in _chunkar(texto):
                    if len(chunk.strip()) > 80:
                        _chunks.append({"texto": chunk.strip(), "fonte": fname})
                print(f"RAG: {fname} indexado", flush=True)
            except Exception as e:
                print(f"RAG: erro em {fname}: {e}", flush=True)

        # Calcula TF-IDF
        N = len(_chunks)
        df = defaultdict(int)
        tfs = []
        for c in _chunks:
            termos = _tokenizar(c["texto"])
            tf = defaultdict(float)
            for t in termos:
                tf[t] += 1
            total = len(termos) or 1
            for t in tf:
                tf[t] /= total
                df[t] += 1
            tfs.append(dict(tf))

        for t, n in df.items():
            _idf[t] = math.log((N + 1) / (n + 1)) + 1

        for tf in tfs:
            vec = {t: v * _idf.get(t, 1) for t, v in tf.items()}
            _tf_vecs.append(vec)

        print(f"RAG: {len(_chunks)} chunks prontos", flush=True)
        _rag_ok = True

    except Exception as e:
        print(f"RAG: erro ({e})", flush=True)
        _rag_ok = True


def _chunkar(texto, tam=500, overlap=60):
    palavras = texto.split()
    chunks, i = [], 0
    while i < len(palavras):
        chunks.append(" ".join(palavras[i:i+tam]))
        i += tam - overlap
    return chunks


def _tokenizar(texto):
    return re.findall(r'\b[a-záéíóúãõâêîôûàç]{3,}\b', texto.lower())


def _similaridade(vec_q, vec_d):
    comum = set(vec_q) & set(vec_d)
    if not comum:
        return 0.0
    dot = sum(vec_q[t] * vec_d[t] for t in comum)
    norm_q = math.sqrt(sum(v**2 for v in vec_q.values())) or 1
    norm_d = math.sqrt(sum(v**2 for v in vec_d.values())) or 1
    return dot / (norm_q * norm_d)


def buscar_contexto(pergunta, n=4):
    if not _chunks:
        return ""
    termos_q = _tokenizar(pergunta)
    if not termos_q:
        return ""
    vec_q = {}
    for t in termos_q:
        vec_q[t] = vec_q.get(t, 0) + 1
    total = len(termos_q)
    vec_q = {t: (v/total) * _idf.get(t, 1) for t, v in vec_q.items()}

    scores = [(_similaridade(vec_q, v), i) for i, v in enumerate(_tf_vecs)]
    scores.sort(reverse=True)

    trechos = []
    for score, idx in scores[:n]:
        if score < 0.01:
            break
        fonte = re.sub(r'_|\.pdf', ' ', _chunks[idx]["fonte"]).strip()
        trechos.append(f"[{fonte}]\n{_chunks[idx]['texto']}")
    return "\n\n---\n\n".join(trechos)


def injetar_contexto(payload, contexto):
    if not contexto:
        return payload
    payload = dict(payload)
    payload["system"] = (
        payload.get("system", "")
        + "\n\n════ CONTEXTO DAS REFERÊNCIAS ════\n"
        + "Use os trechos abaixo como base para sua resposta. "
        + "Cite a fonte entre parênteses quando pertinente.\n\n"
        + contexto
        + "\n════ FIM DO CONTEXTO ════"
    )
    return payload


# ── Handler ────────────────────────────────────────────────────────
class Handler(BaseHTTPRequestHandler):

    def log_message(self, *a): pass

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")

    def do_OPTIONS(self):
        self.send_response(200); self._cors(); self.end_headers()

    def do_GET(self):
        self.send_response(200); self._cors()
        self.send_header("Content-Type", "text/plain"); self.end_headers()
        self.wfile.write(f"RespirIA ativo. RAG: {'pronto' if _rag_ok else 'iniciando'} ({len(_chunks)} chunks)".encode())

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body   = self.rfile.read(length)

        if self.path == "/api/log":
            try:
                self._save_to_sheets(json.loads(body))
                self.send_response(200); self._cors()
                self.send_header("Content-Type", "application/json"); self.end_headers()
                self.wfile.write(b'{"ok":true}')
            except Exception as e:
                self.send_response(500); self._cors(); self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode())
            return

        if self.path != "/api/chat":
            self.send_response(404); self._cors(); self.end_headers()
            return

        try:
            payload = json.loads(body)
            msgs = payload.get("messages", [])
            ultima = next((m["content"] for m in reversed(msgs) if m.get("role") == "user"), "")
            if ultima and _rag_ok:
                ctx = buscar_contexto(ultima)
                if ctx:
                    payload = injetar_contexto(payload, ctx)
            body = json.dumps(payload).encode()
        except Exception:
            pass

        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages", data=body,
            headers={"Content-Type": "application/json",
                     "x-api-key": API_KEY,
                     "anthropic-version": "2023-06-01"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
            self.send_response(200); self._cors()
            self.send_header("Content-Type", "application/json"); self.end_headers()
            self.wfile.write(data)
        except urllib.error.HTTPError as e:
            err = e.read()
            self.send_response(e.code); self._cors()
            self.send_header("Content-Type", "application/json"); self.end_headers()
            self.wfile.write(err)
        except Exception as e:
            self.send_response(500); self._cors()
            self.send_header("Content-Type", "application/json"); self.end_headers()
            self.wfile.write(json.dumps({"error": {"message": str(e)}}).encode())

    def _save_to_sheets(self, payload):
        if not SHEETS_URL: return
        try:
            req = urllib.request.Request(SHEETS_URL,
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=10) as r:
                print(f"Sheets OK: {r.status}", flush=True)
        except Exception as e:
            print(f"Sheets ERRO: {e}", flush=True)


if __name__ == "__main__":
    print(f"RespirIA backend na porta {PORT}", flush=True)
    import threading
    threading.Thread(target=init_rag, daemon=True).start()
    HTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
