"""
RespirIA Backend com RAG
========================
Dependências (requirements.txt):
    chromadb==0.4.22
    pypdf==4.1.0
    sentence-transformers==2.7.0

Variáveis de ambiente no Render:
    ANTHROPIC_API_KEY   — chave da API Anthropic
    GOOGLE_SHEETS_URL   — URL do Apps Script
    PORT                — definido automaticamente pelo Render
"""

from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.request, urllib.error, json, os, re

API_KEY     = os.environ.get("ANTHROPIC_API_KEY", "")
SHEETS_URL  = os.environ.get("GOOGLE_SHEETS_URL", "")
PORT        = int(os.environ.get("PORT", 8000))

# ── RAG: inicializa base vetorial na primeira requisição ───────────
_rag_ready = False
_collection = None

def init_rag():
    global _rag_ready, _collection
    if _rag_ready:
        return
    try:
        import chromadb
        from sentence_transformers import SentenceTransformer
        from pypdf import PdfReader

        print("RAG: inicializando base vetorial...", flush=True)

        # Modelo leve de embeddings (roda na CPU sem GPU)
        model = SentenceTransformer("all-MiniLM-L6-v2")

        client = chromadb.Client()
        _collection = client.get_or_create_collection("respiria_refs")

        # PDFs na mesma pasta do server.py
        pdf_dir = os.path.dirname(os.path.abspath(__file__))
        pdfs = [f for f in os.listdir(pdf_dir) if f.endswith(".pdf")]

        if not pdfs:
            print("RAG: nenhum PDF encontrado — funcionando sem RAG", flush=True)
            _rag_ready = True
            return

        docs, ids, metas = [], [], []
        for fname in pdfs:
            path = os.path.join(pdf_dir, fname)
            try:
                reader = PdfReader(path)
                texto = " ".join(p.extract_text() or "" for p in reader.pages)
                # Divide em chunks de ~600 caracteres com overlap
                chunks = _chunkar(texto, tamanho=600, overlap=80)
                for i, chunk in enumerate(chunks):
                    chunk = chunk.strip()
                    if len(chunk) < 80:
                        continue
                    docs.append(chunk)
                    ids.append(f"{fname}_{i}")
                    metas.append({"fonte": fname})
                print(f"RAG: indexado {fname} ({len(chunks)} chunks)", flush=True)
            except Exception as e:
                print(f"RAG: erro ao processar {fname}: {e}", flush=True)

        if docs:
            embeddings = model.encode(docs).tolist()
            # Insere em lotes para evitar timeout
            batch = 100
            for i in range(0, len(docs), batch):
                _collection.add(
                    documents=docs[i:i+batch],
                    embeddings=embeddings[i:i+batch],
                    ids=ids[i:i+batch],
                    metadatas=metas[i:i+batch],
                )
            print(f"RAG: {len(docs)} chunks indexados com sucesso", flush=True)

        _rag_ready = True

    except ImportError as e:
        print(f"RAG: dependência não encontrada ({e}) — funcionando sem RAG", flush=True)
        _rag_ready = True
    except Exception as e:
        print(f"RAG: erro na inicialização ({e}) — funcionando sem RAG", flush=True)
        _rag_ready = True


def _chunkar(texto, tamanho=600, overlap=80):
    """Divide texto em chunks com overlap."""
    palavras = texto.split()
    chunks, i = [], 0
    while i < len(palavras):
        chunk = " ".join(palavras[i:i+tamanho])
        chunks.append(chunk)
        i += tamanho - overlap
    return chunks


def buscar_contexto(pergunta, n=4):
    """Busca os n chunks mais relevantes para a pergunta."""
    if not _collection:
        return ""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embedding = model.encode([pergunta]).tolist()
        resultados = _collection.query(
            query_embeddings=embedding,
            n_results=n,
            include=["documents", "metadatas"]
        )
        trechos = []
        for doc, meta in zip(resultados["documents"][0], resultados["metadatas"][0]):
            fonte = meta.get("fonte", "referência")
            # Remove nome do arquivo .pdf para ficar mais legível
            fonte_limpa = re.sub(r'_|\.pdf', ' ', fonte).strip()
            trechos.append(f"[{fonte_limpa}]\n{doc.strip()}")
        return "\n\n---\n\n".join(trechos)
    except Exception as e:
        print(f"RAG: erro na busca ({e})", flush=True)
        return ""


def injetar_contexto(payload, contexto):
    """Injeta o contexto RAG no system prompt."""
    if not contexto:
        return payload
    payload = dict(payload)
    system_original = payload.get("system", "")
    payload["system"] = (
        system_original
        + "\n\n════ CONTEXTO DAS REFERÊNCIAS BIBLIOGRÁFICAS ════\n"
        + "Use os trechos abaixo como base para sua resposta. "
        + "Cite a fonte entre parênteses quando usar o conteúdo.\n\n"
        + contexto
        + "\n════ FIM DO CONTEXTO ════"
    )
    return payload


# ── Handler HTTP ───────────────────────────────────────────────────
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
        status = "RespirIA backend ativo! RAG: " + ("pronto" if _rag_ready else "inicializando")
        self.wfile.write(status.encode())

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body   = self.rfile.read(length)

        # ── /api/log ──────────────────────────────────────────────
        if self.path == "/api/log":
            try:
                self._save_to_sheets(json.loads(body))
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

        # ── /api/chat ─────────────────────────────────────────────
        if self.path != "/api/chat":
            self.send_response(404)
            self._cors()
            self.end_headers()
            return

        try:
            payload = json.loads(body)

            # Extrai última mensagem do aluno para busca RAG
            mensagens = payload.get("messages", [])
            ultima_msg = ""
            for m in reversed(mensagens):
                if m.get("role") == "user":
                    ultima_msg = m.get("content", "")
                    break

            # Busca contexto relevante nas referências
            if ultima_msg and _rag_ready:
                contexto = buscar_contexto(ultima_msg)
                if contexto:
                    payload = injetar_contexto(payload, contexto)

            data_out = json.dumps(payload).encode()

        except Exception as e:
            data_out = body  # usa payload original se der erro

        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=data_out,
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
            return
        req = urllib.request.Request(
            SHEETS_URL,
            data=json.dumps(payload).encode(),
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
    # Inicializa RAG em background para não atrasar o start
    import threading
    threading.Thread(target=init_rag, daemon=True).start()
    HTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
