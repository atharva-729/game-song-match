"""Local static server for site/ with HTTP Range support, so audio seeking works (python -m http.server has no
Range support, which makes browsers refuse to seek in the demo mp3s). GitHub Pages supports Range natively.
usage: python scripts/serve.py [port]      then open http://localhost:8000
"""
import functools, http.server, os, re, sys
from pathlib import Path

SITE = Path(__file__).resolve().parents[1] / "site"


class RangeHandler(http.server.SimpleHTTPRequestHandler):
    def send_head(self):
        self._remaining = None
        m = re.fullmatch(r"bytes=(\d*)-(\d*)", self.headers.get("Range", "").strip())
        path = self.translate_path(self.path)
        if not m or not os.path.isfile(path):
            return super().send_head()
        size = os.path.getsize(path)
        start = int(m[1]) if m[1] else max(0, size - int(m[2] or 0))
        end = min(int(m[2]), size - 1) if m[1] and m[2] else size - 1
        if start >= size or start > end:
            self.send_response(416); self.send_header("Content-Range", f"bytes */{size}"); self.end_headers()
            return None
        f = open(path, "rb"); f.seek(start)
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        self._remaining = end - start + 1
        return f

    def copyfile(self, source, outputfile):
        left = getattr(self, "_remaining", None)
        if left is None:
            return super().copyfile(source, outputfile)
        try:
            while left > 0:
                buf = source.read(min(64 * 1024, left))
                if not buf:
                    break
                outputfile.write(buf); left -= len(buf)
        except ConnectionError:
            pass  # the browser dropped a range it no longer needs (normal while seeking)

    def end_headers(self):
        if not self.headers.get("Range"):
            self.send_header("Accept-Ranges", "bytes")
        super().end_headers()

    def log_message(self, *args):
        pass


def make_server(port=0):
    handler = functools.partial(RangeHandler, directory=str(SITE))
    return http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    srv = make_server(port)
    print(f"Serving {SITE} at http://localhost:{port} (Ctrl+C to stop)", flush=True)
    srv.serve_forever()
