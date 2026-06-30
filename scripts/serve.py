"""Serve the project root (with HTTP Range support, for video seeking) and
open the GDPval task viewer."""
import http.server
import os
import re
import socketserver
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PORT = 8765


class RangeRequestHandler(http.server.SimpleHTTPRequestHandler):
    """SimpleHTTPRequestHandler with byte-range support so <video>/<audio>
    seeking works for large local media files."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def send_head(self):
        path = self.translate_path(self.path)
        if os.path.isdir(path) or not os.path.exists(path):
            return super().send_head()

        range_header = self.headers.get("Range")
        if not range_header:
            self.send_header_extra = {}
            response = super().send_head()
            return response

        file_size = os.path.getsize(path)
        match = re.match(r"bytes=(\d*)-(\d*)", range_header)
        if not match:
            return super().send_head()
        start_s, end_s = match.groups()
        start = int(start_s) if start_s else 0
        end = int(end_s) if end_s else file_size - 1
        end = min(end, file_size - 1)
        if start > end or start >= file_size:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{file_size}")
            self.end_headers()
            return None

        f = open(path, "rb")
        f.seek(start)
        length = end - start + 1

        self.send_response(206)
        ctype = self.guess_type(path)
        self.send_header("Content-type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
        self.send_header("Content-Length", str(length))
        self.end_headers()
        self._range = (start, length)
        return f

    def copyfile(self, source, outputfile):
        if hasattr(self, "_range"):
            start, length = self._range
            remaining = length
            chunk_size = 1 << 16
            while remaining > 0:
                chunk = source.read(min(chunk_size, remaining))
                if not chunk:
                    break
                outputfile.write(chunk)
                remaining -= len(chunk)
        else:
            super().copyfile(source, outputfile)

    def end_headers(self):
        # Avoid stale cached HTML/JS/CSS while iterating on the viewer.
        self.send_header("Cache-Control", "no-store, must-revalidate")
        super().end_headers()


def main():
    url = f"http://localhost:{PORT}/viewer/index.html"
    with socketserver.ThreadingTCPServer(("", PORT), RangeRequestHandler) as httpd:
        print(f"Serving {ROOT} at {url}")
        webbrowser.open(url)
        httpd.serve_forever()


if __name__ == "__main__":
    main()
