"""
app.py - Simple Web Server for SLM Enterprise AI Assistant.
"""

import os
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from harness import Harness

INDEX_HTML = Path(__file__).parent / "template" / "index.html"

# Initialize AI Harness
print("Initializing SLM Enterprise AI Harness...")
harness = Harness()
print("Harness ready.")


class ChatHandler(BaseHTTPRequestHandler):
    """Simple HTTP request handler for the UI and chat API."""

    def _send_response(self, status=200, content_type="application/json", data=b""):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        # Serve main index.html
        if self.path in ("/", "/index.html"):
            self._send_response(200, "text/html; charset=utf-8", INDEX_HTML.read_bytes())
            return
        self._send_response(404, "text/plain", b"Not Found")

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length).decode("utf-8")) if length > 0 else {}

        # 1. Chat API Endpoint
        if self.path == "/api/chat":
            result = harness.process(
                query=body.get("query", ""),
                session_id=body.get("session_id", "default"),
                provided_fields=body.get("provided_fields", {}),
                required_fields=body.get("required_fields", []),
            )
            self._send_response(200, "application/json", json.dumps(result).encode("utf-8"))
            return

        # 2. Reset Session Endpoint
        if self.path == "/api/reset":
            harness.reset_session(body.get("session_id", "default"))
            self._send_response(200, "application/json", b'{"status": "reset"}')
            return

        self._send_response(404, "application/json", b'{"error": "Not Found"}')


def main():
    port = int(os.environ.get("PORT", 8000))
    server = HTTPServer(("", port), ChatHandler)
    print("=" * 60)
    print(f">> SLM Enterprise AI Assistant running at: http://localhost:{port}")
    print("=" * 60)
    print("Open http://localhost:8000 in your browser. Press Ctrl+C to exit.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
        server.server_close()


if __name__ == "__main__":
    main()