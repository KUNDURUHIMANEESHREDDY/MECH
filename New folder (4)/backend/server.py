"""Python Backend HTTP API Server (Sprint 3)

Serves HTTP JSON API endpoints for browser & desktop frontends,
dispatching directly to the backend dispatcher engine.
"""

from __future__ import annotations

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from api.dispatcher import build_dispatcher

dispatcher = build_dispatcher()


class BackendRequestHandler(BaseHTTPRequestHandler):
    def _send_cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS, HEAD")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self) -> None:
        self.send_response(200)
        self._send_cors_headers()
        self.end_headers()

    def do_HEAD(self) -> None:
        self.send_response(200)
        self._send_cors_headers()
        self.send_header("Content-Type", "application/json")
        self.end_headers()

    def do_GET(self) -> None:
        if self.path == "/health" or self.path == "/ping":
            self.send_response(200)
            self._send_cors_headers()
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            res = dispatcher["ping"]({})
            self.wfile.write(json.dumps(res).encode("utf-8"))
            return

        self.send_response(404)
        self.end_headers()

    def do_POST(self) -> None:
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)
        
        try:
            req_data = json.loads(body.decode("utf-8")) if body else {}
        except Exception as exc:
            self.send_response(400)
            self._send_cors_headers()
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": f"Invalid JSON: {exc}"}).encode("utf-8"))
            return

        method = req_data.get("method") or self.path.lstrip("/")
        payload = req_data.get("payload") or req_data

        handler = dispatcher.get(method)
        if not handler:
            self.send_response(404)
            self._send_cors_headers()
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": f"Unknown method: {method}"}).encode("utf-8"))
            return

        try:
            result = handler(payload)
            self.send_response(200)
            self._send_cors_headers()
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"ok": True, "result": result}, default=str).encode("utf-8"))
        except Exception as exc:
            self.send_response(500)
            self._send_cors_headers()
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(exc)}).encode("utf-8"))


def run_server(port: int = 8000) -> None:
    server_address = ("0.0.0.0", port)
    httpd = HTTPServer(server_address, BackendRequestHandler)
    print(f"Backend Python API Server running on http://localhost:{port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    port_num = 8000
    if len(sys.argv) > 1:
        try:
            port_num = int(sys.argv[1])
        except ValueError:
            pass
    run_server(port_num)
