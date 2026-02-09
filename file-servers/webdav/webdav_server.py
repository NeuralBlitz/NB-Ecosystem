#!/usr/bin/env python3
"""
Simple WebDAV Server with Basic Authentication
Usage: python3 webdav_server.py [--port PORT] [--auth] [--cert CERT] [--key KEY]
"""

import os
import json
import ssl
import argparse
from http.server import HTTPServer, BaseHTTPRequestHandler
from functools import wraps
import base64

CONFIG_FILE = os.path.join(os.path.dirname(__file__), "config", "users.json")
ROOT_DIR = os.path.join(os.path.dirname(__file__), "data")

def load_users():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE) as f:
            return json.load(f)
    return {"admin": "admin123", "user": "user123"}

def require_auth(f):
    @wraps(f)
    def wrapper(self, *args, **kwargs):
        if not self.authenticate():
            self.send_response(401)
            self.send_header('WWW-Authenticate', 'Basic realm="WebDAV"')
            self.send_header('Content-Type', 'text/plain')
            self.end_headers()
            self.wfile.write(b'Authentication required')
            return
        return f(self, *args, **kwargs)
    return wrapper

class WebDAVHandler(BaseHTTPRequestHandler):
    users = load_users()

    def authenticate(self):
        auth_header = self.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Basic '):
            return False
        encoded = auth_header[6:]
        decoded = base64.b64decode(encoded).decode('utf-8')
        username, password = decoded.split(':', 1)
        return self.users.get(username) == password

    def log_message(self, format, *args):
        print(f"[WebDAV] {args[0]}")

    @require_auth
    def do_GET(self):
        path = self.path.split('?')[0]
        filepath = os.path.join(ROOT_DIR, path.lstrip('/'))
        if os.path.isdir(filepath):
            filepath = os.path.join(filepath, 'index.html')
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                self.send_response(200)
                self.send_header('Content-Type', 'application/octet-stream')
                self.send_header('Content-Length', os.path.getsize(filepath))
                self.end_headers()
                self.wfile.write(f.read())
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b'Not Found')

    @require_auth
    def do_PUT(self):
        path = self.path.split('?')[0]
        filepath = os.path.join(ROOT_DIR, path.lstrip('/'))
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        content_length = int(self.headers.get('Content-Length', 0))
        with open(filepath, 'wb') as f:
            f.write(self.rfile.read(content_length))
        self.send_response(201)
        self.end_headers()
        self.wfile.write(b'Created')

    @require_auth
    def do_DELETE(self):
        path = self.path.split('?')[0]
        filepath = os.path.join(ROOT_DIR, path.lstrip('/'))
        if os.path.exists(filepath):
            os.remove(filepath)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'Deleted')
        else:
            self.send_response(404)
            self.end_headers()

    @require_auth
    def do MKCOL(self):
        path = self.path.split('?')[0]
        dirpath = os.path.join(ROOT_DIR, path.lstrip('/'))
        os.makedirs(dirpath, exist_ok=True)
        self.send_response(201)
        self.end_headers()
        self.wfile.write(b'Created')

def run_server(port=8080, use_ssl=False, cert_file=None, key_file=None):
    os.makedirs(ROOT_DIR, exist_ok=True)
    server = HTTPServer(('0.0.0.0', port), WebDAVHandler)
    if use_ssl and cert_file and key_file:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(cert_file, key_file)
        server.socket = context.wrap_socket(server.socket, server_side=True)
    print(f"[WebDAV] Server running on http{'s' if use_ssl else ''}://0.0.0.0:{port}")
    print(f"[WebDAV] Root directory: {ROOT_DIR}")
    print(f"[WebDAV] Auth required: Yes")
    server.serve_forever()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simple WebDAV Server')
    parser.add_argument('--port', type=int, default=8080, help='Port to listen on')
    parser.add_argument('--auth', action='store_true', help='Enable authentication')
    parser.add_argument('--ssl', action='store_true', help='Enable HTTPS')
    parser.add_argument('--cert', default='cert.pem', help='SSL certificate file')
    parser.add_argument('--key', default='key.pem', help='SSL key file')
    args = parser.parse_args()
    run_server(args.port, args.ssl, args.cert, args.key)
