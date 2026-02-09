#!/usr/bin/env python3
"""
Simple FTP Server with Basic Authentication
Usage: python3 ftp_server.py [--port PORT] [--passive-ports MIN:MAX]
"""

import os
import json
import argparse
from pyftpdlib.authorizers import DummyAuthorizer
from pyftpdlib.handlers import FTPHandler
from pyftpdlib.servers import FTPServer
from pyftpdlib.log import config_logging

CONFIG_FILE = os.path.join(os.path.dirname(__file__), "config", "users.json")
ROOT_DIR = os.path.join(os.path.dirname(__file__), "data")


def load_users():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE) as f:
            return json.load(f)
    return {"admin": "admin123", "user": "user123", "anonymous": ""}


def create_server(port=2121, passive_ports=(60000, 60100)):
    os.makedirs(ROOT_DIR, exist_ok=True)
    users = load_users()

    authorizer = DummyAuthorizer()
    for username, password in users.items():
        if password:
            authorizer.add_user(username, password, ROOT_DIR, perm="elradfmw")
        else:
            authorizer.add_anonymous(ROOT_DIR)

    handler = FTPHandler
    handler.authorizer = authorizer
    handler.passive_ports = range(passive_ports[0], passive_ports[1])
    handler.timeout = 300

    server = FTPServer(("0.0.0.0", port), handler)
    server.max_cons = 50
    server.max_cons_per_ip = 10

    config_logging()
    return server


def run_server(port=2121, passive_min=60000, passive_max=60100):
    server = create_server(port, (passive_min, passive_max))
    print(f"[FTP] Server running on ftp://0.0.0.0:{port}")
    print(f"[FTP] Root directory: {ROOT_DIR}")
    print(f"[FTP] Passive ports: {passive_min}-{passive_max}")
    server.serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple FTP Server")
    parser.add_argument("--port", type=int, default=2121, help="Port to listen on")
    parser.add_argument("--passive-ports", default="60000:60100", help="Passive port range")
    args = parser.parse_args()
    passive_min, passive_max = map(int, args.passive_ports.split(":"))
    run_server(args.port, passive_min, passive_max)
