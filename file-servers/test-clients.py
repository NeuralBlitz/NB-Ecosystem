#!/usr/bin/env python3
"""
Test client for all file servers
Usage: python3 test-clients.py
"""

import socket
import ftplib
import urllib.request
import base64
import json
import os


def test_webdav():
    print("[WebDAV] Testing...")
    try:
        url = "http://localhost:8080/"
        auth = base64.b64encode(b"admin:admin123").decode()
        req = urllib.request.Request(url, headers={"Authorization": f"Basic {auth}"})
        urllib.request.urlopen(req, timeout=2)
        print("[WebDAV] ✓ Connection successful")
        return True
    except Exception as e:
        print(f"[WebDAV] ✗ Failed: {e}")
        return False


def test_ftp():
    print("[FTP] Testing...")
    try:
        ftp = ftplib.FTP()
        ftp.connect("localhost", 2121, timeout=2)
        ftp.login("admin", "admin123")
        ftp.quit()
        print("[FTP] ✓ Connection successful")
        return True
    except Exception as e:
        print(f"[FTP] ✗ Failed: {e}")
        return False


def test_sftp_port():
    print("[SFTP] Testing port 2222...")
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex(("localhost", 2222))
        sock.close()
        if result == 0:
            print("[SFTP] ✓ Port open")
            return True
        else:
            print("[SFTP] ✗ Port closed")
            return False
    except Exception as e:
        print(f"[SFTP] ✗ Failed: {e}")
        return False


def test_config():
    print("\n[Config] Credentials file:")
    config_path = "/home/runner/workspace/file-servers/config/users.json"
    if os.path.exists(config_path):
        with open(config_path) as f:
            users = json.load(f)
            for user in users:
                print(f"  {user}: {users[user]}")
    else:
        print("  Config file not found")


if __name__ == "__main__":
    print("=" * 40)
    print("  File Server Test Suite")
    print("=" * 40)
    test_webdav()
    test_ftp()
    test_sftp_port()
    test_config()
