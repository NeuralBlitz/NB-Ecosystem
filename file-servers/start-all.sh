#!/bin/bash
# Start All File Servers
# Usage: ./start-all.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==================================="
echo "  File Servers Control Panel"
echo "==================================="
echo ""
echo "1) WebDAV  - http://localhost:8080"
echo "2) FTP     - ftp://localhost:2121"
echo "3) SFTP    - sftp://localhost:2222"
echo "4) All     - Start all servers"
echo "5) Stop    - Stop all servers"
echo "6) Status  - Check server status"
echo ""
read -p "Select option: " opt

case $opt in
    1)
        echo "[*] Starting WebDAV server..."
        cd "$SCRIPT_DIR/webdav"
        python3 webdav_server.py --port 8080 --auth &
        ;;
    2)
        echo "[*] Starting FTP server..."
        cd "$SCRIPT_DIR/ftp"
        pip install -q pyftpdlib
        python3 ftp_server.py --port 2121 &
        ;;
    3)
        echo "[*] Starting SFTP server..."
        cd "$SCRIPT_DIR/sftp"
        ./sftp-server.sh start &
        ;;
    4)
        echo "[*] Starting all servers..."
        cd "$SCRIPT_DIR/webdav" && python3 webdav_server.py --port 8080 --auth &
        sleep 1
        cd "$SCRIPT_DIR/ftp" && python3 ftp_server.py --port 2121 &
        sleep 1
        cd "$SCRIPT_DIR/sftp" && ./sftp-server.sh start &
        echo "[+] All servers started!"
        echo ""
        echo "WebDAV: http://localhost:8080"
        echo "  User: admin | Pass: admin123"
        echo "FTP: ftp://localhost:2121"
        echo "  User: admin | Pass: admin123"
        echo "SFTP: sftp://localhost:2222"
        echo "  User: sftpuser | Pass: sftppass"
        ;;
    5)
        echo "[*] Stopping all servers..."
        pkill -f "webdav_server" 2>/dev/null
        pkill -f "ftp_server" 2>/dev/null
        pkill -f "sftp-server" 2>/dev/null
        echo "[+] All servers stopped"
        ;;
    6)
        echo "[*] Server status:"
        pgrep -a "python.*webdav" 2>/dev/null || echo "WebDAV: Not running"
        pgrep -a "python.*ftp" 2>/dev/null || echo "FTP: Not running"
        pgrep -a "sshd.*sftp" 2>/dev/null || echo "SFTP: Not running"
        ;;
    *)
        echo "Invalid option"
        ;;
esac
