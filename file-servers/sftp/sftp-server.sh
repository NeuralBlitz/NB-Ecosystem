#!/bin/bash
# SFTP Server Setup Script
# Usage: ./sftp-server.sh [start|stop|restart]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$SCRIPT_DIR/data"
CONFIG_FILE="$SCRIPT_DIR/config/users.json"
SSH_DIR="$HOME/.ssh"
SFTP_USER="sftpuser"
SFTP_PASS="sftppass"

start() {
    mkdir -p "$DATA_DIR"

    # Create user if not exists
    if ! id "$SFTP_USER" &>/dev/null; then
        useradd -m -d "$DATA_DIR" -s /usr/sbin/nologin "$SFTP_USER" 2>/dev/null || {
            echo "[SFTP] Using existing user or running without chroot"
        }
        echo "$SFTP_USER:$SFTP_PASS" | chpasswd 2>/dev/null || true
    fi

    # Set permissions
    chmod 755 "$DATA_DIR"

    # Start sshd on port 2222
    if [ -f /usr/sbin/sshd ]; then
        /usr/sbin/sshd -f "$SCRIPT_DIR/sshd_config" -E "$SCRIPT_DIR/sshd.log" -p 2222
        echo "[SFTP] Server started on port 2222"
        echo "[SFTP] Connect: sftp -P 2222 $SFTP_USER@localhost"
    else
        echo "[SFTP] sshd not available, using alternative method"
        python3 -m pysftp -p 2222 "$SFTP_USER@localhost" 2>/dev/null || \
        echo "[SFTP] Install openssh-server to run SFTP server"
    fi
}

stop() {
    pkill -f "sshd.*-f.*sftp-server" 2>/dev/null
    echo "[SFTP] Server stopped"
}

restart() {
    stop
    sleep 1
    start
}

case "${1:-start}" in
    start) start ;;
    stop) stop ;;
    restart) restart ;;
    *) echo "Usage: $0 [start|stop|restart]" ;;
esac
