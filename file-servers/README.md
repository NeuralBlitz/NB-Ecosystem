# File Servers Setup

## Quick Start

### SFTP (Already Available)
```bash
# Connect
sftp -oPort=22 user@localhost

# Or use the included script
./sftp-server.sh start
```

### WebDAV Server
```bash
cd /home/runner/workspace/file-servers/webdav
python3 webdav_server.py --help
python3 webdav_server.py --port 8080 --auth
```

### FTP Server
```bash
cd /home/runner/workspace/file-servers/ftp
python3 ftp_server.py --help
python3 ftp_server.py --port 2121 --auth
```

## Credentials

All servers support the same credentials file: `config/users.json`

```json
{
  "username": "password"
}
```

## Security Notes

- These are basic servers for development/testing
- WebDAV and FTP transmit credentials in plaintext
- SFTP is encrypted and recommended for production
- Always use strong passwords and consider firewall rules
