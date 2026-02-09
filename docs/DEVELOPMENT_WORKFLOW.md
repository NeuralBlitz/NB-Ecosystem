# Development Workflow

## Overview

This document outlines the development workflow for NeuralBlitz to ensure IP protection.

## Golden Rules

### ⚠️ NEVER DO THESE

1. ❌ Never `git add` any critical asset directory
2. ❌ Never `git commit` engine code
3. ❌ Never `git push` to any remote
4. ❌ Never commit `.env` files
5. ❌ Never commit API keys or credentials
6. ❌ Never commit partner configurations

### ✅ ALWAYS DO THESE

1. ✅ Always work in `/home/runner/workspace/` directory
2. ✅ Always use `.gitignore` (already configured)
3. ✅ Always verify before committing
4. ✅ Always create backups before major changes
5. ✅ Always test in isolation

## Development Environment

### Directory Structure

```
/home/runner/workspace/
├── NBX-LRS/                    # 🔴 CRITICAL - Never commit
├── NB-Ecosystem/              # 🔴 CRITICAL - Never commit
├── Emergent-Prompt-Architecture/ # 🔴 CRITICAL - Never commit
├── opencode-lrs-agents-nbx/   # 🔴 CRITICAL - Never commit
├── Advanced-Research/         # 🔴 CRITICAL - Never commit
├── quantum_sim/               # 🔴 CRITICAL - Never commit
├── nb-omnibus-router/         # 🟢 SAFE - Omnibus Router (public)
├── neuralblitz-core/          # 🟢 SAFE - Public SDK (interfaces only)
├── neuralblitz-agents/        # 🟢 SAFE - Public SDK (interfaces only)
├── neuralblitz-ui/            # 🟢 SAFE - Public SDK (interfaces only)
├── scripts/                   # 🟡 MEDIUM - Automation scripts
├── docs/                      # 🟢 SAFE - Documentation
├── backups/                   # 🟡 MEDIUM - Encrypted backups
└── .gitignore                # ✅ Configured - IP protection
```

## Git Workflow

### For Public Repositories (SDKs)

These repos contain ONLY interfaces - safe to push:

```bash
# neuralblitz-core/
cd neuralblitz-core
git init
git add .
git commit -m "Initial commit: interface definitions"
git remote add origin https://github.com/yourusername/neuralblitz-core.git
git push -u origin main

# neuralblitz-agents/
cd neuralblitz-agents
git init
git add .
git commit -m "Initial commit: agent interfaces"
git push -u origin main

# neuralblitz-ui/
cd neuralblitz-ui
git init
git add .
git commit -m "Initial commit: UI interfaces"
git push -u origin main
```

### For Omnibus Router

This goes to your private server (not GitHub):

```bash
cd nb-omnibus-router
# Build Docker image
docker build -t neuralblitz-router .
# Deploy to your server
./deploy.sh production
```

### For Critical Assets

**NEVER** use Git for these directories:

```bash
# WRONG - Never do this!
cd NBX-LRS
git init  # ❌ STOP!
git add . # ❌ STOP!
git commit -m "update" # ❌ STOP!

# RIGHT - Just edit files directly!
cd NBX-LRS
vim quantum_spiking_neuron.py  # ✅ Edit directly
python test_neuron.py  # ✅ Test directly
```

## Daily Development Cycle

### 1. Start of Day

```bash
# Check system status
cd /home/runner/workspace
./scripts/backup.sh  # Create backup

# Check git status (should be clean for critical dirs)
git status  # Should show only docs/, scripts/, nb-omnibus-router/

# Verify no .git directories in critical assets
find . -name ".git" -type d | grep -v "^\./\.git$"
```

### 2. During Development

```bash
# Edit engine code directly
cd /home/runner/workspace/NBX-LRS
vim opencode-lrs-agents-nbx/neuralblitz-v50/Advanced-Research/production/quantum_spiking_neuron.py

# Test changes
python3 opencode-lrs-agents-nbx/neuralblitz-v50/Advanced-Research/production/quantum_spiking_neuron.py

# If working on Omnibus Router
cd /home/runner/workspace/nb-omnibus-router
python3 -m uvicorn api.main:app --reload
```

### 3. End of Day

```bash
# Create backup
./scripts/backup.sh encrypt

# Verify no accidental commits
git status

# Check for any new .git directories
find . -name ".git" -type d
```

## Backup Schedule

| Time | Action | Command |
|------|--------|---------|
| Daily | Incremental backup | `./scripts/backup.sh` |
| Weekly | Encrypted backup | `./scripts/backup.sh encrypt` |
| Before major changes | Full backup | `./scripts/backup.sh encrypt` |

## Emergency Procedures

### Accidentally Committed Sensitive Data

```bash
# 1. Identify the issue
git log --name-status | grep -i "NBX-LRS\|NB-Ecosystem\|API_KEY"

# 2. Remove from Git (but file remains)
git rm --cached -r NBX-LRS/
git commit -m "Remove accidentally committed sensitive data"

# 3. Remove .git directory entirely
rm -rf NBX-LRS/.git

# 4. Rotate any exposed credentials
# Check logs, notify team
```

### Suspected Breach

```bash
# 1. Disconnect from network
# 2. Document everything
# 3. Restore from backup
./scripts/restore.sh 20260208_045830

# 4. Rotate all API keys
# 5. Review access logs
# 6. Contact security team
```

## Verification Checklist

Before any push to public repos:

```bash
# 1. Verify .gitignore is working
git status --ignored | head -20

# 2. Check for critical directories in staging
git status | grep -E "NBX-LRS|NB-Ecosystem|Emergent"

# 3. Verify no .env files
find . -name ".env" -type f

# 4. Check for secrets in recent commits
git log --all -p | grep -i "api_key\|password\|secret" | head -20
```

## Partner Onboarding Workflow

### Adding a New Partner

1. **Generate API key**
   ```bash
   # Use the key generator
   python3 scripts/generate_api_key.py
   ```

2. **Add to partners.yaml**
   ```yaml
   partners:
     partner_new:
       api_key: "nb_pat_xxxxxxxxxxxx"
       name: "New Partner"
       tier: "pro"
       active: true
       permissions: ["core", "agents"]
   ```

3. **Send credentials securely**
   ```bash
   # Encrypt the credentials
   gpg --symmetric --cipher-algo AES256 partner_credentials.txt
   ```

4. **Document the partnership**
   ```bash
   echo "Partner: New Partner" >> docs/PARTNERS.md
   echo "Date: $(date)" >> docs/PARTNERS.md
   ```

---
*Document created: 2026-02-08*
