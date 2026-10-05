#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
echo "[*] Pulling latest changes from GitHub..."
git pull
echo "[*] Restarting topgun-api service..."
echo topgun2026 | sudo -S systemctl restart topgun-api
echo "[+] Done! Service status:"
systemctl is-active topgun-api
