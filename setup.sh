#!/usr/bin/env bash
# setup.sh – Comprehensive installer for groq-bot

set -e

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BASE_DIR"

_bold()    { printf '\033[1m%s\033[0m'    "$*"; }
_green()   { printf '\033[0;32m%s\033[0m' "$*"; }
_yellow()  { printf '\033[0;33m%s\033[0m' "$*"; }
_red()     { printf '\033[0;31m%s\033[0m' "$*"; }

info()  { echo "$(_green  '[INFO] ') $*"; }
warn()  { echo "$(_yellow '[WARN] ') $*"; }
error() { echo "$(_red    '[ERR]  ') $*" >&2; exit 1; }
step()  { echo ""; echo "$(_bold "=== $* ===")"; }

# 1. System Dependency Installation
step "Checking System Dependencies"

if command -v pkg &>/dev/null; then
    info "Detected Termux environment."
    pkg update -y
    pkg install -y python python-pip ffmpeg curl termux-api cloudflared
elif command -v apt-get &>/dev/null; then
    info "Detected Debian/Ubuntu environment."
    if [[ $EUID -ne 0 ]]; then
        warn "Not running as root. You might be prompted for sudo password."
        SUDO="sudo"
    else
        SUDO=""
    fi
    $SUDO apt-get update
    $SUDO apt-get install -y python3 python3-pip python3-venv ffmpeg curl
else
    warn "Unsupported package manager. Please ensure python3, pip, ffmpeg, and curl are installed."
fi

# 2. Virtual Environment Setup
step "Setting up Python Virtual Environment"
if [ ! -d "venv" ]; then
    python3 -m venv venv || error "Failed to create venv. Is python3-venv installed?"
    info "Created venv."
else
    info "Virtual environment already exists."
fi

# 3. Python Package Installation
step "Installing Python Packages"
./venv/bin/pip install --upgrade pip
./venv/bin/pip install -r requirements.txt
info "Python dependencies (including yt-dlp) installed."

# 4. Link 'bot' command
step "Installing 'bot' Command"
if [ -f "bot.sh" ]; then
    bash bot.sh install_self
else
    error "bot.sh not found. Installation aborted."
fi

# 5. Environment Template
step "Finalizing Configuration"
if [ ! -f ".env" ]; then
    cat > .env <<EOF
GROQ_API_KEY=
HF_API_KEY=
BOT_PORT=5000
EOF
    info "Created .env template. Please edit it with your API keys."
else
    info ".env already exists."
fi

# 6. Success
echo ""
_bold "Setup Complete!"
echo "--------------------------------------------------------"
echo "1. Edit $(_cyan '.env') and add your GROQ_API_KEY."
echo "2. Run $(_green 'bot config') to verify your settings."
echo "3. Use $(_green 'bot start') to launch the bot."
echo "--------------------------------------------------------"
