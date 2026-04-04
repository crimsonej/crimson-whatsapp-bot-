#!/usr/bin/env bash
# =============================================================================
#  groq-bot · setup.sh
#  The ultimate installer for the Groq RAG WhatsApp bot.
# =============================================================================

set -e

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BASE_DIR"

# ─────────────────────────────────────────────────────────────────────────────
# Output helpers
# ─────────────────────────────────────────────────────────────────────────────
_bold()    { printf '\033[1m%s\033[0m'    "$*"; }
_green()   { printf '\033[0;32m%s\033[0m' "$*"; }
_yellow()  { printf '\033[0;33m%s\033[0m' "$*"; }
_red()     { printf '\033[0;31m%s\033[0m' "$*"; }
_cyan()    { printf '\033[0;36m%s\033[0m' "$*"; }

info()  { echo "$(_green  '[INFO] ') $*"; }
warn()  { echo "$(_yellow '[WARN] ') $*"; }
error() { echo "$(_red    '[ERR]  ') $*" >&2; exit 1; }
step()  { echo ""; echo "$(_bold "=== $* ===")"; }

# ─────────────────────────────────────────────────────────────────────────────
# 1. System Dependency Installation
# ─────────────────────────────────────────────────────────────────────────────
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

# ─────────────────────────────────────────────────────────────────────────────
# 2. Directory Setup
# ─────────────────────────────────────────────────────────────────────────────
step "Creating Project Structure"
mkdir -p docs sticker_cache
info "Created 'docs/' and 'sticker_cache/' directories."

# ─────────────────────────────────────────────────────────────────────────────
# 3. Virtual Environment Setup
# ─────────────────────────────────────────────────────────────────────────────
step "Setting up Python Virtual Environment"
if [ ! -d "venv" ]; then
    python3 -m venv venv || error "Failed to create venv. Is python3-venv installed?"
    info "Created venv."
else
    info "Virtual environment already exists."
fi

# ─────────────────────────────────────────────────────────────────────────────
# 4. Python Package Installation
# ─────────────────────────────────────────────────────────────────────────────
step "Installing Python Packages"
./venv/bin/pip install --upgrade pip
./venv/bin/pip install -r requirements.txt
info "Python dependencies (including yt-dlp) installed."

# ─────────────────────────────────────────────────────────────────────────────
# 5. Environment Configuration & Security
# ─────────────────────────────────────────────────────────────────────────────
step "Securing Environment (.env)"

if [ ! -f ".env" ]; then
    touch .env
    chmod 600 .env
    info "Created .env with secure permissions (600)."
else
    chmod 600 .env
    info "Secured existing .env permissions."
fi

# Initialize keys if missing
update_key() {
    local key_name=$1
    local prompt_msg=$2
    if ! grep -q "^${key_name}=" .env; then
        read -rp "   [?] ${prompt_msg}: " val
        echo "${key_name}=${val}" >> .env
    elif [[ -z "$(grep "^${key_name}=" .env | cut -d'=' -f2)" ]]; then
        read -rp "   [?] ${key_name} is empty. ${prompt_msg}: " val
        sed -i "s/^${key_name}=.*/${key_name}=${val}/" .env
    fi
}

echo "   Interactive API Key Configuration (Enter to skip):"
update_key "GROQ_API_KEY" "Enter your Groq API Key"
update_key "NVIDIA_API_KEY" "Enter your NVIDIA API Key (for vision)"
update_key "HF_API_KEY" "Enter your Hugging Face API Key (for images)"

if ! grep -q "^BOT_PORT=" .env; then
    echo "BOT_PORT=5000" >> .env
fi

# ─────────────────────────────────────────────────────────────────────────────
# 6. Global Command Installation
# ─────────────────────────────────────────────────────────────────────────────
step "Installing 'bot' Command"
if [ -f "bot.sh" ]; then
    bash bot.sh install_self
else
    error "bot.sh not found. Installation aborted."
fi

# ─────────────────────────────────────────────────────────────────────────────
# 7. Success
# ─────────────────────────────────────────────────────────────────────────────
echo ""
_bold "✨ Installation Complete! ✨"
echo "────────────────────────────────────────────────────────"
echo " 1. Drop knowledge files (.txt) into $(_cyan 'docs/') folder."
echo " 2. Run $(_green 'bot config') to tweak your persona/threshold."
echo " 3. Use $(_green 'bot start') to launch your bot."
echo "────────────────────────────────────────────────────────"
echo "Security Note: Your API keys are stored in .env which is"
echo "ignored by git and locked to your user account."
echo ""
