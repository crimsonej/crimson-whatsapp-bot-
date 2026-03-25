#!/usr/bin/env bash
# =============================================================================
#  groq-bot · bot.sh
#  Termux service manager for the Groq RAG WhatsApp bot.
#
#  Cloudflare tunnel modes
#  ───────────────────────
#  Quick tunnel  (default)
#    Uses `cloudflared tunnel --url` – URL changes on every restart.
#
#  Named tunnel  (permanent URL)
#    Requires a free Cloudflare account and a domain you control.
#    Run `bot tunnel-setup` once; afterwards `bot start` always uses the
#    same URL.
#
#  Usage
#  ─────
#    bot start | stop | restart | status | logs | chat | config | reindex
#    bot tunnel-setup   – guided setup for a permanent Cloudflare URL
#    bot uninstall
# =============================================================================

# Note: we intentionally avoid 'set -e' — Termux pip/pkg commands
# can return non-zero on warnings and would kill the script prematurely.

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

BASE="$HOME/groq-bot"
VENV="$BASE/venv"
ENGINE="$BASE/bot.py"
BOT_PID="$BASE/bot.pid"
BOT_LOG="$BASE/bot.log"
DOCS="$BASE/docs"

CF_PID="$BASE/cloudflared.pid"
CF_LOG="$BASE/cloudflared.log"
CF_URL_FILE="$BASE/tunnel.url"
CF_BIN="/data/data/com.termux/files/usr/bin/cloudflared"
CF_CFG="$BASE/cloudflare/config.yml"      # named-tunnel config
CF_NAMED_FLAG="$BASE/.named_tunnel"       # presence = named tunnel configured

GLOBAL="/usr/local/bin/bot"
[[ -w "/usr/local/bin" ]] || GLOBAL="$HOME/bin/bot"
mkdir -p "$HOME/bin"

# ─────────────────────────────────────────────────────────────────────────────
# Output helpers
# ─────────────────────────────────────────────────────────────────────────────

_bold()    { printf '\033[1m%s\033[0m'    "$*"; }
_green()   { printf '\033[0;32m%s\033[0m' "$*"; }
_yellow()  { printf '\033[0;33m%s\033[0m' "$*"; }
_red()     { printf '\033[0;31m%s\033[0m' "$*"; }
_cyan()    { printf '\033[0;36m%s\033[0m' "$*"; }

info()  { echo "  $(_green  '[INFO] ') $*"; }
warn()  { echo "  $(_yellow '[WARN] ') $*"; }
error() { echo "  $(_red    '[ERR]  ') $*" >&2; exit 1; }
step()  { echo ""; echo "  $(_bold "$*")"; }

progress() {
    local pct=$1 msg="$2" width=38
    local filled=$(( pct * width / 100 ))
    local empty=$(( width - filled ))
    local bar
    bar=$(printf '%0.s=' $(seq 1 "$filled") 2>/dev/null || printf "%${filled}s" | tr ' ' '=')
    local space
    space=$(printf "%${empty}s")
    printf "\r  [%s%s] %3d%%  %s" "$bar" "$space" "$pct" "$msg"
}
newline() { printf "\n"; }

# ─────────────────────────────────────────────────────────────────────────────
# Self-install  (copies this script to $PATH as `bot`)
# ─────────────────────────────────────────────────────────────────────────────

install_self() {
    [[ "$0" == "$GLOBAL" ]] && return
    step "Installing 'bot' command..."
    cp "$0" "$GLOBAL" && chmod +x "$GLOBAL" \
        && info "Installed.  Now run: $(_bold 'bot start')" \
        && return 0
    warn "Could not write to $GLOBAL – skipping global command install."
}

# ─────────────────────────────────────────────────────────────────────────────
# Directory setup
# ─────────────────────────────────────────────────────────────────────────────

ensure_dirs() { mkdir -p "$BASE" "$DOCS"; }

# ─────────────────────────────────────────────────────────────────────────────
# Python virtual environment
# ─────────────────────────────────────────────────────────────────────────────

create_venv() {
    [[ -d "$VENV" ]] && return
    step "Creating Python virtual environment..."
    if ! command -v python3 &>/dev/null; then
        info "Installing Python via pkg..."
        pkg install -y python python-pip 2>/dev/null \
            || error "Could not install Python.  Run: pkg install python"
    fi
    python3 -m venv "$VENV" || error "Failed to create virtual environment."
    info "Virtual environment created."
}

activate_venv() {
    # shellcheck disable=SC1091
    source "$VENV/bin/activate" || error "Could not activate virtual environment."
}

# ─────────────────────────────────────────────────────────────────────────────
# Python dependency management  (only installs what is missing)
# ─────────────────────────────────────────────────────────────────────────────

REQUIRED_PKGS=(flask requests together python-dotenv)

check_and_install() {
    create_venv
    activate_venv

    local missing=()
    for pkg in "${REQUIRED_PKGS[@]}"; do
        python3 -c "import $pkg" 2>/dev/null || missing+=("$pkg")
    done

    # Nothing missing — skip silently (fast path on every subsequent run)
    [[ ${#missing[@]} -eq 0 ]] && return 0

    step "Installing Python dependencies..."
    local total=${#missing[@]}
    local i=1                         # start at 1 to avoid arithmetic exit-code 1
    for pkg in "${missing[@]}"; do
        progress $(( i * 100 / total )) "Installing $pkg"
        pip install --quiet --disable-pip-version-check "$pkg" 2>/dev/null             || warn "pip failed for $pkg – try manually: pip install $pkg"
        i=$(( i + 1 ))                # plain arithmetic, never exits non-zero
    done
    newline
    info "All dependencies ready."
}

# ─────────────────────────────────────────────────────────────────────────────
# Termux:API installation  (required for wake lock)
# ─────────────────────────────────────────────────────────────────────────────

install_termux_api() {
    # Check if the termux-wake-lock binary is already present
    command -v termux-wake-lock &>/dev/null && return 0

    step "Installing Termux:API package..."
    pkg install -y termux-api 2>/dev/null         && info "termux-api installed."         || warn "Could not install termux-api automatically. Run: pkg install termux-api"

    # Remind the user about the companion Android app (can't be auto-installed)
    echo ""
    echo "  [1;33m[ACTION REQUIRED][0m"
    echo "  Install the free [1mTermux:API[0m companion app to enable wake lock:"
    echo "  [0;36mhttps://f-droid.org/packages/com.termux.api/[0m"
    echo "  (also available on Google Play)"
    echo ""
}

# ─────────────────────────────────────────────────────────────────────────────
# cloudflared installation
# ─────────────────────────────────────────────────────────────────────────────

install_cloudflared() {
    command -v cloudflared &>/dev/null && return 0

    step "Installing cloudflared..."

    # Termux package manager first (cleanest, keeps it updated via pkg)
    if command -v pkg &>/dev/null && pkg install -y cloudflared 2>/dev/null; then
        info "cloudflared installed via pkg."
        return 0
    fi

    # Fallback: download pre-built binary for this CPU architecture
    local arch url
    arch=$(uname -m)
    case "$arch" in
        aarch64|arm64) url="https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm64" ;;
        armv7l|armv8l) url="https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm"   ;;
        x86_64)        url="https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64" ;;
        i686|i386)     url="https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-386"   ;;
        *) error "Unsupported CPU architecture: $arch.  Install cloudflared manually." ;;
    esac

    info "Downloading binary for $arch..."
    if command -v curl &>/dev/null; then
        curl -fsSL "$url" -o "$CF_BIN" || error "Download failed."
    elif command -v wget &>/dev/null; then
        wget -q "$url" -O "$CF_BIN" || error "Download failed."
    else
        error "curl and wget not found.  Run: pkg install curl"
    fi
    chmod +x "$CF_BIN"
    info "cloudflared installed."
}

# ─────────────────────────────────────────────────────────────────────────────
# Named tunnel setup  (one-time, gives a permanent URL)
# ─────────────────────────────────────────────────────────────────────────────

tunnel_setup() {
    install_cloudflared

    echo ""
    echo "  $(_bold 'Named Tunnel Setup')  –  permanent public URL"
    echo ""
    echo "  You need:"
    echo "    1. A free Cloudflare account  (cloudflare.com)"
    echo "    2. A domain added to Cloudflare (free subdomain via pages.dev also works)"
    echo ""
    echo "  This will open a browser to authenticate cloudflared with your account."
    echo ""
    read -rp "  Continue? [y/N] " ok
    [[ "$ok" =~ ^[Yy]$ ]] || { info "Aborted."; return 0; }

    # Step 1 – authenticate
    step "Step 1/4  Authenticate with Cloudflare..."
    cloudflared tunnel login

    # Step 2 – create tunnel
    step "Step 2/4  Create tunnel..."
    read -rp "  Choose a tunnel name (e.g. groq-bot): " tname
    tname="${tname:-groq-bot}"
    cloudflared tunnel create "$tname"

    # Step 3 – hostname
    step "Step 3/4  Configure hostname..."
    read -rp "  Enter the hostname you want (e.g. bot.yourdomain.com): " hostname
    cloudflared tunnel route dns "$tname" "$hostname"

    # Step 4 – write config.yml
    step "Step 4/4  Writing config..."
    mkdir -p "$(dirname "$CF_CFG")"
    local port="${BOT_PORT:-5000}"
    # Find the credentials file cloudflared just created
    local creds
    creds=$(find "$HOME/.cloudflared" -name "*.json" 2>/dev/null | head -1)

    cat > "$CF_CFG" <<EOF
tunnel: $tname
credentials-file: $creds

ingress:
  - hostname: $hostname
    service: http://localhost:$port
  - service: http_status:404
EOF

    echo "$hostname" > "$CF_NAMED_FLAG"
    echo "https://$hostname" > "$CF_URL_FILE"

    echo ""
    info "Named tunnel configured!"
    info "Your permanent URL: $(_cyan "https://$hostname")"
    info "API endpoint:       $(_cyan "https://$hostname/reply")"
    echo ""
    info "Run $(_bold 'bot start') to launch."
    echo ""
}

# ─────────────────────────────────────────────────────────────────────────────
# Tunnel start / stop
# ─────────────────────────────────────────────────────────────────────────────

_kill_tunnel() {
    if [[ -f "$CF_PID" ]]; then
        kill "$(cat "$CF_PID")" 2>/dev/null || true
        rm -f "$CF_PID"
    fi
}

start_tunnel() {
    local port="${BOT_PORT:-5000}"
    _kill_tunnel
    install_cloudflared

    rm -f "$CF_URL_FILE"

    if [[ -f "$CF_NAMED_FLAG" ]]; then
        # ── Named tunnel (permanent URL) ─────────────────────────────────────
        local hostname
        hostname=$(cat "$CF_NAMED_FLAG")
        info "Starting named tunnel → https://$hostname"
        nohup cloudflared tunnel --config "$CF_CFG" run \
            --no-autoupdate >"$CF_LOG" 2>&1 &
        echo $! > "$CF_PID"
        # Named tunnels come up almost instantly
        sleep 2
        local public_url="https://$hostname"
        echo "$public_url" > "$CF_URL_FILE"
    else
        # ── Quick tunnel (random URL) ─────────────────────────────────────────
        info "Starting quick tunnel (URL changes on restart)..."
        info "Tip: run $(_bold 'bot tunnel-setup') once for a permanent URL."
        nohup cloudflared tunnel --url "http://localhost:$port" \
            --no-autoupdate >"$CF_LOG" 2>&1 &
        echo $! > "$CF_PID"

        # Poll log up to 20 s for the assigned URL
        local waited=0 public_url=""
        while (( waited < 20 )); do
            sleep 1 && (( waited++ ))
            public_url=$(grep -oP 'https://[a-z0-9\-]+\.trycloudflare\.com' \
                "$CF_LOG" 2>/dev/null | head -1)
            [[ -n "$public_url" ]] && break
        done

        if [[ -z "$public_url" ]]; then
            warn "Tunnel started but URL not detected yet – check: bot status"
            return 0
        fi
        echo "$public_url" > "$CF_URL_FILE"
    fi

    _print_url_box
}

stop_tunnel() {
    _kill_tunnel
    rm -f "$CF_URL_FILE"
}

_print_url_box() {
    [[ ! -f "$CF_URL_FILE" ]] && return
    local url endpoint
    url=$(cat "$CF_URL_FILE")
    endpoint="$url/reply"
    local width=66
    local border
    border=$(printf '─%.0s' $(seq 1 $width))
    echo ""
    echo "  ┌${border}┐"
    printf  "  │  %-*s  │\n" $(( width - 2 )) "$(_bold 'Public URL') : $(_cyan "$url")"
    printf  "  │  %-*s  │\n" $(( width - 2 )) "$(_bold 'Endpoint')   : $(_cyan "$endpoint")"
    echo "  └${border}┘"
    echo ""
}

# ─────────────────────────────────────────────────────────────────────────────
# Bot process management
# ─────────────────────────────────────────────────────────────────────────────

start_bot() {
    ensure_dirs

    [[ -f "$ENGINE" ]] || error "bot.py not found at $ENGINE – copy it there first."

    # Guard against double-start
    if [[ -f "$BOT_PID" ]]; then
        local pid
        pid=$(cat "$BOT_PID")
        if kill -0 "$pid" 2>/dev/null; then
            info "Bot is already running (PID $pid).  Use: $(_bold 'bot restart')"
            [[ -f "$CF_URL_FILE" ]] && _print_url_box
            exit 0
        else
            info "Stale PID file removed."
            rm -f "$BOT_PID"
        fi
    fi

    check_and_install
    activate_venv

    # Abort early if no API key is configured — the server cannot prompt
    # for it when running in the background via nohup.
    if ! python3 "$ENGINE" config --check-key 2>/dev/null; then
        echo ""
        error "No API key set. Run  $(_bold 'bot config')  first, then  $(_bold 'bot start')  again."
    fi

    step "Starting bot server..."

    # Ensure Termux:API pkg is present (needed for wake lock)
    install_termux_api

    # Prevent Android from killing Termux while the bot is running
    if command -v termux-wake-lock &>/dev/null; then
        termux-wake-lock
        info "Wake lock acquired."
    else
        warn "termux-wake-lock not found – install Termux:API to keep bot alive."
        warn "Run: pkg install termux-api  (then install the Termux:API companion app)"
    fi

    if [[ -z "${START_DEV_SERVER:-}" ]]; then
        info "Mode: Gunicorn (Production - 4 Threads)"
        nohup "$VENV/bin/gunicorn" -w 2 --threads 4 --timeout 120 -b 0.0.0.0:${BOT_PORT:-5000} bot:app >"$BOT_LOG" 2>&1 &
    else
        info "Mode: Flask Dev Server"
        nohup python3 "$ENGINE" server >"$BOT_LOG" 2>&1 &
    fi
    echo $! > "$BOT_PID"

    sleep 2
    if kill -0 "$(cat "$BOT_PID")" 2>/dev/null; then
        info "Bot started (PID $(cat "$BOT_PID"))."
    else
        termux-wake-unlock 2>/dev/null || true
        rm -f "$BOT_PID"
        error "Bot failed to start – check logs with: bot logs"
    fi

    start_tunnel
}

stop_bot() {
    stop_tunnel

    if [[ ! -f "$BOT_PID" ]]; then
        info "Bot is not running."
        return 0
    fi

    local pid
    pid=$(cat "$BOT_PID")
    if kill "$pid" 2>/dev/null; then
        info "Bot stopped (PID $pid)."
    else
        warn "Process $pid not found – cleaning up."
    fi
    rm -f "$BOT_PID"

    # Release wake lock so Android can sleep normally again
    if command -v termux-wake-unlock &>/dev/null; then
        termux-wake-unlock
        info "Wake lock released."
    fi
}

# ─────────────────────────────────────────────────────────────────────────────
# Status
# ─────────────────────────────────────────────────────────────────────────────

status_bot() {
    echo ""
    echo "  $(_bold 'groq-bot status')"
    echo ""

    # Bot process
    if [[ -f "$BOT_PID" ]] && kill -0 "$(cat "$BOT_PID")" 2>/dev/null; then
        echo "  Bot     : $(_green 'RUNNING')  (PID $(cat "$BOT_PID"), port ${BOT_PORT:-5000})"
    else
        echo "  Bot     : $(_red 'STOPPED')"
        [[ -f "$BOT_PID" ]] && rm -f "$BOT_PID"
    fi

    # Tunnel
    if [[ -f "$CF_PID" ]] && kill -0 "$(cat "$CF_PID")" 2>/dev/null; then
        local url
        url=$(cat "$CF_URL_FILE" 2>/dev/null || echo "detecting…")
        local mode="quick"
        [[ -f "$CF_NAMED_FLAG" ]] && mode="named (permanent)"
        echo "  Tunnel  : $(_green 'RUNNING')  ($mode)"
        echo "  URL     : $(_cyan "$url")"
        echo "  API     : $(_cyan "$url/reply")"
    else
        echo "  Tunnel  : $(_red 'STOPPED')"
        [[ -f "$CF_PID" ]] && rm -f "$CF_PID"
    fi

    # Docs
    local doc_count
    doc_count=$(find "$DOCS" -maxdepth 1 -type f 2>/dev/null | wc -l | tr -d ' ')
    echo "  Docs    : $doc_count file(s) in $DOCS"
    echo ""
}

# ─────────────────────────────────────────────────────────────────────────────
# Misc commands
# ─────────────────────────────────────────────────────────────────────────────

logs_bot() {
    [[ -f "$BOT_LOG" ]] || error "No log file yet – start the bot first."
    tail -f "$BOT_LOG"
}

reindex_bot() {
    ensure_dirs; check_and_install; activate_venv
    python3 "$ENGINE" reindex
}

config_bot() {
    ensure_dirs; check_and_install; activate_venv
    python3 "$ENGINE" config
}

chat_bot() {
    ensure_dirs; check_and_install; activate_venv
    python3 "$ENGINE" chat
}

uninstall_bot() {
    echo ""
    read -rp "  Remove ALL bot files and the 'bot' command? [y/N] " confirm
    [[ "$confirm" =~ ^[Yy]$ ]] || { info "Aborted."; exit 0; }
    stop_bot 2>/dev/null || true
    rm -rf "$VENV" "$BASE"
    rm -f  "$GLOBAL"
    info "Bot completely removed."
}

# ─────────────────────────────────────────────────────────────────────────────
# Help
# ─────────────────────────────────────────────────────────────────────────────

usage() {
    echo ""
    echo "  $(_bold 'groq-bot')  –  Groq RAG WhatsApp bot manager"
    echo ""
    echo "  $(_bold 'Usage:')  bot <command>"
    echo ""
    echo "  $(_bold 'Core')"
    echo "    start          Start bot (Gunicorn - 4 Threads)"
    echo "    start-dev      Start bot (Flask Dev Server - Single Thread)"
    echo "    stop           Stop bot server + close tunnel"
    echo "    restart        Restart server + tunnel"
    echo "    status         Show running status and public URL"
    echo "    logs           Stream server logs  (Ctrl-C to quit)"
    echo ""
    echo "  $(_bold 'Tools')"
    echo "    chat           Interactive terminal chat (no server needed)"
    echo "    config         Edit API key, model, and settings"
    echo "    reindex        Rebuild document index from docs/"
    echo ""
    echo "  $(_bold 'Tunnel')"
    echo "    tunnel-setup   One-time setup for a permanent URL  (recommended)"
    echo ""
    echo "  $(_bold 'Other')"
    echo "    uninstall      Remove all bot files and the bot command"
    echo ""
    echo "  $(_bold 'Docs directory:')  $DOCS"
    echo "  Drop .txt files there, then run $(_bold 'bot reindex')."
    echo ""
}

# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

install_self

case "${1:-}" in
    start)        start_bot   ;;
    start-dev)    START_DEV_SERVER=1 start_bot ;;
    stop)         stop_bot    ;;
    restart)      stop_bot; sleep 1; start_bot ;;
    status)       status_bot  ;;
    logs)         logs_bot    ;;
    chat)         chat_bot    ;;
    config)       config_bot  ;;
    reindex)      reindex_bot ;;
    tunnel-setup) tunnel_setup ;;
    uninstall)    uninstall_bot ;;
    *)            usage ;;
esac
