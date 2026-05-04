#!/usr/bin/env python3
import os
import sys
import subprocess
import shutil
import json

def _print_step(msg):
    print(f"\n\033[1m=== {msg} ===\033[0m")

def _print_info(msg):
    print(f"\033[0;32m[INFO]\033[0m {msg}")

def _print_warn(msg):
    print(f"\033[0;33m[WARN]\033[0m {msg}")

def _print_err(msg):
    print(f"\033[0;31m[ERR]\033[0m {msg}")

def check_bridge():
    _print_step("Checking for whatsapp-bridge")
    bridge_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "whatsapp-bridge")
    if not os.path.isdir(bridge_path):
        _print_warn("Could not find '../whatsapp-bridge' directory.")
        _print_warn("This bot requires the whatsapp-bridge to function properly.")
        _print_warn("Please make sure to clone/download the whatsapp-bridge repository next to this one.")
    else:
        _print_info("Found whatsapp-bridge directory.")

def setup_venv():
    _print_step("Setting up Python Virtual Environment")
    if not os.path.isdir("venv"):
        try:
            subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
            _print_info("Created venv.")
        except subprocess.CalledProcessError:
            _print_err("Failed to create venv. Ensure python3-venv is installed.")
            sys.exit(1)
    else:
        _print_info("Virtual environment already exists.")

def install_deps():
    _print_step("Installing Python Packages")
    pip_exe = os.path.join("venv", "bin", "pip") if os.name != "nt" else os.path.join("venv", "Scripts", "pip.exe")
    if not os.path.isfile(pip_exe):
        _print_err(f"Could not find pip at {pip_exe}")
        sys.exit(1)
    subprocess.run([pip_exe, "install", "--upgrade", "pip"], check=True)
    subprocess.run([pip_exe, "install", "-r", "requirements.txt"], check=True)
    _print_info("Dependencies installed.")

def setup_env():
    _print_step("Securing Environment")
    if not os.path.isfile(".env"):
        with open(".env", "w") as f:
            f.write("BOT_PORT=5000\n")
        _print_info("Created .env file.")
        if os.name != "nt":
            os.chmod(".env", 0o600)
    else:
        _print_info(".env file already exists.")

def setup_config():
    _print_step("Setting up Configuration")
    if not os.path.isfile("config.json"):
        if os.path.isfile("config.json.example"):
            shutil.copy("config.json.example", "config.json")
            _print_info("Copied config.json.example to config.json.")
        else:
            _print_warn("config.json.example not found. Creating a default config.json.")
            default_cfg = {
                "providers": {},
                "models": [{"id": "llama-3.3-70b-versatile", "provider": "groq", "tier": "primary"}],
                "active_model": "llama-3.3-70b-versatile"
            }
            with open("config.json", "w") as f:
                json.dump(default_cfg, f, indent=2)
    else:
        _print_info("config.json already exists.")

def setup_pm2():
    _print_step("Checking Background Process Setup (PM2)")
    if shutil.which("pm2"):
        _print_info("pm2 found! Generating ecosystem.config.js...")
        ecosystem = """module.exports = {
  apps : [{
    name   : "groq-bot",
    script : "bot.py",
    args   : "start",
    interpreter: "venv/bin/python",
    watch  : false,
    env: {
      "BOT_PORT": 5000
    }
  }]
}
"""
        if os.name == "nt":
            ecosystem = ecosystem.replace("venv/bin/python", "venv\\\\Scripts\\\\python.exe")
        with open("ecosystem.config.js", "w") as f:
            f.write(ecosystem)
        _print_info("You can start the bot using: pm2 start ecosystem.config.js")
    else:
        _print_warn("pm2 is not installed globally. To run the bot in the background across platforms, consider installing it: npm install -g pm2")

def main():
    print("🚀 Starting Groq Bot Setup...")
    check_bridge()
    setup_venv()
    install_deps()
    setup_env()
    setup_config()
    setup_pm2()
    print("\n✨ Installation Complete! ✨")
    print("Make sure to configure your API keys in config.json or .env")

if __name__ == "__main__":
    main()
