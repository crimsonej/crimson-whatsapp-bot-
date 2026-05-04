# Groq Bot (Crimsonej) 🤖

A lightweight, RAG-powered WhatsApp auto-reply server built for Termux and Linux. Featuring real-time web search, user profiling, and multi-modal capabilities (image generation & media downloading).

## 🚀 Features

- **RAG (Retrieval-Augmented Generation):** Local knowledge retrieval using TF-IDF similarity.
- **Real-time Web Search:** Smart detection for queries needing live info (DuckDuckGo powered).
- **User Profiles:** Remembers names, facts, and preferences across sessions.
- **Media Commands:**
  - `/imagine <prompt>`: Generate images using Hugging Face (SDXL) or Pollinations AI.
  - `/song-audio <name/URL>`: Search and download YouTube audio.
  - `/song-video <name/URL>`: Search and download YouTube video.
- **Secure Tunneling:** Integrated Cloudflare Tunnel support for a permanent public URL.

## 🛠️ Setup & Installation

### Prerequisite

1.  **Groq API Key:** Get one at [console.groq.com](https://console.groq.com/).
2.  **Hugging Face API Key:** (Optional) [huggingface.co](https://huggingface.co/).
3.  **NVIDIA API Key:** (Optional, for Vision) [build.nvidia.com](https://build.nvidia.com/).

### Quick Install (Cross-Platform)

The cross-platform Python installer will automatically check for the bridge, install dependencies, and configure background processes (via PM2 or similar).

```bash
# Clone the repository
git clone <repo_url>
cd groq-bot

# Run the installer
python3 install.py
```
*Note: Make sure `python3` and `pip` are installed on your system.*

### Manual Configuration

If you prefer to manually configure your environment:
1.  Copy `config.json.example` to `config.json`.
2.  Create a `.env` file from the template:
    ```env
    GROQ_API_KEY=your_key_here
    HF_API_KEY=your_hf_key_here
    NVIDIA_API_KEY=your_nvidia_key_here
    BOT_PORT=5000
    ```
3.  Start the bot:
    ```bash
    bot start
    ```

## 🎮 Command Reference

| Command | Description |
| :--- | :--- |
| `bot start` | Start the Gunicorn production server in background. |
| `bot stop` | Stop the server, release wake-lock, and close tunnel. |
| `bot status` | Show PIDs, public URLs, and document count. |
| `bot chat` | Interactive terminal chat (great for testing). |
| `bot config` | Interactively edit settings (model, threshold, etc). |
| `bot reindex` | Rebuild the knowledge base from `docs/*.txt`. |
| `bot logs` | Tail the server logs in real-time. |

## 🛠️ Advanced Configuration

Edit `config.json` or use `bot config` to tune:
- **Multi-Model Setup (OpenClaw style):** You can define multiple `providers` (Groq, NVIDIA, HuggingFace) and a list of `models` to allow easy switching between a primary model and fallbacks.
- `relevance_threshold`: Higher (0.15+) makes the bot more selective; lower (0.05) makes it more talkative.
- `session_ttl`: How long the bot remembers conversation context.
- `active_model`: The ID of the currently active model (e.g. `llama-3.3-70b-versatile`).

## ❓ Troubleshooting

- **Image generation fails?** Ensure `HF_API_KEY` is set in `.env` for Hugging Face, or it will fallback to Pollinations.
- **Bot stays silent?** Check the `relevance_threshold`. If your query doesn't match any `docs/*.txt` files, it might stay silent.
- **Port already in use?** Change `BOT_PORT` in `.env` and restart.

## 📂 Project Structure

- `bot.py`: Main Flask application, session management, and Groq API logic.
- `realtime_search.py`: DuckDuckGo integration and "Smart Search" heuristic.
- `profiles.py`: Persistent user profiles (facts, names, preferences).
- `docs/`: Your personal knowledge base. Drop any `.txt` files here.
- `bot.sh`: The engine behind the `bot` command. Handles PIDs, tunnels, and venv.

## 🤝 Credits & Support

Created by **Crimson (Elijah)**. Optimized for performance and personality.
For bugs, please check `bot.log` or run `bot logs`.
