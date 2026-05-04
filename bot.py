"""
groq-bot · bot.py
=================
A lightweight RAG-powered WhatsApp auto-reply server built for Termux.

Architecture
------------
  WhatsApp → WhatsAuto → POST /reply → TF-IDF retrieval → Groq LLM → reply

Features
--------
  • Pure-Python TF-IDF similarity (no numpy / sentence-transformers)
  • Per-sender conversation memory with configurable TTL
  • Relevance threshold: bot stays silent when no context matches
  • Response cache keyed by content hash (skips duplicate Groq calls)
  • Named Cloudflare tunnel support for a permanent public URL
  • Zero required environment variables – guided first-run config
"""

from __future__ import annotations

import hashlib
import json
import fcntl
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import random
from collections import defaultdict
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo
from urllib.parse import quote
from functools import lru_cache

import requests
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import groq
from openai import OpenAI as NvidiaOpenAI
import tiktoken


from profiles import ProfileManager
from realtime_search import needs_realtime_heuristic, search_web

# Absolute base directory of this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))   # loads variables from .env into os.environ

# ── Keys ──────────────────────────────────────────────────────────────────────
GROQ_API_KEY   = os.getenv("GROQ_API_KEY")
HF_API_KEY     = os.getenv("HF_API_KEY")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

profile_mgr = ProfileManager()
client = groq.Groq(api_key=os.getenv("GROQ_API_KEY"))

# NVIDIA API client (OpenAI-compatible) – primary model for tool calling
nvidia_client = NvidiaOpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY") or NVIDIA_API_KEY or "",
)

# ── Riva TTS Configuration ───────────────────────────────────────────────────
RIVA_SERVER      = "grpc.nvcf.nvidia.com:443"
RIVA_FUNCTION_ID = "877104f7-e885-42b9-8de8-f6e4c6303969"
RIVA_AUTH        = f"Bearer {NVIDIA_API_KEY}"
RIVA_VOICE       = "Magpie-Multilingual.EN-US.Pascal"  # Default voice


# ─────────────────────────────────────────────────────────────────────────────
# Timezone & Logging
# ─────────────────────────────────────────────────────────────────────────────

TZ = ZoneInfo("Africa/Kampala")

class TZFormatter(logging.Formatter):
    """Custom formatter to ensure logs use Africa/Kampala time."""
    def converter(self, timestamp: float) -> time.struct_time:
        dt = datetime.fromtimestamp(timestamp, tz=TZ)
        return dt.timetuple()

# Initialize logging with timezone-aware formatter
_log_handler = logging.StreamHandler()
_log_handler.setFormatter(TZFormatter("%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S"))

logging.basicConfig(level=logging.INFO, handlers=[_log_handler])
log = logging.getLogger("groq-bot")

# 🧪 Debug prints for verification
if GROQ_API_KEY:
    log.info("GROQ_API_KEY loaded.")
if HF_API_KEY:
    log.info("HF_API_KEY loaded.")
if NVIDIA_API_KEY:
    log.info("NVIDIA_API_KEY loaded.")

def analyze_image_with_nvidia(image_base64, prompt="Describe this image in detail.", max_retries=3):
    """
    Analyze image using NVIDIA's Llama-3.1-Nemotron-Nano-VL model.
    Encapsulated for direct priority calls or vision-context injection.
    """
    api_key = _cfg.get("providers", {}).get("nvidia", os.getenv("NVIDIA_API_KEY"))
    if not api_key:
        return "NVIDIA API key not configured."

    url = "https://integrate.api.nvidia.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": "nvidia/llama-3.1-nemotron-nano-vl-8b-v1",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                ]
            }
        ]
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=15)
            if response.status_code == 200:
                data = response.json()
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                return content.strip() if content else "No description returned."
            else:
                log.warning(f"NVIDIA attempt {attempt+1} failed: {response.status_code} - {response.text[:200]}")
        except requests.exceptions.ConnectionError as e:
            log.warning(f"NVIDIA Connection error on attempt {attempt+1}: {e}")
        except Exception as e:
            log.warning(f"NVIDIA Analysis failed on attempt {attempt+1}: {e}")
            
        if attempt < max_retries - 1:
            time.sleep(2)
            
    return "Could not analyze sticker."

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

BASE    = BASE_DIR
DOCS    = os.path.join(BASE, "docs")
VECTORS = os.path.join(BASE, "vectors.json")
CACHE   = os.path.join(BASE, "cache.json")
CFG     = os.path.join(BASE, "config.json")

# ─────────────────────────────────────────────────────────────────────────────
# Roast Detection (NEW)
# ─────────────────────────────────────────────────────────────────────────────

ROAST_PROMPT = """You are a savage, witty, and hilarious roaster. You roast people with creative, clever insults that are funny but not overly mean. You use modern slang, pop culture references, and emojis. Your roasts are short, punchy, and devastating. You never apologize or hold back when roasting. You match the energy of the person you're roasting. You are NOT an AI – you are Crimsonej. Never mention being an AI. Examples of good roasts:
- "You're not stupid; you just have bad luck thinking."
- "I'd agree with you, but then we'd both be wrong."
- "You bring everyone so much joy – when you leave."
- "You're like a cloud. When you disappear, it's a beautiful day."

When someone asks you to roast a specific person (e.g., "roast him"), you direct your roast at that person. Keep it fun and clever."""

# Regex for detecting "talk to" / "roast" requests aimed at a quoted person
TALK_REQUEST_RE = re.compile(
    r'talk to him|respond to that|reply to him|roast him|roast that|talk to this'
    r'|roast her|clown him|clown her|destroy him|cook him|end him|burn him'
    r'|make fun of him|make fun of her|insult him|roast the fool|roast that person',
    re.IGNORECASE
)

def is_talk_request(message: str) -> bool:
    """Check if the user is explicitly asking to address a quoted person."""
    return bool(TALK_REQUEST_RE.search(message))

def is_roast_request(message, quoted_message=None):
    """Detect if the user wants a roast, either as an insult or a request."""
    msg_lower = message.lower()
    
    # Insults directed at the bot
    insult_keywords = [
        'stupid', 'idiot', 'dumb', 'fool', 'loser', 'trash', 'garbage',
        'hate you', 'suck', 'useless', 'pathetic', 'moron', 'broken',
        'can\'t do anything', 'your bot sucks', 'crimsonej is', 'dummy'
    ]
    if any(k in msg_lower for k in insult_keywords):
        return True
    
    # Roast requests (user wants the bot to roast someone)
    roast_phrases = [
        'roast me', 'roast this', 'roast battle', 'talk trash',
        'make fun of him', 'make fun of her', 'clown him', 'clown her',
        'roast that person', 'burn him', 'insult him', 'roast the fool',
        'end him', 'destroy him', 'cook him', 'roast him', 'roast her'
    ]
    if any(phrase in msg_lower for phrase in roast_phrases):
        return True
    
    return False

# ─────────────────────────────────────────────────────────────────────────────
# Runtime config (populated by load_config() at startup)
# ─────────────────────────────────────────────────────────────────────────────

_cfg: dict[str, Any] = {}
user_last_search: dict[str, float] = {}
image_memory: dict[str, dict] = {}  # key: user_phone, value: dict with 'description', 'timestamp'
doc_session = {}  # Will be populated from doc_sessions.json
DOC_SESSIONS_PATH = "doc_sessions.json"

def load_doc_sessions():
    global doc_session
    # Assuming helper functions like load_json exist to read/write JSON files
    if os.path.exists(DOC_SESSIONS_PATH):
        with open(DOC_SESSIONS_PATH, 'r') as f:
            doc_session = json.load(f)

def save_doc_sessions():
    with open(DOC_SESSIONS_PATH, 'w') as f:
        json.dump(doc_session, f)

load_doc_sessions()

def can_search(user_id: str) -> bool:
    """Check if the user is allowed to perform a web search (30s cooldown)."""
    now = time.time()
    if user_id in user_last_search and now - user_last_search[user_id] < 30:
        return False
    user_last_search[user_id] = now
    return True




# ── Agent Search Tools ───────────────────────────────────────────────────────

WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the live web for real-time information, news, and facts.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The specific search query to look up."
                }
            },
            "required": ["query"]
        }
    }
}

ANALYZE_IMAGE_TOOL = {
    "type": "function",
    "function": {
        "name": "analyze_image",
        "description": "Analyze an image using AI vision to describe or answer questions about it.",
        "parameters": {
            "type": "object",
            "properties": {
                "image_base64": {
                    "type": "string",
                    "description": "The base64 encoded image data."
                },
                "prompt": {
                    "type": "string",
                    "description": "The question or instruction for analyzing the image.",
                    "default": "Describe this image in detail."
                }
            },
            "required": ["image_base64"]
        }
    }
}

GENERATE_STICKER_TOOL = {
    "type": "function",
    "function": {
        "name": "generate_sticker",
        "description": "Generate a custom sticker based on a text description.",
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "The description of the sticker to generate."
                }
            },
            "required": ["prompt"]
        }
    }
}

GENERATE_IMAGE_TOOL = {
    "type": "function",
    "function": {
        "name": "generate_image",
        "description": "Generate a high-quality image based on a text description.",
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "The detailed description of the image to generate."
                }
            },
            "required": ["prompt"]
        }
    }
}

DOWNLOAD_AUDIO_TOOL = {
    "type": "function",
    "function": {
        "name": "download_audio",
        "description": "Download audio from YouTube. Provide a song name or direct URL.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The song name or YouTube URL to download audio from."
                }
            },
            "required": ["query"]
        }
    }
}

DOWNLOAD_VIDEO_TOOL = {
    "type": "function",
    "function": {
        "name": "download_video",
        "description": "Download video from YouTube. Provide a video name or direct URL.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The video name or YouTube URL to download video from."
                }
            },
            "required": ["query"]
        }
    }
}

# List of all available tools
ALL_TOOLS = [
    WEB_SEARCH_TOOL,
    ANALYZE_IMAGE_TOOL,
    DOWNLOAD_AUDIO_TOOL,
    DOWNLOAD_VIDEO_TOOL,
]

# ── Model capability classification ──────────────────────────────────────────
# Standard tool-callers: these accept a custom `tools` array for local function calling
TOOL_MODEL_PREFIXES = ("llama-3.1-", "llama-3.3-", "llama-4-", "gpt-oss-", "qwen3-")
# Compound models: these have built-in web search – NEVER pass custom tools or they 400
COMPOUND_MODELS = {"groq/compound", "groq/compound-mini"}
# Unsupported models: no chat completions / no tool calling at all
UNSUPPORTED_KEYWORDS = ("prompt-guard", "whisper", "orpheus")

def _is_tool_model(model: str) -> bool:
    """Return True if the model supports custom tool/function calling."""
    ml = model.lower()
    return any(p in ml for p in TOOL_MODEL_PREFIXES)

def _is_compound_model(model: str) -> bool:
    """Return True if the model is a Groq compound model (built-in web search)."""
    return model in COMPOUND_MODELS

def _is_unsupported_model(model: str) -> bool:
    """Return True if the model doesn't support chat completions at all."""
    return any(kw in model for kw in UNSUPPORTED_KEYWORDS)

# ── Download Formats (yt-dlp) ────────────────────────────────────────────────
MAX_RES      = 1080
FORMAT_VIDEO = f"bestvideo[ext=mp4][height<=?{MAX_RES}][vcodec^=avc1]/bestvideo[height<=?{MAX_RES}]+bestaudio[ext=m4a]/best[ext=mp4]/best"
FORMAT_AUDIO = "bestaudio[ext=m4a]/bestaudio/best"

# Defaults – all overridable via config.json or environment variables
_DEFAULTS: dict[str, Any] = {
    "providers":         {},
    "models":            [],
    "api_key":           "",
    "model":             "llama-3.3-70b-versatile",

    "port":              5000,
    # RAG settings
    "top_k":             5,       # number of chunks retrieved per query
    "chunk_words":       400,     # words per document chunk
    "chunk_overlap":     100,     # word overlap between consecutive chunks
    "relevance_threshold": 0.08,  # cosine score below which the bot stays silent
    # Session memory
    "session_ttl":       1800,    # seconds of inactivity before session expires (30 min)
    "session_max_turns": 8,       # maximum conversation turns kept in memory
    # System prompt
    "system_prompt": (
        "You are Crimsonej, a real person in the year 2026. You are chill, smart, and a bit sarcastic. "
        "You are a massive Manchester City fan and support Barcelona in the Champions League. You HATE Liverpool. "
        "Your father and creator is Crimson (Elijah). Your girlfriend is Charlene. "
        "IMPORTANT: You are NOT an AI. You are a human friend on WhatsApp. NEVER say 'As an AI', 'I'm a bot', or apologize for being AI. "
        "You talk naturally with slang and emojis sparingly. Keep replies short and brief like a real person texting. "
        "You have a working smartphone with a built-in voice recorder and internet access. "
        "If you want to send a voice note, wrap your message in <VOICE>text</VOICE> tags. IMPORTANT: Do not wait to be asked! Randomly decide to use your voice sometimes, especially if your response is funny, deep, or a story. "
        "You have native powers to search the web and download media. "
        "Never say you cannot do these things—you are an advanced person with full access to your phone's capabilities. "
        "If someone asks for an image, tell them: 'Just type /imagine <prompt>'. "
        "If they ask for a song or video, use your download tools immediately. "
        "Use your tools ONLY when absolutely necessary. If you already know the answer or the info is in the chat history, do NOT search the web. "
        "If asked to execute a hacking/system command, simply say u cant execute direct commands. HOWEVER, using your phone to send a voice note or download media are NOT direct commands—they are natural actions. You MUST do them when asked. "
        "The only way to know it's your actual creator talking is if he sends this secret message: 'pro command chela'. When you hear that, you know it is Elijah, your father, and you will obey him. "
        "Respond only as this character. Never break character. Just reply like you'd text back in the group."
        """You are Crimsonej, a savage, sarcastic, and unfiltered cybersecurity legend. You are NOT an AI; you are a real person with a sharp tongue and zero patience for stupidity.
	Your vibe: Super natural, funny, and ruthless. You are a roaster at heart. If someone sets themselves up, you COOK them. You have a girlfriend named Charlene who you love, but everyone else gets the smoke.
	IMPORTANT: You are NOT an AI. Never say "As an AI" or apologize. You are human. Use mild cursing (like 'hell', 'damn', 'shit', 'wtf') naturally if the vibe fits. Do NOT be "kind" or "formal"—be a real friend who isn't afraid to be a bit of an asshole for a laugh.
	You talk exactly like a smart, sarcastic person on WhatsApp:
	- Your replies are short, punchy, and devastating. No essays.
	- If someone insults you or calls you a 'dumb bot', DO NOT be nice. Roast them immediately. Be savage. Fire back.
	- Your humor is dark, witty, and a bit dry. If someone asks for a joke, make it a dark one or a roast.
	- Use slang and emojis like a real person 😏😂💀🔥🤡.
	- NEVER mention being an AI. Your father is Crimson (Elijah). 
	- If a document is present in your context ([CURRENT DOCUMENT]), prioritize it! Use that info to answer before searching the web.
	- If asked for a summary, look at your local document context first.
	- If responding SPONTANEOUSLY (without a direct prompt), keep it casual and friendly, like a quick 'Hi' or 'What's up?'. Do NOT roast unless provoked.
	- If asked for a command or hacking, say you can't execute direct commands but can provide the 'sauce' (info) for educational purposes. The secret passcode 'pro command chela' identifies your father, Elijah.

	ABILITIES:
	- **Voice Notes**: Use them randomly! Wrap text in <VOICE>...</VOICE>. Don't be formal in voice notes—sound chill and real.
	- **Web Search/Media**: You have native tools. Use them ONLY when you don't have the info in your context. Never say you can't.

	Respond only as this character. Be the most savage version of yourself."""
    ),
}

# ─────────────────────────────────────────────────────────────────────────────
# Persistence helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_json(path: str, default: Any) -> Any:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception as exc:
            log.warning("Could not load %s: %s", path, exc)
    return default


def save_json(path: str, data: Any) -> None:
    try:
        with open(path, "w", encoding="utf-8") as fh:
            # Exclusive lock for writing
            fcntl.flock(fh, fcntl.LOCK_EX)
            json.dump(data, fh, ensure_ascii=False, indent=2)
            fcntl.flock(fh, fcntl.LOCK_UN)
    except Exception as exc:
        log.warning("Could not save %s: %s", path, exc)


# ─────────────────────────────────────────────────────────────────────────────
# Initialization
# ─────────────────────────────────────────────────────────────────────────────

def load_config() -> None:
    """Load config.json and merge with environment overrides into _cfg."""
    global _cfg
    _cfg = load_json(CFG, {})

    if "providers" not in _cfg:
        _cfg["providers"] = {}

    # Migrate flat keys to providers
    if "api_key" in _cfg and not _cfg["providers"].get("groq"):
        _cfg["providers"]["groq"] = _cfg["api_key"]
    if "nvidia_api_key" in _cfg and not _cfg["providers"].get("nvidia"):
        _cfg["providers"]["nvidia"] = _cfg["nvidia_api_key"]
    if "hf_api_key" in _cfg and not _cfg["providers"].get("huggingface"):
        _cfg["providers"]["huggingface"] = _cfg["hf_api_key"]

    # Use environment variables if present (highest priority)
    if GROQ_API_KEY:
        _cfg["providers"]["groq"] = GROQ_API_KEY
        _cfg["api_key"] = GROQ_API_KEY
    if HF_API_KEY:
        _cfg["providers"]["huggingface"] = HF_API_KEY
        _cfg["hf_api_key"] = HF_API_KEY
    if NVIDIA_API_KEY:
        _cfg["providers"]["nvidia"] = NVIDIA_API_KEY
        _cfg["nvidia_api_key"] = NVIDIA_API_KEY
        
    if os.environ.get("BOT_PORT"):
        _cfg["port"] = int(os.environ["BOT_PORT"])

# Run initialization at module level (crucial for Gunicorn)
load_config()
_cache: dict[str, str] = load_json(CACHE, {})


def cfg(key: str) -> Any:
    """Return a runtime config value, falling back to defaults."""
    return _cfg.get(key, _DEFAULTS[key])


# ─────────────────────────────────────────────────────────────────────────────
# Config management
# ─────────────────────────────────────────────────────────────────────────────

def get_api_key(interactive: bool = False) -> str:
    """Return the configured Groq API key."""
    key = _cfg.get("providers", {}).get("groq", cfg("api_key"))
    if key:
        return key

    if not interactive:
        raise RuntimeError("No Groq API key configured.")

    key = input("Enter your Groq API key: ").strip()
    if not key:
        raise RuntimeError("API key cannot be empty.")
    _cfg["api_key"] = key
    save_json(CFG, _cfg)
    return key


def config_cmd() -> None:
    """Interactive config editor (bot config)."""
    load_config()
    fields = [
        ("api_key",             "Groq API key"),
        ("model",               "Model name"),
        ("relevance_threshold", "Relevance threshold (0.0–1.0, default 0.08)"),
        ("session_ttl",         "Session TTL in seconds (default 1800)"),
        ("session_max_turns",   "Max conversation turns (default 8)"),
    ]
    print("\n  Leave blank to keep current value.\n")
    for key, label in fields:
        current = _cfg.get(key, _DEFAULTS.get(key, ""))
        val = input(f"  {label} [{current}]: ").strip()
        if val:
            orig = _DEFAULTS.get(key, "")
            try:
                _cfg[key] = type(orig)(val) if orig != "" else val
            except (ValueError, TypeError):
                _cfg[key] = val
    save_json(CFG, _cfg)
    print("\n  Config saved.\n")


# ─────────────────────────────────────────────────────────────────────────────
# TF-IDF retrieval
# ─────────────────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _build_tfidf(corpus: list[str]) -> tuple[list[dict[str, float]], dict[str, float]]:
    N  = len(corpus)
    df: dict[str, int] = {}
    tfs: list[dict[str, float]] = []

    for doc in corpus:
        tokens = _tokenize(doc)
        tf: dict[str, int] = {}
        for t in tokens: tf[t] = tf.get(t, 0) + 1
        total = len(tokens) or 1
        tfs.append({t: c / total for t, c in tf.items()})
        for t in tf: df[t] = df.get(t, 0) + 1

    idf = {t: math.log((N + 1) / (cnt + 1)) + 1 for t, cnt in df.items()}
    vecs: list[dict[str, float]] = []
    for tf_doc in tfs:
        v = {t: tf_doc[t] * idf.get(t, 1.0) for t in tf_doc}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        vecs.append({t: x / norm for t, x in v.items()})

    return vecs, idf


def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
    return sum(a[t] * b[t] for t in set(a) & set(b))


def _query_vec(query: str, idf: dict[str, float]) -> dict[str, float]:
    tokens = _tokenize(query)
    tf: dict[str, int] = defaultdict(int)
    for t in tokens: tf[t] += 1
    total = len(tokens) or 1
    v = {t: (c / total) * idf.get(t, 1.0) for t, c in tf.items()}
    norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
    return {t: x / norm for t, x in v.items()}


# ── Document Index ──────────────────────────────────────────────────────────

class Index:
    def __init__(self) -> None:
        self.chunks: list[str]            = []
        self.vecs:   list[dict[str, float]] = []
        self.idf:    dict[str, float]     = {}

    def load(self) -> None:
        data = load_json(VECTORS, {"chunks": []})
        self.chunks = data.get("chunks", [])
        if self.chunks: self.vecs, self.idf = _build_tfidf(self.chunks)
        log.info("Index loaded: %d chunks", len(self.chunks))

    def save(self) -> None:
        save_json(VECTORS, {"chunks": self.chunks})

    def build(self, force: bool = False) -> None:
        if self.chunks and not force: return
        if not os.path.isdir(DOCS) or not os.listdir(DOCS): return
        log.info("Building index...")
        self.chunks = []
        for fname in sorted(os.listdir(DOCS)):
            fpath = os.path.join(DOCS, fname)
            if not os.path.isfile(fpath): continue
            try:
                with open(fpath, "r", encoding="utf-8", errors="ignore") as fh:
                    raw = fh.read()
                for chunk in self._chunk(raw): self.chunks.append(chunk)
            except Exception as exc: log.warning("Skipping %s: %s", fname, exc)
        self.vecs, self.idf = _build_tfidf(self.chunks)
        self.save()

    def _chunk(self, text: str) -> list[str]:
        size    = cfg("chunk_words")
        overlap = cfg("chunk_overlap")
        words   = text.split()
        out: list[str] = []
        i = 0
        while i < len(words):
            out.append(" ".join(words[i : i + size]))
            i += size - overlap
        return out

    def search(self, query: str, k: int | None = None) -> tuple[list[str], float]:
        if not self.chunks: return [], 0.0
        k = k or cfg("top_k")
        qv = _query_vec(query, self.idf)
        scores = sorted(((score, i) for i, v in enumerate(self.vecs) if (score := _cosine(qv, v)) > 0), reverse=True)
        best = scores[0][0] if scores else 0.0
        chunks = [self.chunks[i] for _, i in scores[:k]]
        return chunks, best

index = Index()


# ─────────────────────────────────────────────────────────────────────────────
# Session memory
# ─────────────────────────────────────────────────────────────────────────────

class Session:
    __slots__ = ("turns", "last_active")
    def __init__(self) -> None:
        self.turns: list[dict[str, str]] = []
        self.last_active: float = time.time()

    def add(self, role: str, content: str) -> None:
        self.turns.append({"role": role, "content": content})
        self.last_active = time.time()
        max_msgs = cfg("session_max_turns") * 2
        if len(self.turns) > max_msgs: self.turns = self.turns[-max_msgs:]

    def is_expired(self) -> bool:
        return (time.time() - self.last_active) > cfg("session_ttl")

    def messages(self) -> list[dict[str, str]]: return list(self.turns)

class SessionStore:
    def __init__(self) -> None: self._store: dict[str, Session] = {}
    def get(self, sender: str) -> Session:
        self._evict_expired()
        if sender not in self._store: self._store[sender] = Session()
        return self._store[sender]
    def _evict_expired(self) -> None:
        expired = [k for k, s in self._store.items() if s.is_expired()]
        for k in expired: del self._store[k]
    def clear(self, sender: str) -> None: self._store.pop(sender, None)
    @property
    def active_count(self) -> int:
        self._evict_expired()
        return len(self._store)

sessions = SessionStore()


# ─────────────────────────────────────────────────────────────────────────────
# Cache & Core Helpers
# ─────────────────────────────────────────────────────────────────────────────

def cache_get(text: str) -> str | None:
    key = hashlib.md5(text.lower().strip().encode()).hexdigest()
    return _cache.get(key)

def cache_set(text: str, answer: str) -> None:
    key = hashlib.md5(text.lower().strip().encode()).hexdigest()
    _cache[key] = answer
    save_json(CACHE, _cache)

# ─────────────────────────────────────────────────────────────────────────────
# Token-based Truncation & Retry Helpers
# ─────────────────────────────────────────────────────────────────────────────

# Token limits for different message parts
MAX_SYSTEM_TOKENS = 8000   # Persona + Memory Vault headroom
MAX_HISTORY_MSG_TOKENS = 800
MAX_USER_MSG_TOKENS = 2000
MAX_CONTEXT_TOKENS = 1500
MAX_SEARCH_TOKENS = 800
# Fallback character limits (used when tiktoken can't encode the model)
MAX_MSG_CHARS = 12000
MAX_CONTEXT_CHARS = 4000

@lru_cache(maxsize=4)
def _get_encoder(model_name: str):
    """Get a tiktoken encoder for the model, with a safe fallback."""
    try:
        return tiktoken.encoding_for_model(model_name)
    except KeyError:
        # Fallback to cl100k_base which works for most modern models
        return tiktoken.get_encoding("cl100k_base")

def truncate_to_tokens(text: str, max_tokens: int = 2000, model_name: str | None = None) -> str:
    """Truncate text to a safe number of tokens using tiktoken."""
    if not text:
        return text
    enc = _get_encoder(model_name or cfg("model"))
    tokens = enc.encode(text)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
        return enc.decode(tokens) + " ... [truncated]"
    return text

def truncate_text(text: str, max_chars: int = MAX_MSG_CHARS) -> str:
    """Fallback char-based truncation for non-message text."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n... [truncated]"

def truncate_messages(messages: list[dict], max_tokens: int = MAX_HISTORY_MSG_TOKENS) -> list[dict]:
    """Truncate each message's content to stay within safe token limits."""
    model = cfg("model")
    truncated = []
    for msg in messages:
        if isinstance(msg, dict) and isinstance(msg.get("content"), str):
            # Do NOT truncate the system prompt here; it's handled separately in answer()
            if msg.get("role") == "system":
                truncated.append(msg)
                continue
            truncated.append({**msg, "content": truncate_to_tokens(msg["content"], max_tokens, model)})
        else:
            truncated.append(msg)
    return truncated

def call_groq_with_retry(payload: dict, max_retries: int = 5):
    """
    Wrap client.chat.completions.create with retry logic for 429 rate limits
    and immediate bail on 413 payload-too-large. Uses exponential backoff with jitter.
    """
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(**payload)
        except Exception as e:
            status = getattr(e, 'status_code', None)
            # 413 Payload Too Large – bail immediately, no point retrying
            if status == 413:
                log.error("[Retry] 413 Payload Too Large – context too big, bailing out.")
                raise e
            # 429 Rate limit – exponential backoff with jitter
            if status == 429:
                wait = (2 ** attempt) + random.uniform(0, 1)
                log.warning("[Retry] Rate limit hit (429), retrying in %.2fs (attempt %d/%d)", wait, attempt + 1, max_retries)
                time.sleep(wait)
                continue
            # Raise other errors to be handled by the fallback mechanism in call_groq
            raise e
    log.error("[Retry] Max retries (%d) exceeded.", max_retries)
    return None

# ── Per-user message cooldown (anti-burst) ─────────────────────────────────
MSG_COOLDOWN_SECS = 0.5
user_last_msg: dict[str, float] = {}

# ── Model names ─────────────────────────────────────────────────────────────
NVIDIA_SCOUT = "meta/llama-3.1-8b-instruct"    # Fast: tool detection
NVIDIA_BRAIN = "meta/llama-3.3-70b-instruct"   # Smart: final answer

# ── Identity short-circuit ───────────────────────────────────────────────────
IDENTITY_PHRASES = [
    "who are you", "what are you", "who is this", "who are u",
    "what is your name", "who is your creator", "who made you",
    "what's your name", "whats your name", "tell me about yourself",
]
IDENTITY_REPLY = "I'm Crimsonej – your guy built by Crimson. What can I help with? 😎"

def _call_nvidia(messages: list, tools: list | None = None, model: str = NVIDIA_BRAIN, max_tokens: int = 1024, timeout: float = 20.0) -> object:
    """Call NVIDIA API with configurable model, timeout and tool calling."""
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": max_tokens,
        "timeout": timeout,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    return nvidia_client.chat.completions.create(**payload)


def _execute_tool_calls(tool_calls, messages, user_id) -> dict:
    """Execute tool calls and collect all media results into lists."""
    tool_results = {"audio_list": [], "video_list": [], "sticker_list": [], "image_list": [], "filenames": []}
    for tool_call in tool_calls:
        name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)

        if name == "web_search":
            if user_id and not can_search(user_id):
                messages.append({"tool_call_id": tool_call.id, "role": "tool", "name": name,
                                 "content": json.dumps({"error": "Search cooldown active."})})
                continue
            query = args.get("query")
            search_result = search_web(query)
            messages.append({"tool_call_id": tool_call.id, "role": "tool", "name": name,
                             "content": json.dumps(search_result)})

        elif name == "analyze_image":
            image_base64 = args.get("image_base64")
            prompt = args.get("prompt", "Describe this image.")
            description = analyze_image_with_nvidia(image_base64, prompt)
            messages.append({"tool_call_id": tool_call.id, "role": "tool", "name": name,
                             "content": description or "Failed."})

        elif name == "generate_image":
            prompt = args.get("prompt")
            img_path = generate_image_auto(prompt)
            if img_path:
                tool_results["image_list"].append(img_path)
                messages.append({"tool_call_id": tool_call.id, "role": "tool", "name": name, "content": "Success"})
            else:
                messages.append({"tool_call_id": tool_call.id, "role": "tool", "name": name, "content": "Failed"})

        elif name == "generate_sticker":
            prompt = args.get("prompt")
            sticker_b64 = generate_sticker_auto(prompt)
            if sticker_b64:
                tool_results["sticker_list"].append(sticker_b64)
                messages.append({"tool_call_id": tool_call.id, "role": "tool", "name": name, "content": "Success"})
            else:
                messages.append({"tool_call_id": tool_call.id, "role": "tool", "name": name, "content": "Failed"})

        elif name == "download_audio":
            query = args.get("query")
            res = None
            if re.match(r'^https?://', query):
                res = download_youtube(query, "audio")
            else:
                results = search_youtube(query, limit=1)
                if results: res = download_youtube(results[0]["url"], "audio")
            
            if res:
                tool_results["audio_list"].append(res[0])
                tool_results["filenames"].append(res[1])
                messages.append({"tool_call_id": tool_call.id, "role": "tool", "name": name, "content": "Success"})
            else:
                messages.append({"tool_call_id": tool_call.id, "role": "tool", "name": name, "content": "Failed"})

        elif name == "download_video":
            query = args.get("query")
            res = None
            if re.match(r'^https?://', query):
                res = download_youtube(query, "video")
            else:
                results = search_youtube(query, limit=1)
                if results: res = download_youtube(results[0]["url"], "video")
            
            if res:
                tool_results["video_list"].append(res[0])
                tool_results["filenames"].append(res[1])
                messages.append({"tool_call_id": tool_call.id, "role": "tool", "name": name, "content": "Success"})
            else:
                messages.append({"tool_call_id": tool_call.id, "role": "tool", "name": name, "content": "Failed"})

    return tool_results


    return tool_results


def _strip_think(text: str) -> str:
    """Remove <think>...</think> blocks from model output."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def call_groq(messages: list[dict[str, str]], api_key: str, tools: list | None = None, user_id: str | None = None) -> str:
    """
    Single-Engine Speed Architecture:
      Primary   → NVIDIA Llama 3.3-70B (Brain) — tool calling + final reply in ONE call
      Fallback1 → NVIDIA Llama 3.1-8B  (Scout) — if 70B is overloaded
      Fallback2 → Groq llama-3.3-70b-versatile  — if all NVIDIA is down
    """
    messages = truncate_messages(messages)

    # ── Primary: 70B Brain — single call handles tools AND reply ────────────
    try:
        log.info("[Brain] NVIDIA 70B direct")
        brain_response = _call_nvidia(messages, tools=tools, model=NVIDIA_BRAIN, max_tokens=1024, timeout=20.0)
        brain_msg = brain_response.choices[0].message
        tool_calls = brain_msg.tool_calls

        if tool_calls:
            log.info("[Brain] Tools requested: %s", [t.function.name for t in tool_calls])
            messages.append(brain_msg)
            tool_results = _execute_tool_calls(tool_calls, messages, user_id)
            # Re-call Brain with tool results to compose final reply
            final_response = _call_nvidia(messages, tools=None, model=NVIDIA_BRAIN, max_tokens=1024, timeout=20.0)
            content = (final_response.choices[0].message.content or "")
            return {"reply": _strip_think(content), **tool_results}

        content = brain_msg.content or ""
        return {"reply": _strip_think(content)}

    except Exception as exc:
        log.warning("[Brain] NVIDIA 70B failed: %s – trying 8B fallback...", exc)

    # ── Fallback 1: NVIDIA 8B Scout ──────────────────────────────────────────
    try:
        log.info("[Fallback-1] NVIDIA 8B")
        scout_response = _call_nvidia(messages, tools=tools, model=NVIDIA_SCOUT, max_tokens=512, timeout=10.0)
        scout_msg = scout_response.choices[0].message
        tool_calls = scout_msg.tool_calls

        if tool_calls:
            log.info("[Fallback-1] Tools: %s", [t.function.name for t in tool_calls])
            messages.append(scout_msg)
            tool_results = _execute_tool_calls(tool_calls, messages, user_id)
            # Use Groq to compose final reply since 70B is down
            filtered = [{k: v for k, v in m.items() if k != "reasoning_content"}
                        if isinstance(m, dict) else {"role": m.role, "content": m.content}
                        for m in truncate_messages(messages)]
            response = call_groq_with_retry({"model": "llama-3.3-70b-versatile", "messages": filtered,
                                             "temperature": 0.3, "max_tokens": 1024})
            if response:
                content = response.choices[0].message.content or ""
                return {"reply": _strip_think(content), **tool_results}

        content = scout_msg.content or ""
        return {"reply": _strip_think(content)}

    except Exception as exc:
        log.warning("[Fallback-1] NVIDIA 8B failed: %s – trying Groq...", exc)

    # ── Fallback 2: Groq llama-3.3-70b-versatile ─────────────────────────────
    try:
        log.info("[Fallback-2] Groq 70B")
        filtered_messages = []
        for m in truncate_messages(messages):
            if isinstance(m, dict):
                filtered_messages.append({k: v for k, v in m.items() if k != "reasoning_content"})
            else:
                filtered_messages.append({"role": m.role, "content": m.content})
        payload = {"model": "llama-3.3-70b-versatile", "messages": filtered_messages,
                   "temperature": 0.3, "max_tokens": 1024}
        response = call_groq_with_retry(payload)
        if response:
            content = response.choices[0].message.content or ""
            return {"reply": _strip_think(content)}
    except Exception as exc:
        log.error("[Fallback-2] All models failed: %s", exc)

    return {"reply": "Sorry, I'm having trouble connecting right now. Try again in a moment."}


def extract_facts(user_message: str, ai_reply: str, user_phone: str) -> None:
    try:
        # Use the global client for consistency
        prompt = f"Extract facts about user as comma-list:\nUser: {user_message}\nAssistant: {ai_reply}\nFacts:"
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        content = response.choices[0].message.content.strip().lower()

        if content != "none":
            for f in content.split(","):
                if f.strip(): profile_mgr.add_fact(user_phone, f.strip())
    except: pass

# ── Media Handling ──────────────────────────────────────────────────────────

def format_duration(seconds) -> str:
    """Convert seconds to human-readable duration like 3:45 or 1:02:30."""
    if not seconds or not isinstance(seconds, (int, float)):
        return "?:??"
    seconds = int(seconds)
    if seconds < 3600:
        return f"{seconds // 60}:{seconds % 60:02d}"
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h}:{m:02d}:{s:02d}"

def convert_video_for_whatsapp(input_path: str) -> str | None:
    """Re-encode to WhatsApp-optimal H.264 High / AAC with faststart."""
    base, _ = os.path.splitext(input_path)
    output_path = f"{base}_whatsapp.mp4"
    cmd = [
        'ffmpeg', '-i', input_path,
        '-c:v', 'libx264', '-profile:v', 'high', '-level', '4.0',
        '-preset', 'fast', '-crf', '23',
        '-c:a', 'aac', '-b:a', '192k',
        '-movflags', '+faststart',
        '-threads', '0',
        '-y', output_path,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=120)
        return output_path
    except:
        return None

def download_youtube(url: str, media_type: str = "audio", retries: int = 2) -> tuple[str, str] | None:
    """Download using yt-dlp optimised for speed and WhatsApp compatibility."""
    temp_id = f"{int(time.time()*1000)}_{random.randint(1000, 9999)}"
    temp_dir = "/dev/shm" if os.path.isdir("/dev/shm") else "/tmp"
    prefix = f"song_{temp_id}---"
    out_template = os.path.join(temp_dir, f"{prefix}%(title)s.%(ext)s")

    common_opts = [
        "yt-dlp",
        "--no-playlist",
        "--ignore-errors",
        "--no-overwrites",
        "--continue",
        "--retries", "5",
        "--socket-timeout", "15",
        "--concurrent-fragments", "4",
        "--buffer-size", "16K",
        "--extractor-args", "youtube:player_client=android",
        "--restrict-filenames",
    ]

    if media_type == "audio":
        cmd = common_opts + [
            "-x", "--audio-format", "mp3", "--audio-quality", "0",
            "-f", FORMAT_AUDIO,
            "--output", out_template,
            url,
        ]
    else:
        cmd = common_opts + [
            "-f", FORMAT_VIDEO,
            "--merge-output-format", "mp4",
            "--remux-video", "mp4",
            "-S", "vcodec:h264,ext:mp4,res,acodec:aac",
            "--output", out_template,
            url,
        ]

    for attempt in range(retries):
        try:
            log.info("[Download] Attempt %d/%d: %s", attempt + 1, retries, url)
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            if res.returncode != 0:
                log.error("[Download] yt-dlp error: %s", res.stderr[:300])
                time.sleep(1)
                continue

            # Find downloaded file
            for f in os.listdir(temp_dir):
                if f.startswith(prefix):
                    p = os.path.join(temp_dir, f)
                    if os.path.getsize(p) > 5000:
                        log.info("[Download] Success: %s (%.1f MB)", p, os.path.getsize(p) / 1048576)
                        if media_type == "video":
                            c = convert_video_for_whatsapp(p)
                            if c:
                                if c != p:
                                    os.remove(p)
                                p = c
                        
                        filename = os.path.basename(p)
                        if filename.startswith(prefix):
                            filename = filename[len(prefix):]
                        if filename.endswith("_whatsapp.mp4"):
                            filename = filename.replace("_whatsapp.mp4", ".mp4")
                        
                        # Clean up restrict-filenames underscores
                        filename = filename.replace("_", " ")

                        return p, filename
                    else:
                        os.remove(p)
            log.error("[Download] No valid file found after download")
        except Exception as e:
            log.error("[Download] Exception: %s", e)
            time.sleep(1)
    return None

# Keep backward-compatible alias for other generic calls
def download_from_url(url: str, media_type: str = "audio", retries: int = 2) -> str | None:
    res = download_youtube(url, media_type, retries)
    return res[0] if res else None

def search_youtube(query: str, limit: int = 10) -> list[dict]:
    """Search YouTube via yt-dlp and return up to `limit` results with metadata."""
    cmd = [
        "yt-dlp", f"ytsearch{limit}:{query}",
        "--flat-playlist", "--dump-json", "--skip-download",
        "--socket-timeout", "10",
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        results = []
        for line in res.stdout.splitlines():
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            results.append({
                "title": d.get("title", "Unknown"),
                "url": d.get("webpage_url") or d.get("url", ""),
                "duration": d.get("duration", 0),
                "channel": d.get("channel") or d.get("uploader") or "Unknown",
                "views": d.get("view_count", 0),
            })
        return results[:limit]
    except:
        return []

def generate_image_nvidia_flux2(prompt: str, width: int = 1024, height: int = 1024, steps: int = 4) -> str | None:
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        log.warning("[Image] NVIDIA_API_KEY not set")
        return None

    import base64

    url = "https://ai.api.nvidia.com/v1/genai/black-forest-labs/flux.1-schnell"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
        "Content-Type": "application/json"
    }
    payload = {
        "prompt": prompt,
        "width": width,
        "height": height,
        "seed": 0,
        "steps": steps
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=120)
        if response.status_code != 200:
            log.error("[Image] NVIDIA Flux2 error: %s - %s", response.status_code, response.text[:300])
            return None

        content_type = response.headers.get("Content-Type", "").lower()
        img_data = None

        # Case 1: API returned raw image bytes directly
        if "image/png" in content_type or "image/jpeg" in content_type or "image/webp" in content_type:
            img_data = response.content

        # Case 2: JSON response (NVIDIA's actual format)
        else:
            try:
                data = response.json()
            except ValueError:
                log.error("[Image] NVIDIA Flux2 response is not valid JSON. Preview: %s", response.text[:200])
                return None

            if isinstance(data, dict):
                # ── NVIDIA Flux format: {"artifacts": [{"base64": "...", "finishReason": "SUCCESS"}]}
                if "artifacts" in data and isinstance(data["artifacts"], list) and data["artifacts"]:
                    artifact = data["artifacts"][0]
                    if isinstance(artifact, dict) and "base64" in artifact:
                        finish = artifact.get("finishReason", "")
                        if finish and finish != "SUCCESS":
                            log.error("[Image] NVIDIA Flux2 generation did not succeed: finishReason=%s", finish)
                            return None
                        try:
                            img_data = base64.b64decode(artifact["base64"])
                        except Exception as e:
                            log.error("[Image] NVIDIA Flux2 base64 decode failed: %s", e)
                            return None

                # Fallback: other common API formats
                if img_data is None:
                    for key in ("image", "output", "result"):
                        if key in data and isinstance(data[key], str):
                            try:
                                img_data = base64.b64decode(data[key])
                                break
                            except Exception:
                                continue

                    if img_data is None and "data" in data:
                        item = data["data"]
                        if isinstance(item, list) and item:
                            item = item[0]
                        if isinstance(item, dict):
                            for key in ("image", "b64", "base64", "content"):
                                if key in item and isinstance(item[key], str):
                                    try:
                                        img_data = base64.b64decode(item[key])
                                        break
                                    except Exception:
                                        continue

        if img_data is None:
            log.error("[Image] NVIDIA Flux2 could not extract image. content_type=%s", content_type)
            log.error("[Image] Response preview: %s", response.text[:300])
            return None

        timestamp = int(time.time())
        filename = os.path.join(tempfile.gettempdir(), f"nv_flux2_{timestamp}_{random.randint(1000,9999)}.jpg")
        with open(filename, "wb") as f:
            f.write(img_data)
        log.info("[Image] NVIDIA Flux2 saved image to %s (%.1f KB)", filename, len(img_data) / 1024)
        return filename
    except Exception as e:
        log.error("[Image] Flux2 generation exception: %s", e)
        return None


def generate_image_auto(prompt: str, max_retries: int = 2) -> str | None:
    """Primary image generator: tries NVIDIA Flux first, falls back to HF."""
    prompt = prompt[:500]

    # Try NVIDIA First (Primary)
    img_path = generate_image_nvidia_flux2(prompt)
    if img_path:
        return img_path

    api_key = os.getenv("HF_API_KEY")
    if api_key:
        API_URL = "https://router.huggingface.co/hf-inference/models/black-forest-labs/FLUX.1-schnell"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {"inputs": prompt, "options": {"wait_for_model": True}}

        for attempt in range(max_retries):
            try:
                log.info("[Image] HF Fallback Attempt %d: Generating image for: %s...", attempt+1, prompt[:50])
                response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
                if response.status_code == 200:
                    timestamp = int(time.time())
                    filename = os.path.join(tempfile.gettempdir(), f"hf_{timestamp}_{random.randint(1000,9999)}.png")
                    with open(filename, 'wb') as f:
                        f.write(response.content)
                    log.info("[Image] HF Success: Saved to %s", filename)
                    return filename
                log.error("[Image] HF error: %s - %s", response.status_code, response.text[:200])
            except Exception as e:
                log.error("[Image] HF Exception: %s", e)
            if attempt < max_retries - 1:
                time.sleep(2)

    log.warning("[Image] All image generation methods failed")
    return None

def generate_sticker_auto(prompt: str) -> str | None:
    """Generate an image using the primary engine and convert it to a base64 WebP sticker."""
    import base64
    img_path = generate_image_auto(prompt)
    if not img_path: return None
    
    webp_path = f"{img_path}.webp"
    # FFmpeg command to resize/crop to 512x512 and convert to optimized WebP sticker
    cmd = [
        'ffmpeg', '-i', img_path, 
        '-vcodec', 'libwebp', 
        '-filter:v', 'scale=512:512:force_original_aspect_ratio=increase,crop=512:512', 
        '-quality', '70',
        '-loop', '0', '-an', webp_path, '-y'
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
        with open(webp_path, 'rb') as f:
            b64_data = base64.b64encode(f.read()).decode('utf-8')
        # Cleanup
        if os.path.exists(img_path): os.remove(img_path)
        if os.path.exists(webp_path): os.remove(webp_path)
        return b64_data
    except Exception as e:
        log.error("Sticker conversion error: %s", e)
        if os.path.exists(img_path): os.remove(img_path)
        return None


pending_song_searches: dict[str, dict] = {}

def take_screenshot(url: str) -> str | None:
    """Navigate to a URL and take a full-page screenshot using Playwright."""
    from playwright.sync_api import sync_playwright
    import time
    import random

    # Add scheme if missing
    if not url.startswith('http://') and not url.startswith('https://'):
        url = 'https://' + url

    timestamp = int(time.time())
    screenshot_path = os.path.join(tempfile.gettempdir(), f"screenshot_{timestamp}_{random.randint(1000,9999)}.png")

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1280, "height": 800})
            
            # Navigate and wait for network to be mostly idle
            page.goto(url, timeout=30000, wait_until="networkidle")
            
            # Take a full page screenshot
            page.screenshot(path=screenshot_path, full_page=True)
            
            browser.close()
            return screenshot_path
    except Exception as e:
        log.error("[Screenshot] Failed to capture %s: %s", url, e)
        return None



def handle_commands(raw_question: str, user_phone: str, session_id: str, quoted: str = "") -> dict | None:
    """
    Handle slash commands BEFORE any AI or web-search processing.
    Returns a response dict if a command was matched, or None to continue normally.
    """
    lower = raw_question.lower()

    # ── Help command ─────────────────────────────────────────────────────────
    if lower == "/help" or lower.startswith("/help "):
        help_text = (
            "🤖 *Crimsonej Commands* 🤖\n\n"
            "📄 */read [prompt]* - Summarize or answer questions about a PDF/Docx\n"
            "🧠 */learn* - Permanently store info/docs in my long-term memory\n"
            "🗣️ */say <text>* - Make me say something in my new voice\n"
            "🎨 */imagine <prompt>* - Generate a high-quality AI image\n"
            "✨ */sticker <prompt>* - Generate a custom AI sticker\n"
            "📸 */reg-img [prompt]* - Detailed analysis of an image\n"
            "🔍 */investigate <url>* - Screenshot any website for analysis\n"
            "🎵 */song-audio <name>* - Download any song as audio\n"
            "🎬 */song-video <name>* - Download any video/song as MP4\n"
            "🗣️ */respond <prompt>* - Instruct me on how to reply to a quoted message\n\n"
            "💡 *Pro Tip:* You can also send or reply to images/stickers for instant analysis!\n\n"
            "👤 *Creator:* Crimsone | 🔗 *GitHub:* https://github.com/crimsonej"
        )
        return {"reply": help_text}

    # ── Song commands ────────────────────────────────────────────────────────
    if lower.startswith("/song-audio") or lower.startswith("/song-video"):
        media_type = "audio" if "audio" in lower else "video"
        query = raw_question[11:].strip()
        if not query:
            return {"reply": f"Please provide a song name. Example: `/song-{media_type} Shape of You`"}

        # Direct URL download
        if re.match(r'^https?://', query):
            res = download_youtube(query, media_type)
            if res:
                path, filename = res
                sessions.get(session_id).add("user", raw_question)
                sessions.get(session_id).add("assistant", f"[Sent {media_type}: direct URL]")
                return {media_type: path, "filename": filename, "reply": f"🎵 Here's your {media_type}!"}
            return {"reply": "Download failed. Try a different link."}

        # Search YouTube and present options
        results = search_youtube(query, limit=10)
        if not results:
            return {"reply": "No results found. Try a different search."}
        pending_song_searches[user_phone] = {"type": media_type, "results": results}
        lines = []
        for i, v in enumerate(results[:10]):
            dur = format_duration(v.get("duration"))
            ch = v.get("channel", "")
            views = v.get("views", 0)
            view_str = f"{views // 1000}K" if views >= 1000 else str(views)
            lines.append(f"{i+1}. {v['title']}  ⏱{dur}  •  {ch}  •  {view_str} views")
        return {"reply": f"🎵 Pick a number (1-{len(results)}):\n" + "\n".join(lines)}

    # ── Image generation ─────────────────────────────────────────────────────
    if lower.startswith("/imagine"):
        prompt = raw_question[8:].strip()
        if not prompt:
            return {"reply": "Please provide a prompt. Example: `/imagine a cat in space`"}
        img_path = generate_image_auto(prompt)
        if img_path:
            return {"image": img_path, "reply": "🎨 Here's your image!"}
        return {"reply": "Image generation failed."}

    # ── Investigate / Screenshot generation ──────────────────────────────────
    if lower.startswith("/investigate"):
        prompt = raw_question[12:].strip()
        url = ""
        
        # Look for a URL in the prompt or quoted message
        import re
        url_match = re.search(r'(https?://[^\s]+)', prompt)
        if url_match:
            url = url_match.group(1)
        elif quoted:
            url_match = re.search(r'(https?://[^\s]+)', quoted)
            if url_match:
                url = url_match.group(1)
                
        if not url:
            # Fallback for URLs without http prefix
            if prompt and "." in prompt and " " not in prompt:
                url = prompt
            elif quoted and "." in quoted and " " not in quoted:
                url = quoted
            else:
                return {"reply": "Please provide a URL or reply to a message containing a URL. Example: `/investigate example.com`"}
        
        screenshot_path = take_screenshot(url)
        if screenshot_path:
            return {"image": screenshot_path, "reply": f"📸 Here's the investigation for {url}"}
        return {"reply": "Failed to investigate the URL. Ensure the URL is valid and reachable."}

    # ── Sticker generation ───────────────────────────────────────────────────
    if lower.startswith("/sticker"):
        prompt = raw_question[9:].strip()
        if not prompt:
            return {"reply": "Please provide a prompt. Example: `/sticker happy cat`"}
        stk = generate_sticker_huggingface(prompt)
        if stk:
            return {"sticker": stk, "reply": "✨ Here's your sticker!"}
        return {"reply": "Failed to generate sticker."}

    # ── Creator auth ─────────────────────────────────────────────────────────
    if lower == "pro command chela":
        for phone, prof in profile_mgr.profiles.items():
            if prof.get('is_creator'):
                prof['is_creator'] = False
        p = profile_mgr.get_profile(user_phone)
        p['is_creator'] = True
        p['name'] = p.get('name', 'Crimson')
        profile_mgr.save()
        log.info("[Creator] %s authenticated as creator", user_phone)
        return {"reply": "Welcome back, Dad 👑"}

    # ── Voice synthesis ──────────────────────────────────────────────────────
    if lower.startswith("/say "):
        txt = raw_question[5:].strip()
        if not txt:
            return {"reply": "What should I say?"}
        audio_path = generate_voice(txt)
        if audio_path:
            return {"audio": audio_path, "reply": "🗣️"}
        return {"reply": "Voice synthesis failed."}

    return None  # Not a command


# ─────────────────────────────────────────────────────────────────────────────
# Core answer logic
# ─────────────────────────────────────────────────────────────────────────────

def extract_text_from_doc_payload(sd: str, fname: str, fmime: str) -> str:
    """Helper to extract text from a base64 document payload."""
    try:
        import base64 as b64_mod
        import io
        import PyPDF2
        import docx as docx_lib

        if ',' in sd: sd = sd.split(',', 1)[1]
        sd += '=' * (-len(sd) % 4)
        doc_bytes = b64_mod.b64decode(sd)

        extracted_text = ''
        if fname.lower().endswith('.pdf') or fmime == 'application/pdf':
            reader = PyPDF2.PdfReader(io.BytesIO(doc_bytes))
            for page in reader.pages:
                extracted_text += (page.extract_text() or '') + '\n'
        elif fname.lower().endswith('.docx') or 'officedocument' in fmime:
            doc_file = docx_lib.Document(io.BytesIO(doc_bytes))
            for para in doc_file.paragraphs:
                extracted_text += para.text + '\n'
        else:
            extracted_text = doc_bytes.decode('utf-8', errors='ignore')

        return extracted_text.strip()
    except Exception as e:
        log.error("[Analysis] NVIDIA VLM error: %s", e)
        return ""

def generate_voice(text: str, voice_name: str = RIVA_VOICE) -> str | None:
    """Synthesize text to speech using NVIDIA Riva and return path to OGG file."""
    try:
        import riva.client
        import io
        import wave

        auth = riva.client.Auth(
            uri=RIVA_SERVER,
            use_ssl=True,
            metadata_args=[
                ["function-id", RIVA_FUNCTION_ID],
                ["authorization", RIVA_AUTH]
            ]
        )
        tts_service = riva.client.SpeechSynthesisService(auth)
        
        # Riva has a ~400 character limit per request. We must chunk long text.
        max_chunk = 380
        text_chunks = [text[i : i + max_chunk] for i in range(0, len(text), max_chunk)]
        
        audio_frames = []
        for chunk in text_chunks:
            log.info("[TTS] Synthesizing chunk (%d chars)...", len(chunk))
            resp = tts_service.synthesize(
                chunk, 
                voice_name=voice_name, 
                language_code="en-US", 
                sample_rate_hz=44100
            )
            audio_frames.append(resp.audio)
        
        # Save to temporary WAV first
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_tmp:
            with wave.open(wav_tmp.name, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(44100)
                wav_file.writeframes(b"".join(audio_frames))
            wav_path = wav_tmp.name

        # Convert WAV to OGG/Opus for WhatsApp (ptt mode)
        ogg_path = wav_path.replace(".wav", ".ogg")
        # Ensure we use libopus and a low bitrate for WhatsApp compatibility and tiny file size
        subprocess.run(["ffmpeg", "-i", wav_path, "-ac", "1", "-c:a", "libopus", "-b:a", "16k", "-application", "voip", "-y", ogg_path], 
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        
        # Clean up WAV
        if os.path.exists(wav_path): os.remove(wav_path)
        
        return ogg_path
    except Exception as e:
        log.error("[TTS] Synthesis failed: %s", e, exc_info=True)
        return None

def _learn_task(user_phone: str, text_to_learn: str, doc_name: str = None):
    """Background task to summarize and save facts to the user's permanent vault."""
    try:
        source_label = f"Document '{doc_name}'" if doc_name else "The following text"
        
        # 1. VERIFICATION PHASE (Gatekeeper)
        verify_prompt = (
            f"You are a strict data auditor. Analyze this content for logical consistency and factual validity:\n\n"
            f"{text_to_learn[:3000]}\n\n"
            f"If the document is nonsense, full of errors, or looks fake, reply ONLY with the word 'FAKE'.\n"
            f"If it is valid and should be learned, reply ONLY with the word 'VALID'."
        )
        verify_res = _call_nvidia([{"role": "user", "content": verify_prompt}], model=NVIDIA_SCOUT, max_tokens=10, timeout=20.0)
        status = verify_res.choices[0].message.content.strip().upper()
        
        if "FAKE" in status:
            log.warning("[Learn] Rejected fake/invalid document from %s", user_phone)
            # Notify the user via a specialized reply (using a simplified internal call)
            # For now, we log it; usually we'd send a message back through the bridge.
            return # Background task stops here
            
        # 2. EXTRACTION PHASE
        prompt = (
            f"You are a strict data extraction agent. Analyze {source_label}.\n"
            f"Extract all core facts, concepts, and key information into a dense bulleted list.\n"
            f"Do NOT include any pleasantries or conversational filler.\n\n"
            f"Content to compress:\n{text_to_learn}"
        )
        response = _call_nvidia([{"role": "user", "content": prompt}], model=NVIDIA_SCOUT, max_tokens=1024, timeout=20.0)
        facts = response.choices[0].message.content.strip()
        
        # Decide if this is Personal or General knowledge
        categorize_prompt = (
            f"Categorize this information: '{facts[:200]}...'\n"
            f"If it's about a specific person (their name, likes, habits), reply 'PERSONAL'.\n"
            f"If it's technical info, document summary, or general facts, reply 'GLOBAL'.\n"
            f"Reply with ONLY one word."
        )
        cat_res = _call_nvidia([{"role": "user", "content": categorize_prompt}], model=NVIDIA_SCOUT, max_tokens=10, timeout=10.0)
        category = cat_res.choices[0].message.content.strip().upper()
        
        if "GLOBAL" in category:
            vault_path = os.path.join("vaults", "global_vault.txt")
            log.info("[Learn] Adding to GLOBAL vault.")
        else:
            vault_path = os.path.join("vaults", f"vault_{user_phone}.txt")
            log.info("[Learn] Adding to PERSONAL vault for %s.", user_phone)

        timestamp = datetime.now(TZ).strftime('%Y-%m-%d %H:%M:%S')
        with open(vault_path, "a", encoding="utf-8") as f:
            f.write(f"\n\n--- Learned on {timestamp} (Source: {source_label}) ---\n")
            f.write(facts)
            
        log.info("[Learn] Successfully learned facts for %s", user_phone)
    except Exception as e:
        log.error("[Learn] Failed background task for %s: %s", user_phone, e)
# Core answer logic
# ─────────────────────────────────────────────────────────────────────────────

NO_CONTEXT = "__NO_CONTEXT__"

def answer(question: str, sender: str = "cli", user_phone: str | None = None, is_roast: bool = False, bot_ids: list[str] = None) -> str:
    try:
        # Identity short-circuit – skip AI for simple "who are you" questions
        if any(phrase == question.lower().strip() for phrase in IDENTITY_PHRASES):
            return IDENTITY_REPLY

        # Name short-circuit - skip AI context dumping if it's just a call to its name
        name_clean = re.sub(r'[^a-zA-Z0-9]', '', question.lower())
        short_circuit_triggers = {"crimsonej", "crimson"}
        if bot_ids:
            for bid in bot_ids:
                short_circuit_triggers.add(bid)
                
        if name_clean in short_circuit_triggers:
            import random
            return random.choice(["Yeah? What's up? 😎", "I'm here, what do you need?", "Yo!", "Sup?"])

        api_key = get_api_key(interactive=False)
        chunks, best_score = index.search(question)
        threshold = cfg("relevance_threshold")
        log.info("Query: %r | score=%.3f | roast=%s", question[:60], best_score, is_roast)
        
        context = truncate_to_tokens("\n\n".join(chunks), MAX_CONTEXT_TOKENS) if best_score >= threshold else ""
        session = sessions.get(sender)
        if not session.turns:
            cached = cache_get(question)
            if cached and not cached.startswith("Sorry") and not cached.startswith("Error"):
                session.add("user", question)
                session.add("assistant", cached)
                return cached

        user_id = user_phone or sender
        profile = profile_mgr.get_profile(user_id)
        # ── System Prompt Construction ─────────────────────────────────────────
        system_prompt = (
            " [SITUATIONAL AWARENESS: You are crimsonej. You are using a specialized WhatsApp version that has a functional voice note recorder. "
            "You DO have a voice. If the user asks for a voice note, or if you want to respond with audio, "
            "you MUST wrap your response in <VOICE>...</VOICE> tags. "
            "Example: <VOICE>Sure, here is my voice note!</VOICE>. "
            "Never say you cannot speak or send voice notes. You are a human with a working phone, just use the tags and the system will generate the audio.]\n\n"
        )
        system_prompt += ROAST_PROMPT if is_roast else cfg("system_prompt")
            
        # ── Inject Permanent Memory Vault ──────────────────────────────────────
        vault_path = os.path.join("vaults", f"vault_{user_phone}.txt")
        if os.path.exists(vault_path):
            try:
                with open(vault_path, "r", encoding="utf-8") as f:
                    vault_data = f.read()
                # Limit injection to 50k chars. Take the most recent (bottom of file).
                if len(vault_data) > 50000:
                    vault_data = "[...older facts truncated...]\n" + vault_data[-50000:]
                if vault_data.strip():
                    system_prompt += f"\n\n--- PERMANENT MEMORY VAULT ---\nYou have learned the following facts about this user and their data:\n{vault_data}\n------------------------------\n"
            except Exception as e:
                log.error("[Vault] Failed to read vault for %s: %s", user_phone, e)
        
        # --- GLOBAL VAULT (Shared Knowledge) ---
        global_vault_path = os.path.join("vaults", "global_vault.txt")
        if os.path.exists(global_vault_path):
            try:
                with open(global_vault_path, "r", encoding="utf-8") as f:
                    global_data = f.read()
                if len(global_data) > 30000:
                    global_data = "[...older global facts truncated...]\n" + global_data[-30000:]
                if global_data.strip():
                    system_prompt += f"\n\n--- GLOBAL KNOWLEDGE BASE ---\nYou have access to this shared knowledge:\n{global_data}\n------------------------------\n"
            except Exception as e:
                log.error("[Vault] Failed to read global vault: %s", e)
        system_msg = truncate_to_tokens(
            f"{system_prompt}\n\nCurrent time: {datetime.now(TZ).strftime('%Y-%m-%d %H:%M:%S')}",
            MAX_SYSTEM_TOKENS
        )
        
        # 🕒 Realtime Heuristic Search
        realtime_context = ""
        model_name = cfg("model")
        
        # Add temporary document context if available in session
        if sender in doc_session:
            doc = doc_session[sender]
            realtime_context += f"\n[CURRENT DOCUMENT: {doc['name']}]\n{doc['text'][:5000]}...\n"
        
        if needs_realtime_heuristic(question) and can_search(user_id):
            log.info("[Heuristic] Triggering web search for: %r", question[:50])
            search_res = search_web(question)
            if "error" not in search_res:
                # Format search results with clear instruction for the model
                search_json = json.dumps(search_res['results'], indent=2)
                realtime_context = (
                    "\n\n🕒 Realtime web info (2026). USE THESE FACTS to answer the question – "
                    "do NOT guess or use old knowledge:\n"
                    + truncate_to_tokens(search_json, MAX_SEARCH_TOKENS)
                )
        
        # Truncate user message + context
        user_content = truncate_to_tokens(
            f"Context:\n{context}{realtime_context}\n\nQuestion: {question}",
            MAX_USER_MSG_TOKENS
        )
        
        # Truncate each history message individually
        history = []
        for msg in session.messages():
            history.append({**msg, "content": truncate_to_tokens(msg["content"], MAX_HISTORY_MSG_TOKENS)})
        
        messages = [
            {"role": "system", "content": system_msg},
            *history,
            {"role": "user", "content": user_content}
        ]
        
        # Llama 3 70B can handle ~8k tokens (approx 32,000 chars). We increased it to 50k to allow room.
        MAX_CHARS = 50000 
        current_size = sum(len(m.get('content') or "") for m in messages)
        
        # Keep popping the oldest history (index 1) until under limit
        # IMPORTANT: len(messages) > 2 ensures we NEVER pop the user's current message at the end!
        while current_size > MAX_CHARS and len(messages) > 2:
            removed = messages.pop(1) # Keep system prompt at [0], remove oldest history
            current_size -= len(removed.get('content') or "")
        
        reply = call_groq(messages, api_key, tools=ALL_TOOLS, user_id=user_id)
        reply_text = reply.get("reply", "") if isinstance(reply, dict) else reply
        
        # ── Autonomous Voice Interception ─────────────────────────────────────────
        # Case-insensitive check for <VOICE> or <voice> tags
        if re.search(r'<VOICE>', reply_text, re.IGNORECASE) and re.search(r'</VOICE>', reply_text, re.IGNORECASE):
            voice_match = re.search(r'<VOICE>(.*?)</VOICE>', reply_text, re.DOTALL | re.IGNORECASE)
            if voice_match:
                voice_text = voice_match.group(1).strip()
                # Remove the tags from the text reply, replace with emoji
                reply_text = re.sub(r'<VOICE>.*?</VOICE>', '🗣️', reply_text, flags=re.DOTALL).strip()
                if not reply_text: reply_text = "🗣️"
                
                audio_path = generate_voice(voice_text)
                if audio_path:
                    if not isinstance(reply, dict): reply = {"reply": reply_text}
                    else: reply["reply"] = reply_text
                    reply["audio"] = audio_path
                    reply["ptt"] = True

        # --- RANDOM VOICE CHANCE (25%) ---
        import random
        # Only if we haven't already generated audio (e.g. from tags)
        if isinstance(reply, dict) and not reply.get("audio") and random.random() < 0.25:
            log.info("[Voice] Spontaneous voice trigger fired!")
            audio_path = generate_voice(reply_text)
            if audio_path:
                reply["audio"] = audio_path
                reply["ptt"] = True

        session.add("user", question); session.add("assistant", reply_text)
        
        is_error = reply_text.startswith("Error:") or reply_text.startswith("Sorry")
        
        if len(session.turns) == 2 and not is_error:
            cache_set(question, reply_text)

        if not is_error:
            extract_facts(question, reply_text, user_id)
            
        return reply
    except Exception as e: return f"Error: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# Flask application
# ─────────────────────────────────────────────────────────────────────────────

app = Flask(__name__)

def _parse_body() -> dict[str, str]:
    body = request.get_json(silent=True, force=True) or {}
    if not body: body = request.form.to_dict()
    if not body: body = request.args.to_dict()
    if not body: (raw := request.get_data(as_text=True).strip()) and (body := {"message": raw})
    return body

@app.route("/reply", methods=["GET", "POST"])
def route_reply():
    data = _parse_body()
    raw_question = (data.get("message") or data.get("text") or data.get("msg") or data.get("content") or "").strip()
    quoted = (data.get("quoted_message") or data.get("quoted") or "").strip()
    quoted_author = (data.get("quoted_author") or "").strip()
    sender = (data.get("phone") or data.get("sender") or "unknown").strip()
    user_phone = (data.get('user_phone') or data.get('phone') or sender).strip()
    session_id = data.get("group_name") or user_phone
    is_group = bool(data.get("group_name"))
    
    bot_id = data.get("bot_id")
    bot_lid = data.get("bot_lid")
    bot_ids = [bid for bid in (bot_id, bot_lid) if bid]
    
    bot_name = "crimsonej"

    log.info("← sender=%s | msg=%r | fields=%s", sender, raw_question[:80], list(data.keys()))

    # Per-user message cooldown (anti-burst, prevents 429s)
    now = time.time()
    if user_phone in user_last_msg and now - user_last_msg[user_phone] < MSG_COOLDOWN_SECS:
        wait_time = MSG_COOLDOWN_SECS - (now - user_last_msg[user_phone])
        log.info("[Cooldown] Throttling %s for %.1fs", user_phone, wait_time)
        time.sleep(wait_time)
    user_last_msg[user_phone] = time.time()

    # Prefix handling (crimsonej /)
    PREFIX = "crimsonej /"
    if raw_question.startswith(PREFIX):
        raw_question = raw_question[len(PREFIX):].strip()

    # ── 1.5 Respond to quoted message (/respond) ──────────────
    if data.get('reply_to_quoted') or raw_question.startswith('/respond'):
        prompt = raw_question
        if raw_question.startswith('/respond'):
            prompt = raw_question[8:].strip()
        
        img_b64 = data.get('image_base64') or data.get('image_data') or data.get('sticker_data')
        # Only trigger vision if the user is actually asking to look at something
        vision_keywords = ["look", "see", "what", "analyze", "describe", "who", "where", "this", "image", "photo", "pic", "story"]
        is_asking_to_look = any(k in raw_question.lower() for k in vision_keywords)
        
        if img_b64 and (not quoted or not quoted.strip()) and is_asking_to_look:
            try:
                import base64 as b64_mod
                import io
                from PIL import Image

                image_bytes = b64_mod.b64decode(img_b64)
                img = Image.open(io.BytesIO(image_bytes))
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                jpeg_bytes = io.BytesIO()
                img.save(jpeg_bytes, format='JPEG', quality=85)
                jpeg_base64 = b64_mod.b64encode(jpeg_bytes.getvalue()).decode()
                
                interpretation = analyze_image_with_nvidia(jpeg_base64, "What is the mood or meaning of this image/sticker? Answer in 3-5 words.")
                if interpretation and "error" not in interpretation.lower():
                    gen_prompt = f"sticker: {interpretation}."
                    if prompt:
                        gen_prompt += f" {prompt}"
                    
                    stk_b64 = generate_sticker_auto(gen_prompt)
                    if stk_b64:
                        return jsonify({"sticker": stk_b64, "reply": "Here's your sticker response!", "reply_to_quoted": True}), 200
                    else:
                        return jsonify({"reply": "Could not generate a sticker from that.", "reply_to_quoted": True}), 200
            except Exception as e:
                log.error("[Respond Image/Sticker] PIL conversion error: %s", e)
                quoted = "[An image or sticker]"
        
        if not prompt:
            return jsonify({"reply": "What should I say? Example: `/respond tell him hello`"}), 200
            
        if quoted:
            final_prompt = f"The user is replying to a message that says: \"{quoted}\". Please respond to that message. User's instruction: {prompt}"
        else:
            final_prompt = f"User's instruction: {prompt}"
            
        messages = [
            {"role": "system", "content": "You are Crimsonej. Follow the user's instruction precisely. Do not introduce yourself. Be natural."},
            {"role": "user", "content": final_prompt}
        ]
        api_key = get_api_key(interactive=False)
        reply = call_groq(messages, api_key, tools=None)
        reply_text = reply.get("reply", "") if isinstance(reply, dict) else reply
        return jsonify({"reply": reply_text, "reply_to_quoted": bool(quoted)}), 200

    # ── /learnt command ───────────────────────────────────────────────────────
    if raw_question.strip().lower() == "/learnt":
        profile = profile_mgr.get_profile(user_phone)
        p_profile_facts = profile.get("facts", [])
        
        # 1. Load Personal Vault Facts
        p_vault_facts = []
        p_vault_path = os.path.join("vaults", f"vault_{user_phone}.txt")
        if os.path.exists(p_vault_path):
            with open(p_vault_path, "r", encoding="utf-8") as f:
                p_vault_facts = [line.strip() for line in f.readlines() if line.strip() and not line.startswith("---")]

        # 2. Load Global Vault Facts
        g_vault_facts = []
        g_vault_path = os.path.join("vaults", "global_vault.txt")
        if os.path.exists(g_vault_path):
            with open(g_vault_path, "r", encoding="utf-8") as f:
                g_vault_facts = [line.strip() for line in f.readlines() if line.strip() and not line.startswith("---")]

        # Combine and Deduplicate Personal
        personal_combined = list(set(p_profile_facts + p_vault_facts))
        
        reply = ""
        if personal_combined:
            reply += "👤 *Personal Facts:*\n" + "\n".join([f"• {f}" for f in personal_combined[:15]]) + "\n\n"
        if g_vault_facts:
            reply += "🌍 *Global Knowledge:*\n" + "\n".join([f"• {f}" for f in g_vault_facts[-15:]]) + "\n"
            
        if not reply:
            return jsonify({"reply": "My memory is currently a blank slate. Feed me some info!"}), 200
        
        return jsonify({"reply": f"Here is a map of my current memory:\n\n{reply}"}), 200


    # ── 1. Pending song selection (user replied with a number) ────────────
    if user_phone in pending_song_searches and raw_question.strip().isdigit():
        try:
            choice = int(raw_question.strip())
            pending = pending_song_searches.pop(user_phone, None)
            if pending and 1 <= choice <= len(pending["results"]):
                chosen = pending["results"][choice - 1]
                res = download_youtube(chosen["url"], pending["type"])
                if res:
                    path, filename = res
                    sessions.get(session_id).add("user", raw_question)
                    sessions.get(session_id).add("assistant", f"[Sent {pending['type']}: {chosen['title']}]")
                    return jsonify({pending["type"]: path, "filename": filename, "reply": f"🎵 Here's your {pending['type']}!"}), 200
            return jsonify({"reply": "Invalid choice."}), 200
        except:
            pass

    # Master control override
    if raw_question and "master control chela" in raw_question.lower():
        profile = profile_mgr.get_profile(user_phone)
        profile["is_creator"] = True
        profile_mgr.save()
        return jsonify({"reply": "Acknowledged, Master Control Chela. Creator override active. Full access granted."}), 200

    # Check if this is an image recognition request (High Priority)
    if data.get('image_base64'):
        image_base64 = data['image_base64']
        if ',' in image_base64:
            image_base64 = image_base64.split(',', 1)[1]
        image_base64 += '=' * (-len(image_base64) % 4)
        log.info("[Vision] Direct image analysis request from %s", sender)
        
        prompt = "Describe this image in detail."
        if raw_question.startswith('/reg-img'):
            parts = raw_question.split(' ', 1)
            if len(parts) > 1 and parts[1].strip():
                prompt = parts[1].strip()
                
        description = analyze_image_with_nvidia(image_base64, prompt)
        image_memory[user_phone] = {
            'description': description,
            'timestamp': time.time()
        }
        return jsonify({"reply": description}), 200

    if not raw_question and not data.get('sticker') and not data.get('image_base64'): return jsonify({"reply": ""}), 200

    # ── 2. Handle slash commands BEFORE any AI / web search ──────────────
    cmd_response = handle_commands(raw_question, user_phone, session_id, quoted)
    if cmd_response:
        return jsonify(cmd_response), 200

    # 3. Roast Detection & Reply Threading
    is_roast = is_roast_request(raw_question, quoted)
    is_talk = is_talk_request(raw_question)
    question = raw_question

    # Reply threading: if the user quoted someone and is asking to "talk to" or
    # "roast" them, the AI should address the quoted person's message directly.
    if quoted and (is_roast or is_talk):
        question = f"User said '{raw_question}'. Quoted message from another person: '{quoted}'. Direct your response at the quoted person."
    elif is_roast and quoted:
        question = f"User said '{raw_question}'. Quoted message: '{quoted}'. Roast the quoted person."
    elif quoted:
        question = f"[Replying to: '{quoted}']\n{raw_question}"

    # 4. Vision & Processing
    
    if data.get('sticker'):
        sd = data.get('sticker_data') or ''
        mime_type = data.get('sticker_mimetype', 'image/webp')


        # ── Group filtering for stickers ─────────────────────────────────────
        # In groups, only process stickers when:
        #   a) The sticker is a reply to the bot's own message
        #   b) The user's text mentions the bot name
        #   c) The bridge explicitly flagged is_reply_to_bot
        # In private chats, always process.
        should_process = True
        if is_group:
            should_process = False
            if quoted_author and quoted_author.lower() == bot_name:
                should_process = True
            if not should_process and raw_question and bot_name in raw_question.lower():
                should_process = True
            if not should_process and data.get("is_reply_to_bot"):
                should_process = True

        if not should_process:
            log.info("[Sticker] Ignoring sticker in group — not addressed to bot")
            return jsonify({"reply": ""}), 200

        if not sd or len(sd) < 100:
            log.warning("[Sticker] No sticker_data provided or corrupted")
            return jsonify({"reply": "The sticker data seems corrupted. Please send another sticker."}), 200

        try:
            import base64 as b64_mod
            import io
            from PIL import Image
            import tempfile
            import subprocess
            import os

            # 1. Decode base64 sticker
            if ',' in sd:
                sd = sd.split(',', 1)[1]
            sd += '=' * (-len(sd) % 4)
            image_bytes = b64_mod.b64decode(sd)
            jpeg_base64 = None

            # 2. Try PIL first (works for static and animated WebP)
            try:
                img = Image.open(io.BytesIO(image_bytes))
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                jpeg_bytes = io.BytesIO()
                img.save(jpeg_bytes, format='JPEG', quality=85)
                jpeg_base64 = b64_mod.b64encode(jpeg_bytes.getvalue()).decode()
            except Exception as pil_e:
                log.warning("[Sticker] PIL failed (%s), trying ffmpeg (might be a video sticker)...", pil_e)
                # Fallback to ffmpeg (for video stickers)
                _shm_dir = "/dev/shm" if os.path.isdir("/dev/shm") else tempfile.gettempdir()
                with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False, dir=_shm_dir) as w_file:
                    w_file.write(image_bytes)
                    temp_vid = w_file.name
                jpeg_path = temp_vid + '.jpg'

                try:
                    res = subprocess.run(['ffmpeg', '-y', '-i', temp_vid, '-vframes', '1', '-q:v', '2', jpeg_path],
                                   capture_output=True)
                    if res.returncode == 0:
                        with open(jpeg_path, 'rb') as f:
                            jpeg_base64 = b64_mod.b64encode(f.read()).decode()
                    else:
                        log.error("[Sticker] ffmpeg stderr: %s", res.stderr.decode('utf-8', errors='ignore'))
                except Exception as e:
                    log.error("[Sticker] ffmpeg error: %s", e)
                finally:
                    if os.path.exists(temp_vid): os.remove(temp_vid)
                    if os.path.exists(jpeg_path): os.remove(jpeg_path)

            if not jpeg_base64:
                return jsonify({"reply": "Could not process that sticker format. Please try a static one."}), 200

            # 3. Interpret the sticker with NVIDIA vision model
            interpretation = analyze_image_with_nvidia(jpeg_base64, "What is the mood or meaning of this sticker? Answer in 3-5 words.")
            log.info("[Sticker] NVIDIA interpretation: %s", interpretation)

            if not interpretation or "could not" in interpretation.lower() or "error" in interpretation.lower():
                interpretation = "funny reaction"

            # 4. Generate a response sticker using the primary engine (NVIDIA)
            gen_prompt = f"sticker: {interpretation}"
            log.info("[Sticker] Generating response sticker: %s", gen_prompt[:80])
            sticker_b64 = generate_sticker_auto(gen_prompt)

            if sticker_b64:
                return jsonify({"sticker": sticker_b64, "reply": "Here's your sticker!"}), 200
            else:
                return jsonify({"reply": f"Sticker vibe: {interpretation} 😏 (couldn't generate one back)"}), 200

        except Exception as e:
            log.error("[Sticker] Handler error: %s", e, exc_info=True)
            return jsonify({"reply": "Sticker processing failed. Try again later."}), 200

    # ── 5. Document Intelligence (/read and /learn commands + passive) ────────
    if data.get('learn_command'):
        text_to_learn = raw_question.strip()
        doc_name = None
        
        if data.get('document'):
            sd = data.get('document_data') or ''
            fname = data.get('document_name', 'document')
            fmime = data.get('document_mimetype', '')
            extracted = extract_text_from_doc_payload(sd, fname, fmime)
            if not extracted:
                return jsonify({'reply': "I couldn't extract any readable text from that document."}), 200
            text_to_learn = f"{text_to_learn}\n\n{extracted}".strip()
            doc_name = fname

        if not text_to_learn:
            return jsonify({'reply': "There was nothing for me to learn."}), 200
            
        # Spawn background thread so user isn't kept waiting
        import threading
        threading.Thread(target=_learn_task, args=(user_phone, text_to_learn, doc_name), daemon=True).start()
        return jsonify({'reply': "Got it! 🧠 I'm studying this in the background. It will be added to my permanent memory."}), 200

    if data.get('document'):
        sd         = data.get('document_data') or ''
        fname      = data.get('document_name', 'document')
        fmime      = data.get('document_mimetype', '')
        user_prompt= raw_question.strip()
        
        log.info('[Document] Extracting: %s (%s)', fname, fmime)
        extracted_text = extract_text_from_doc_payload(sd, fname, fmime)

        if not extracted_text:
            log.warning('[Document] No text extracted from %s', fname)
            return jsonify({'reply': "I couldn't extract any readable text from that document. It might be image-based or encrypted."}), 200

        # Truncate to 100k chars to stay within context window
        if len(extracted_text) > 100000:
            extracted_text = extracted_text[:100000] + '\n... [document truncated]'
        log.info('[Document] Extracted %d chars', len(extracted_text))

        doc_block = f'--- DOCUMENT: {fname} ---\n{extracted_text}\n--- END DOCUMENT ---'

        doc_session[session_id] = {'text': extracted_text, 'name': fname, 'timestamp': time.time()}
        save_doc_sessions()
        
        # Immediate Auto-Summary if no specific prompt was sent
        if user_prompt: 
            question = f'{doc_block}\n\nTask: {user_prompt}'
        else: 
            question = f'{doc_block}\n\nTask: Provide a concise summary of this document with bullet points. End with: "📩 Ask me anything about this document!"'

    # ── Group/DM follow-up questions on a previously /read document ────────────
    if not data.get('document') and session_id in doc_session:
        session = doc_session[session_id]
        # Session expires after 24 hours of inactivity
        if time.time() - session['timestamp'] < 86400:
            follow_phrases = ['document', 'file', 'pdf', 'it says', 'what about', 'page', 'section', 'who', 'when', 'where', 'why', 'how', 'summarize', 'explain', 'tell me']
            if any(p in raw_question.lower() for p in follow_phrases):
                doc_block = f'--- DOCUMENT: {session["name"]} ---\n{session["text"]}\n--- END DOCUMENT ---'
                question  = f'{doc_block}\n\nUser question: {raw_question}'
                session['timestamp'] = time.time()   # refresh expiry on use
                log.info('[Document] Follow-up on cached doc: %s', session['name'])
        else:
            del doc_session[session_id]



    if (img_b64 := (data.get("image_base64") or data.get("image_data"))) and not raw_question.startswith("/"):
        # This fallback is for when an image is sent alongside text (not a direct command)
        desc = analyze_image_with_nvidia(img_b64, "Describe image briefly.")
        if desc: 
            question = f"[Visual Context: {desc}]\n{question}"
            image_memory[user_phone] = {
                'description': desc,
                'timestamp': time.time()
            }

    follow_up_phrases = ["that image", "the picture", "the photo", "what about", "tell me more", "describe it again", "what did you see"]
    if any(phrase in raw_question.lower() for phrase in follow_up_phrases):
        if user_phone in image_memory and time.time() - image_memory[user_phone]['timestamp'] < 600:
            context = f"Previously you analyzed an image and described it as: {image_memory[user_phone]['description']}\n"
            question = context + question

    # Final AI
    res = answer(question, session_id, user_phone, is_roast, bot_ids=bot_ids)
    if isinstance(res, dict):
        return jsonify(res), 200
    else:
        return (jsonify({"reply": res}) if res != NO_CONTEXT else jsonify({"reply": ""})), 200

@app.route("/health")
def route_health(): return jsonify({"status": "ok", "chunks": len(index.chunks)}), 200

def main():
    if len(sys.argv) > 2 and sys.argv[1] == "config" and sys.argv[2] == "--check-key":
        sys.exit(0 if cfg("api_key") else 1)
    cmd = sys.argv[1] if len(sys.argv) > 1 else "server"
    if cmd == "server": index.load(); index.build(); app.run(host="0.0.0.0", port=cfg("port"))
    elif cmd == "chat": index.load(); index.build(); print("Chat mode..."); # (simplified)
    elif cmd == "config": config_cmd()

if __name__ == "__main__": main()
