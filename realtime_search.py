"""
realtime_search.py – DuckDuckGo-powered web search with smart detection.

Provides:
  - needs_realtime_search(query, session_turns)  → bool
  - search_duckduckgo(query, max_results)         → dict
"""

import os
import re
import time
import logging
from ddgs import DDGS

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Keywords & patterns that signal a need for current/live data
# ─────────────────────────────────────────────────────────────────────────────

REALTIME_KEYWORDS = {
    # Time-sensitive / temporal
    'today', 'now', 'right now', 'currently', 'latest', 'current',
    'recent', 'recently', 'this week', 'this month', 'this year',
    'yesterday', 'tomorrow', 'tonight', 'last night', 'upcoming',
    'breaking', 'update', 'updates', 'live', 'ongoing', 'happening',

    # Weather
    'weather', 'forecast', 'temperature', 'rain', 'storm', 'hurricane',
    'tornado', 'climate', 'humidity', 'wind speed',

    # News & events
    'news', 'headline', 'headlines', 'announcement', 'announced',
    'breaking news', 'report', 'reports', 'incident', 'crisis',
    'war', 'conflict', 'protest', 'earthquake', 'flood',

    # Sports
    'score', 'scores', 'match', 'game', 'sport', 'sports',
    'fixture', 'fixtures', 'standings', 'league', 'tournament',
    'premier league', 'champions league', 'world cup', 'nba', 'nfl',
    'la liga', 'serie a', 'bundesliga', 'epl', 'ucl',
    'goal', 'goals', 'halftime', 'full time', 'lineup', 'transfer',
    'injury', 'suspended', 'red card', 'yellow card',

    # Finance & crypto
    'price', 'prices', 'stock', 'stocks', 'market', 'markets',
    'bitcoin', 'btc', 'ethereum', 'eth', 'crypto', 'cryptocurrency',
    'forex', 'exchange rate', 'usd', 'dollar', 'euro', 'yen',
    'trading', 'nasdaq', 'dow jones', 's&p', 'shares', 'invest',
    'inflation', 'gdp', 'economy', 'recession',

    # Politics & governance
    'president', 'election', 'elections', 'vote', 'voting', 'poll',
    'polls', 'results', 'government', 'parliament', 'senate',
    'congress', 'law', 'bill', 'legislation', 'sanction', 'sanctions',
    'minister', 'prime minister', 'governor', 'mayor',

    # Technology & releases
    'release', 'released', 'launch', 'launched', 'version',
    'update', 'patch', 'changelog', 'ios', 'android', 'windows',
    'iphone', 'samsung', 'pixel', 'mac', 'macbook',
    'ai model', 'chatgpt', 'openai', 'google', 'tesla', 'spacex',
    'twitter', 'x.com', 'tiktok', 'instagram', 'facebook', 'meta',

    # Entertainment & pop culture
    'movie', 'film', 'album', 'song', 'concert', 'tour',
    'box office', 'streaming', 'netflix', 'spotify', 'youtube',
    'trailer', 'premiere', 'grammy', 'oscar', 'emmy', 'award',
    'celebrity', 'died', 'death', 'born', 'wedding', 'divorce',
    'scandal', 'controversy', 'viral', 'trending', 'trend',

    # Health & science
    'covid', 'pandemic', 'vaccine', 'outbreak', 'virus',
    'who', 'disease', 'symptoms', 'cure', 'treatment', 'drug',
    'fda', 'approved', 'clinical trial', 'study', 'research',
    'nasa', 'space', 'rocket', 'satellite', 'asteroid',

    # Travel & logistics
    'flight', 'flights', 'airline', 'airport', 'delayed',
    'cancelled', 'visa', 'travel ban', 'border',

    # Shopping & products
    'deal', 'deals', 'discount', 'sale', 'coupon', 'promo',
    'availability', 'in stock', 'out of stock', 'shipping',

    # General lookups that often expect fresh data
    'how much', 'how many', 'who is', 'who won', 'who was',
    'what happened', 'when is', 'where is', 'is it true',
    'did', 'has', 'have they', 'are they', 'will',
}

# Patterns that indicate the user wants MORE info on a previous topic
FOLLOWUP_PATTERNS = [
    r'\btell me more\b',
    r'\bmore about\b',
    r'\bmore info\b',
    r'\bmore details\b',
    r'\belaborate\b',
    r'\bexplain (more|further|that)\b',
    r'\bwhat else\b',
    r'\bgo deeper\b',
    r'\bexpand on\b',
    r'\bcan you (find|search|look|check)\b',
    r'\bgive me more\b',
    r'\bwhat about\b',
    r'\bhow about\b',
    r'\band what\b',
    r'\balso\b.*\b(search|find|look)\b',
    r'\bwhat do you know about\b',
    r'\btell me about\b',
    r'\bsearch for\b',
    r'\blook up\b',
    r'\bgoogle\b',
    r'\bfind out\b',
    r'\bcheck (online|the web|internet)\b',
]

# Questions that are implicitly about current state or general information lookups
IMPLICIT_CURRENT_PATTERNS = [
    # General question starters
    r'\b(who|what|where|when|why|how) (is|are|was|were|did|does|do|has|have|will|would|can|could|to)\b',
    r'^(who|what|where|when|why|how)\b',
    
    # Specific targeted questions that might not be at the start
    r'\b(can you|could you|please) (explain|tell|find|search|look)\b',
    r'\b(is it true|did it happen)\b',
    r'\b(is|are|was|were) .+ (alive|dead|married|single|pregnant|retired)\b',
    
    # "What is the X of Y" patterns
    r'\bwhat(\'s| is) the\b',
    
    # Catch-all for basic informational queries (e.g. "what happened to X")
    r'\bwhat happened\b',
    r'^(explain|tell me about|describe) \b',
    
    # Distance, height, weight, etc.
    r'\bhow (tall|far|heavy|long|fast|old|much|many)\b',
]

# Compiled patterns
_followup_re = [re.compile(p, re.IGNORECASE) for p in FOLLOWUP_PATTERNS]
_implicit_re = [re.compile(p, re.IGNORECASE) for p in IMPLICIT_CURRENT_PATTERNS]


# ─────────────────────────────────────────────────────────────────────────────
# Search cache
# ─────────────────────────────────────────────────────────────────────────────

search_cache: dict[str, tuple] = {}
CACHE_TTL = 300  # 5 minutes


def _get_cached(query: str):
    """Return cached search result or None."""
    if query in search_cache:
        result, ts = search_cache[query]
        if time.time() - ts < CACHE_TTL:
            return result
    return None


def _set_cache(query: str, result: dict):
    search_cache[query] = (result, time.time())
    # Prune old entries if cache gets large
    if len(search_cache) > 200:
        cutoff = time.time() - CACHE_TTL
        stale = [k for k, (_, ts) in search_cache.items() if ts < cutoff]
        for k in stale:
            del search_cache[k]


# ─────────────────────────────────────────────────────────────────────────────
# Decision: does this message need live data?
# ─────────────────────────────────────────────────────────────────────────────

def _keyword_match(query: str) -> bool:
    """Check if query contains any real-time keyword."""
    q_lower = query.lower()
    return any(kw in q_lower for kw in REALTIME_KEYWORDS)


def _is_followup(query: str) -> bool:
    """Check if the message is a follow-up request."""
    return any(pat.search(query) for pat in _followup_re)


def _is_implicit_current(query: str) -> bool:
    """Check if the query implicitly asks about current state."""
    return any(pat.search(query) for pat in _implicit_re)


def _previous_was_realtime(session_turns: list[dict]) -> bool:
    """
    Check if the most recent exchange involved real-time search.
    We look at the last assistant message for the '🕒 Realtime web info'
    marker (injected by bot.py when search is triggered).
    """
    if not session_turns:
        return False
    # Walk backwards to find the last assistant message
    for turn in reversed(session_turns):
        if turn.get("role") == "assistant":
            content = turn.get("content", "")
            return "Realtime web info" in content or "🕒" in content
        if turn.get("role") == "user":
            # also check if the previous user message had real-time keywords
            return _keyword_match(turn.get("content", ""))
    return False


def needs_realtime_search(query: str, session_turns: list[dict] | None = None) -> bool:
    """
    Master decision function: should we search the web for this query?

    Checks (in order):
      1. Direct keyword match (e.g. "weather in Paris")
      2. Implicit current-state questions ("who is the president")
      3. Follow-up on a previous real-time topic ("tell me more about that")
      4. Explicit search requests ("search for X", "google Y")

    Parameters
    ----------
    query          : The user's message text.
    session_turns  : The current session's conversation turns (list of
                     {"role": ..., "content": ...} dicts). Used to detect
                     follow-ups that refer back to a real-time topic.
    """
    # 1. Direct keyword match
    if _keyword_match(query):
        return True

    # 2. Implicit current-state question
    if _is_implicit_current(query):
        return True

    # 3. Follow-up on a real-time conversation
    if _is_followup(query):
        if session_turns and _previous_was_realtime(session_turns):
            return True

    # 4. Explicit search request
    if _is_followup(query):
        # Patterns like "search for X", "look up Y" → always search
        explicit = re.search(
            r'\b(search|google|look up|find out|check online)\b',
            query, re.IGNORECASE,
        )
        if explicit:
            return True

    return False


# Keep backward-compatible alias
def needs_realtime_heuristic(query: str) -> bool:
    """Backward-compatible wrapper (no session context)."""
    return needs_realtime_search(query)


# ─────────────────────────────────────────────────────────────────────────────
# DuckDuckGo search (free, no API key)
# ─────────────────────────────────────────────────────────────────────────────

def search_duckduckgo(query: str, max_results: int = 3) -> dict:
    """
    Search using DuckDuckGo via the duckduckgo-search library.

    Returns
    -------
    dict with keys:
      "answer"  : str   – summary snippet from the top result
      "results" : list  – [{title, content, url}, ...]
    or:
      {"error": "..."}  – on failure
    """
    cached = _get_cached(query)
    if cached:
        logger.info("[DDG] Cache hit for: %r", query)
        return cached

    try:
        ddgs = DDGS()
        results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return {"error": "No results found"}

        answer = results[0].get("body", "")[:300] if results else ""
        formatted = {
            "answer": answer,
            "results": [
                {
                    "title": r.get("title", ""),
                    "content": r.get("body", "")[:500],
                    "url": r.get("href", ""),
                }
                for r in results[:max_results]
            ],
        }
        _set_cache(query, formatted)
        logger.info("[DDG] Got %d results for: %r", len(results), query)
        return formatted

    except Exception as e:
        logger.error("[DDG] Search error: %s", e)
        return {"error": str(e)}
