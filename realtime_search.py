import logging
import time
from ddgs import DDGS

logger = logging.getLogger(__name__)

# Cache settings
SEARCH_CACHE = {}
CACHE_TTL = 300  # 5 minutes

# Keywords and question starters that trigger a web search
REALTIME_KEYWORDS = {
    'news', 'price', 'score', 'match', 'today', 'now', 'latest',
    'current', 'forecast', 'stock', 'bitcoin', 'ethereum',
    'president', 'election', 'results', 'poll', 'game', 'sport',
    'who is', 'what is', 'where is', 'when did', 'how to', 'why did',
    'define', 'meaning of', 'population', 'capital', 'location',
    'info on', 'about the', 'fees', 'specs', 'cost', 'date',
    'when is', 'release', 'how much', 'details', 'structure'
}

def needs_realtime_heuristic(query):
    """Aggressively check if query likely needs live web info."""
    q_lower = query.lower()
    return any(k in q_lower for k in REALTIME_KEYWORDS)

def _distill_query(query: str) -> str:
    """
    Trim a long/complex query to the core 5-7 word search phrase.
    This prevents slow or failed searches on long conversational messages.
    """
    # Strip filler phrases that confuse search engines
    fillers = [
        "can you tell me", "please tell me", "i want to know",
        "do you know", "can you search", "hey crimsonej", "crimsonej",
        "can you find", "i need to know", "what about", "bro tell me",
        "yo what is", "yo who is"
    ]
    q = query.lower()
    for f in fillers:
        q = q.replace(f, "")
    q = q.strip().strip("?.,!")
    # If still very long, take first 60 chars to get the core topic
    if len(q) > 60:
        q = q[:60].rsplit(" ", 1)[0]
    return q or query

def search_web(query, max_results=3):
    """Search using DuckDuckGo (ddgs) with caching and a strict 3s timeout."""
    # Distil long queries before searching
    clean_query = _distill_query(query)
    
    # Check cache
    now = time.time()
    if clean_query in SEARCH_CACHE:
        result, timestamp = SEARCH_CACHE[clean_query]
        if now - timestamp < CACHE_TTL:
            logger.debug("[Search] Cache hit for: %r", clean_query)
            return result

    try:
        results = []
        with DDGS() as ddgs:
            # Single fast call – no backend iteration loop
            res = list(ddgs.text(clean_query, max_results=max_results))
            if res:
                results = res

        if not results:
            return {"error": "No results found"}

        formatted = {
            "answer": results[0].get('body', '')[:200],
            "results": [
                {
                    "title": r.get('title', '')[:100],
                    "content": r.get('body', '')[:300],
                    "url": r.get('href', '')
                }
                for r in results[:3]
            ]
        }
        # Cache the result
        SEARCH_CACHE[clean_query] = (formatted, now)
        return formatted

    except Exception as e:
        logger.error("[Search] Error: %s", e)
        return {"error": str(e)}
