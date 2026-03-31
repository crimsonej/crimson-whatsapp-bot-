import logging
import time
from ddgs import DDGS


logger = logging.getLogger(__name__)

# Cache settings
SEARCH_CACHE = {}
CACHE_TTL = 300  # 5 minutes

# Keywords and question starters that trigger a web search
REALTIME_KEYWORDS = {
    'news', 'price', 'score', 'match', 'today', 'now',
    'current', 'forecast', 'stock', 'bitcoin', 'ethereum',

    'president', 'election', 'results', 'poll', 'game', 'sport',
    'who is', 'what is', 'where is', 'when did', 'how to', 'why did',
    'define', 'meaning of', 'population', 'capital', 'location'
}

def needs_realtime_heuristic(query):
    """Aggressively check if query likely needs live web info."""
    q_lower = query.lower()
    # Check for direct keywords or question phrases
    return any(k in q_lower for k in REALTIME_KEYWORDS)

def search_web(query, max_results=3):
    """Search using DuckDuckGo (ddgs) with caching."""
    # Check cache
    now = time.time()
    if query in SEARCH_CACHE:
        result, timestamp = SEARCH_CACHE[query]
        if now - timestamp < CACHE_TTL:
            return result

    try:
        ddgs = DDGS()
        results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return {"error": "No results found"}

        formatted = {
            "answer": results[0].get('body', '')[:300],
            "results": [
                {
                    "title": r.get('title', ''),
                    "content": r.get('body', '')[:500],
                    "url": r.get('href', '')
                }
                for r in results[:max_results]
            ]
        }
        # Cache the result
        SEARCH_CACHE[query] = (formatted, now)
        return formatted
    except Exception as e:
        logger.error(f"Search error: {e}")
        return {"error": str(e)}
