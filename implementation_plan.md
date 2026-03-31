# Search Enhancements and Import Update

This plan updates the DuckDuckGo search integration to use the `ddgs` library as requested, narrows the search triggers, and adds a per-user cooldown to prevent excessive tool usage.

## User Review Required

> [!IMPORTANT]
> - The import in `realtime_search.py` will be changed to `from ddgs import DDGS`.
> - A 30-second search cooldown per user will be implemented in `bot.py`.
> - The `latest` keyword will be removed from the realtime search heuristic to reduce false positives.
> - Search results will be injected directly into the context sent to Groq if the heuristic matches.

## Proposed Changes

### Realtime Search Integration

#### [MODIFY] [realtime_search.py](file:///home/joa/groq-bot/realtime_search.py)
- Change import to `from ddgs import DDGS`.
- Remove `latest` from `REALTIME_KEYWORDS`.

### Bot Logic

#### [MODIFY] [bot.py](file:///home/joa/groq-bot/bot.py)
- **Globals**: Add `user_last_search = {}`.
- **Functions**: Add `can_search(user_id)` function.
- **Answer Logic**:
  - In `answer()`, call `needs_realtime_heuristic(question)` and `can_search(user_id)`.
  - If triggered, fetch results via `search_web()` and prepend to `context` with the header `🕒 Realtime web info:`.
- **Tool Logic**:
  - Update `call_groq` to also respect the `can_search` cooldown for its tool calls if needed.

## Open Questions

- None.

## Verification Plan

### Automated Tests
- Run `python3 bot.py` to check for syntax errors.
- Test the `needs_realtime_heuristic` with various strings.

### Manual Verification
- Send a message with "weather" and verify it triggers a search and populates wait/results.
- Send a second search within 30 seconds and verify the cooldown prevents the search from being repeated (it should use RAG or memory instead).
- Verify the `from ddgs import DDGS` import doesn't cause a crash.