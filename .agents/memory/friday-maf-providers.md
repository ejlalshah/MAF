---
name: FRIDAY-MAF providers and search
description: LLM and search tool configuration for FRIDAY-MAF
---
## LLM Providers
Default: ollama (LLM_PROVIDER=ollama, LLM_MODEL=llama3.2, OLLAMA_BASE_URL=http://localhost:11434)
Stub fallback: OllamaProvider._stub_response() returns valid JSON shapes when Ollama is offline.
Other options: openai | anthropic | gemini (set LLM_PROVIDER + LLM_API_KEY in .env or env vars)

## Web Search
Uses ddgs package (pip install ddgs — renamed from duckduckgo_search).
WebSearchTool._ddg_search tries "ddgs" first, then "duckduckgo_search" as fallback import.
No API key required. Graceful stub fallback on network errors.

**Why:** All original paid APIs replaced with free alternatives per project spec.
