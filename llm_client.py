"""
llm_client.py -- LLM call wrapper with retry logic and multi-provider support.

Provides:
    get_llm_response(prompt, system_prompt=None, temperature=None) -> str

Supports:
    - OpenAI API (openai.OpenAI) — also works with DeepSeek via DEEPSEEK_API_KEY
    - Anthropic API (anthropic.Anthropic) — if ANTHROPIC_API_KEY is set

API key priority:
    1. DEEPSEEK_API_KEY → uses OpenAI SDK pointed at DeepSeek endpoint
    2. OPENAI_API_KEY  → uses OpenAI SDK with default/configured endpoint
    3. ANTHROPIC_API_KEY → uses Anthropic SDK

Retry logic:
    - Max 3 retries with exponential backoff (1s, 2s, 4s)
    - Catches RateLimitError, APITimeoutError, and connection errors
    - Returns stripped response content

Usage:
    from llm_client import get_llm_response
    answer = get_llm_response("What is 2+2?", system_prompt="Be concise.")
"""

import logging
import os
import time
from typing import Optional

from config import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_RETRIES = 3
"""Maximum number of retries on transient failures."""

INITIAL_BACKOFF = 1.0
"""Initial backoff in seconds for exponential retry (1s → 2s → 4s)."""

DEEPSEEK_BASE_URL = os.environ.get(
    "DEEPSEEK_BASE_URL",
    "https://api.deepseek.com",
)
"""Base URL for DeepSeek API (OpenAI-compatible). Override via DEEPSEEK_BASE_URL env var."""


# ---------------------------------------------------------------------------
# Internal: resolve API credentials
# ---------------------------------------------------------------------------

def _resolve_credentials() -> dict:
    """Detect available API credentials and return provider config.

    Priority: DEEPSEEK_API_KEY > OPENAI_API_KEY > ANTHROPIC_API_KEY.

    Returns:
        Dict with keys: provider ('openai' or 'anthropic'), api_key, base_url (optional).

    Raises:
        RuntimeError: If no API key is found in environment variables.
    """
    # Check DeepSeek first (OpenAI-compatible)
    deepseek_key = os.environ.get("DEEPSEEK_API_KEY")
    if deepseek_key:
        logger.info("Using DeepSeek API (OpenAI-compatible)")
        return {
            "provider": "openai",
            "api_key": deepseek_key,
            "base_url": DEEPSEEK_BASE_URL,
        }

    # Check OpenAI
    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
        logger.info("Using OpenAI API")
        return {
            "provider": "openai",
            "api_key": openai_key,
            "base_url": os.environ.get("OPENAI_BASE_URL"),
        }

    # Check Anthropic
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if anthropic_key:
        logger.info("Using Anthropic API")
        return {
            "provider": "anthropic",
            "api_key": anthropic_key,
            "base_url": None,
        }

    raise RuntimeError(
        "No API key found. Set one of: DEEPSEEK_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY.\n"
        "Example: export DEEPSEEK_API_KEY='sk-...'"
    )


# ---------------------------------------------------------------------------
# Internal: OpenAI / DeepSeek call
# ---------------------------------------------------------------------------

def _call_openai(
    prompt: str,
    system_prompt: Optional[str],
    temperature: float,
    api_key: str,
    base_url: Optional[str],
) -> str:
    """Call OpenAI-compatible API (OpenAI or DeepSeek).

    Args:
        prompt: The user prompt.
        system_prompt: Optional system-level instruction.
        temperature: Sampling temperature.
        api_key: API key string.
        base_url: Optional custom base URL (e.g., DeepSeek endpoint).

    Returns:
        Stripped response content string.

    Raises:
        RuntimeError: After exhausting all retries.
    """
    import openai

    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    client = openai.OpenAI(**client_kwargs)

    model = config.llm_model
    max_tokens = config.llm_max_tokens

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.debug(f"OpenAI call attempt {attempt}/{MAX_RETRIES} (model={model})")
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = response.choices[0].message.content
            if content is None:
                raise RuntimeError("LLM returned empty content (None)")
            logger.debug(f"OpenAI response: {len(content)} chars")
            return content.strip()

        except openai.RateLimitError as e:
            last_error = e
            logger.warning(f"Rate limit hit (attempt {attempt}/{MAX_RETRIES}): {e}")
        except openai.APITimeoutError as e:
            last_error = e
            logger.warning(f"API timeout (attempt {attempt}/{MAX_RETRIES}): {e}")
        except openai.APIConnectionError as e:
            last_error = e
            logger.warning(f"API connection error (attempt {attempt}/{MAX_RETRIES}): {e}")
        except openai.APIError as e:
            # Server-side errors (5xx) — retry; client errors (4xx) — don't retry
            last_error = e
            if getattr(e, "status_code", 500) and e.status_code < 500:
                raise RuntimeError(f"OpenAI API client error (4xx): {e}") from e
            logger.warning(f"API server error (attempt {attempt}/{MAX_RETRIES}): {e}")

        # Exponential backoff
        if attempt < MAX_RETRIES:
            delay = INITIAL_BACKOFF * (2 ** (attempt - 1))
            logger.info(f"Retrying in {delay:.1f}s...")
            time.sleep(delay)

    raise RuntimeError(
        f"LLM call failed after {MAX_RETRIES} retries. "
        f"Last error: {last_error}"
    )


# ---------------------------------------------------------------------------
# Internal: Anthropic call
# ---------------------------------------------------------------------------

def _call_anthropic(
    prompt: str,
    system_prompt: Optional[str],
    temperature: float,
    api_key: str,
) -> str:
    """Call Anthropic API.

    Args:
        prompt: The user prompt.
        system_prompt: Optional system-level instruction.
        temperature: Sampling temperature.
        api_key: Anthropic API key.

    Returns:
        Stripped response content string.

    Raises:
        RuntimeError: After exhausting all retries.
    """
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)

    model = config.llm_model
    max_tokens = config.llm_max_tokens

    kwargs = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system_prompt:
        kwargs["system"] = system_prompt
    if temperature > 0:
        kwargs["temperature"] = temperature

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.debug(f"Anthropic call attempt {attempt}/{MAX_RETRIES} (model={model})")
            response = client.messages.create(**kwargs)
            content = response.content[0].text
            if not content:
                raise RuntimeError("Anthropic returned empty content")
            logger.debug(f"Anthropic response: {len(content)} chars")
            return content.strip()

        except anthropic.RateLimitError as e:
            last_error = e
            logger.warning(f"Rate limit hit (attempt {attempt}/{MAX_RETRIES}): {e}")
        except anthropic.APITimeoutError as e:
            last_error = e
            logger.warning(f"API timeout (attempt {attempt}/{MAX_RETRIES}): {e}")
        except anthropic.APIConnectionError as e:
            last_error = e
            logger.warning(f"API connection error (attempt {attempt}/{MAX_RETRIES}): {e}")
        except anthropic.APIStatusError as e:
            last_error = e
            if e.status_code < 500:
                raise RuntimeError(f"Anthropic API client error (4xx): {e}") from e
            logger.warning(f"API server error (attempt {attempt}/{MAX_RETRIES}): {e}")

        # Exponential backoff
        if attempt < MAX_RETRIES:
            delay = INITIAL_BACKOFF * (2 ** (attempt - 1))
            logger.info(f"Retrying in {delay:.1f}s...")
            time.sleep(delay)

    raise RuntimeError(
        f"Anthropic call failed after {MAX_RETRIES} retries. "
        f"Last error: {last_error}"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_llm_response(
    prompt: str,
    system_prompt: Optional[str] = None,
    temperature: Optional[float] = None,
) -> str:
    """Call the LLM with retry logic and return the stripped response.

    Automatically detects the available API key and routes to the correct
    provider (OpenAI, DeepSeek, or Anthropic).

    Args:
        prompt: The user prompt to send.
        system_prompt: Optional system-level instruction (default None).
        temperature: Sampling temperature. Uses config.llm_temperature (0.0) if None.

    Returns:
        Stripped LLM response string.

    Raises:
        RuntimeError: If no API key is found, or all retries are exhausted.
        ValueError: If prompt is empty or whitespace-only.

    Example:
        >>> get_llm_response("Say 'hello'")
        'hello'
    """
    if not prompt or not prompt.strip():
        raise ValueError("prompt must not be empty")

    if temperature is None:
        temperature = config.llm_temperature

    creds = _resolve_credentials()

    if creds["provider"] == "openai":
        return _call_openai(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            api_key=creds["api_key"],
            base_url=creds["base_url"],
        )
    elif creds["provider"] == "anthropic":
        return _call_anthropic(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            api_key=creds["api_key"],
        )
    else:
        raise ValueError(f"Unknown provider: {creds['provider']}")


# ---------------------------------------------------------------------------
# Streaming API
# ---------------------------------------------------------------------------

def get_llm_response_stream(
    prompt: str,
    system_prompt: Optional[str] = None,
    temperature: Optional[float] = None,
    callback: Optional[callable] = None,
):
    """Streaming LLM call — yields tokens one at a time via generator.

    Supports OpenAI / DeepSeek (OpenAI-compatible) with stream=True.
    Anthropic streaming is NOT supported in v0.3 (returns non-streaming fallback).

    Args:
        prompt: The user prompt to send.
        system_prompt: Optional system-level instruction.
        temperature: Sampling temperature. Uses config.llm_temperature if None.
        callback: Optional callable(event_type, token) called on each token.
                  event_type is always "generate" for token events.

    Yields:
        Individual token strings from the LLM response.

    Raises:
        RuntimeError: If no API key is found or streaming fails mid-stream.

    Example:
        >>> for token in get_llm_response_stream("Say hello"):
        ...     print(token, end="")
    """
    if not prompt or not prompt.strip():
        raise ValueError("prompt must not be empty")

    if temperature is None:
        temperature = config.llm_temperature

    creds = _resolve_credentials()

    if creds["provider"] != "openai":
        # Anthropic: fall back to non-streaming, yield full response at once
        logger.warning("Streaming not supported for Anthropic; yielding full response.")
        full = get_llm_response(prompt, system_prompt, temperature)
        if callback:
            callback("generate", full)
        yield full
        return

    # OpenAI / DeepSeek streaming
    import openai

    client_kwargs = {"api_key": creds["api_key"]}
    if creds.get("base_url"):
        client_kwargs["base_url"] = creds["base_url"]
    client = openai.OpenAI(**client_kwargs)

    model = config.llm_model
    max_tokens = config.llm_max_tokens

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    try:
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            stream_options={"include_usage": False},
        )

        for chunk in stream:
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    token = delta.content
                    if callback:
                        try:
                            callback("generate", token)
                        except Exception:
                            pass  # Callback failure should not break streaming
                    yield token

    except openai.APIError as e:
        logger.error(f"Streaming API error: {e}")
        raise RuntimeError(f"LLM streaming failed: {e}") from e
    except Exception as e:
        logger.error(f"Streaming unexpected error: {e}")
        raise RuntimeError(f"LLM streaming interrupted: {e}") from e


# ---------------------------------------------------------------------------
# Module self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print("llm_client module test")
    print("=" * 60)

    # Test 1: API key detection
    print("\n[Test 1] API key detection...")
    try:
        creds = _resolve_credentials()
        print(f"  [OK] Provider: {creds['provider']}")
        print(f"  [OK] API key: {creds['api_key'][:8]}... (length={len(creds['api_key'])})")
    except RuntimeError as e:
        print(f"  [WARN] {e}")
        print("  Set DEEPSEEK_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY to run the live test.")

    # Test 2: Live LLM call (only if key is available)
    print("\n[Test 2] Live LLM call...")
    try:
        response = get_llm_response("Say 'hello' in lowercase.", system_prompt="Respond with only one word, no punctuation.")
        print(f"  [OK] Response: '{response}'")
        if "hello" in response.lower():
            print("  [OK] Response contains 'hello'")
    except RuntimeError as e:
        print(f"  [WARN] {e}")

    # Test 3: Empty prompt validation
    print("\n[Test 3] Empty prompt validation...")
    try:
        get_llm_response("")
        print("  [FAIL] Should have raised ValueError")
    except ValueError as e:
        print(f"  [OK] ValueError: {e}")

    print("\n" + "=" * 60)
    print("llm_client self-test complete.")
