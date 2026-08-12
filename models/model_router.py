"""
Model selection strategy.

We deliberately use TWO different models for two different sub-tasks rather
than one model for everything:

  Sub-task                          Model (Provider)              Why
  ---------------------------------------------------------------------------------------------
  Intent routing / cheap             Llama 3.1 8B Instant (Groq)   Groq's LPU inference is
  classification, vision labelling                                extremely low-latency
  normalisation, reflection critique                               (~fractions of a second) and
                                                                    near-free per token. This task
                                                                    is a short classification /
                                                                    formatting job that does not
                                                                    need deep reasoning, so paying
                                                                    for a larger model would be
                                                                    wasted latency and cost.

  Deep reasoning / final synthesis   DeepSeek V4 Flash             The final user-facing answer
  (combining vision result + RAG     model via OpenRouter          must combine multiple retrieved
  context into a coherent, cited                                  facts, reason about which are
  explanation)                                                    relevant, and write fluent,
                                                                    well-structured prose. Higher
                                                                    reasoning quality justifies the
                                                                    higher cost/latency here because
                                                                    it is the answer the user reads.

See README.md for the full comparison table (cost/latency/context/quality).

Both providers are called through a single thin wrapper (`call_model`) so the
rest of the codebase does not need to know which provider/model is behind a
given "role" (ROUTER vs SYNTHESIS) — this also makes it trivial to swap models
by editing MODEL_CONFIG below without touching agent code.
"""
import os
import requests

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


class ProviderAuthenticationError(RuntimeError):
    """Raised when a configured model-provider credential is rejected."""


def _api_key(name: str) -> str:
    """Read the latest credential after Streamlit has loaded its secrets."""
    return os.environ.get(name, "").strip()

MODEL_CONFIG = {
    "ROUTER": {
        "provider": "groq",
        "model": "llama-3.1-8b-instant",
    },
    "REFLECTION": {
        "provider": "groq",
        "model": "llama-3.1-8b-instant",
    },
    "SYNTHESIS": {
        "provider": "openrouter",
        # Claude 3.5 Sonnet was retired from OpenRouter.  This is a current,
        # low-cost text model suitable for the grounded synthesis step.
        "model": "deepseek/deepseek-v4-flash-0731",
    },
    "VISION": {
        "provider": "openrouter",
        "model": "openai/gpt-4o-mini",
        # Why gpt-4o-mini here rather than reusing the synthesis model from
        # SYNTHESIS: this sub-task is "look at one image, return a short
        # structured JSON label" — a narrow, high-volume-friendly job where
        # gpt-4o-mini's strong OCR/vision grounding at a fraction of the
        # cost/latency of a larger model is the better fit. The expensive
        # model is reserved for the sub-task that actually needs deep
        # reasoning: writing the final synthesized answer.
    },
}


def call_model(role: str, system_prompt: str, user_prompt: str, max_tokens: int = 500) -> str:
    """
    role: one of "ROUTER", "REFLECTION", "SYNTHESIS" (see MODEL_CONFIG above).
    Returns the assistant's text response as a plain string.
    """
    cfg = MODEL_CONFIG[role]

    if cfg["provider"] == "groq":
        groq_api_key = _api_key("GROQ_API_KEY")
        if not groq_api_key:
            return _stub_response(role, user_prompt)
        headers = {"Authorization": f"Bearer {groq_api_key}"}
        body = {
            "model": cfg["model"],
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.2,
        }
        resp = requests.post(GROQ_URL, headers=headers, json=body, timeout=30)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    elif cfg["provider"] == "openrouter":
        openrouter_api_key = _api_key("OPENROUTER_API_KEY")
        if not openrouter_api_key:
            return _stub_response(role, user_prompt)
        headers = {
            "Authorization": f"Bearer {openrouter_api_key}",
            "HTTP-Referer": "https://streamlit.io",
            "X-Title": "Asian Currency Intelligence Assistant",
        }
        body = {
            "model": cfg["model"],
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.3,
        }
        resp = requests.post(OPENROUTER_URL, headers=headers, json=body, timeout=60)
        if resp.status_code == 401:
            return _stub_response(role, user_prompt)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    raise ValueError(f"Unknown provider for role {role}")


def call_vision_model(system_prompt: str, user_prompt: str, image_base64: str,
                       mime_type: str = "image/jpeg", max_tokens: int = 300) -> str:
    """
    Calls the VISION role model (OpenRouter, multimodal) with an image + text
    prompt. Returns the assistant's text response as a plain string (expected
    to be a JSON object — parsing happens in the caller, agents/vision_agent.py).
    """
    cfg = MODEL_CONFIG["VISION"]
    openrouter_api_key = _api_key("OPENROUTER_API_KEY")
    if not openrouter_api_key:
        return _stub_response("VISION", user_prompt)

    headers = {
        "Authorization": f"Bearer {openrouter_api_key}",
        "HTTP-Referer": "https://streamlit.io",
        "X-Title": "Asian Currency Intelligence Assistant",
    }
    body = {
        "model": cfg["model"],
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{image_base64}"},
                    },
                ],
            },
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    resp = requests.post(OPENROUTER_URL, headers=headers, json=body, timeout=60)
    if resp.status_code == 401:
        raise ProviderAuthenticationError(
            "OpenRouter rejected OPENROUTER_API_KEY (HTTP 401). "
            "Create a new OpenRouter API key and update .streamlit/secrets.toml."
        )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def _stub_response(role: str, user_prompt: str) -> str:
    """
    Used only when no API key is configured (e.g. local demo without secrets),
    so the app remains runnable / screenshot-able without live keys.
    Replace by setting GROQ_API_KEY / OPENROUTER_API_KEY in Streamlit secrets.
    """
    return (
        f"[DEMO MODE - no API key set for role={role}] "
        f"Would call {MODEL_CONFIG[role]['model']} with prompt: {user_prompt[:120]}..."
    )
