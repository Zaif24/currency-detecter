"""
VisionAgent — identifies the country/denomination of a banknote from an
uploaded image.

Design pattern demonstrated: TOOL-USE.
  The agent does not "look at" the image itself; it calls a well-defined
  tool (`currency_classifier_tool`) and interprets that tool's structured
  output. The tool itself now wraps a REAL multimodal LLM call (OpenRouter,
  role="VISION" in models/model_router.py) rather than a heuristic — see
  "Fallback" below for the one case where it still uses a heuristic.

Why an LLM call and not a trained CNN:
  This coursework repo ships without a licensed banknote image dataset (see
  scripts/download_datasets.md for why), so there was nothing to train a
  classifier on. Calling a vision-capable LLM avoids needing a dataset or
  training step at all, and is genuinely accurate on real photos — unlike
  the earlier placeholder, which only averaged pixel colour across the whole
  image (hand, background, lighting included) and was consequently unreliable
  on real-world photos. This is "Option B" from scripts/download_datasets.md.

Fallback:
  If OPENROUTER_API_KEY is not configured, the tool falls back to the old
  colour-heuristic so the pipeline is still runnable without any API key
  (e.g. for offline demoing) — but this path is clearly labelled as a
  fallback in the returned payload (`source` field) so it's never confused
  with a real classification.
"""
import base64
import io
import json
import re

from PIL import Image
import numpy as np

from agents.protocol import AgentMessage, MessageBus
from models.model_router import (
    ProviderAuthenticationError,
    _api_key,
    call_vision_model,
)

VALID_COUNTRIES = ["sri_lanka", "india", "japan", "china", "thailand"]

VISION_SYSTEM_PROMPT = (
    "You are a banknote identification system for Asian currencies. "
    "You will be shown a photo of a banknote (it may be held in a hand, at an "
    "angle, or partially visible). Identify the issuing country and the "
    "denomination as precisely as you can from any visible numerals, script, "
    "portraits, or design elements.\n\n"
    "Respond with ONLY a single JSON object, no other text, in exactly this shape:\n"
    '{"country": "<one of: sri_lanka, india, japan, china, thailand, unknown>", '
    '"denomination": "<e.g. Rs.1000, ₹500, ¥1000, ¥100, ฿20>", '
    '"confidence": <float 0.0-1.0>}\n\n'
    "If you cannot identify it confidently, use \"unknown\" for country and your "
    "best guess (or \"unknown\") for denomination, with a low confidence value. "
    "Never fabricate a confident answer for a banknote you cannot actually read."
)


def _image_to_base64_jpeg(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _parse_json_response(raw: str) -> dict:
    """Robust-ish JSON extraction in case the model wraps JSON in extra text."""
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    return {"country": "unknown", "denomination": "unknown", "confidence": 0.0}


# ---------- Fallback heuristic (only used when no OpenRouter key is set) ----------
_COLOR_REFERENCE = {
    (34, 139, 34): ("sri_lanka", "Rs.20"),
    (128, 0, 128): ("sri_lanka", "Rs.5000"),
    (255, 140, 0): ("sri_lanka", "Rs.500"),
    (65, 105, 225): ("sri_lanka", "Rs.1000"),
    (255, 215, 0): ("india", "₹200"),
    (176, 196, 222): ("india", "₹100"),
    (0, 0, 139): ("japan", "¥1000"),
    (139, 69, 19): ("japan", "¥10000"),
    (178, 34, 34): ("china", "¥100"),
    (0, 100, 0): ("china", "¥50"),
    (46, 139, 87): ("thailand", "฿20"),
    (75, 0, 130): ("thailand", "฿500"),
}


def _heuristic_fallback(image: Image.Image) -> dict:
    img = image.convert("RGB").resize((64, 64))
    arr = np.array(img).reshape(-1, 3)
    dominant = arr.mean(axis=0)
    best_country, best_denom, best_dist = None, None, float("inf")
    for ref_color, (country, denom) in _COLOR_REFERENCE.items():
        dist = np.linalg.norm(dominant - np.array(ref_color))
        if dist < best_dist:
            best_dist = dist
            best_country, best_denom = country, denom
    confidence = max(0.10, min(0.5, 1 - (best_dist / 441.7)))  # capped low: this is a rough guess
    return {
        "country": best_country,
        "denomination": best_denom,
        "confidence": round(confidence, 2),
        "source": "heuristic_fallback_NO_API_KEY",
    }
# ------------------------------------------------------------------------------


def currency_classifier_tool(image: Image.Image) -> dict:
    """
    TOOL: identifies country + denomination from a banknote photo.
    Output: {"country": str, "denomination": str, "confidence": float, "source": str}
    """
    if not _api_key("OPENROUTER_API_KEY"):
        return _heuristic_fallback(image)

    image_b64 = _image_to_base64_jpeg(image)
    try:
        raw = call_vision_model(
            system_prompt=VISION_SYSTEM_PROMPT,
            user_prompt="Identify this banknote.",
            image_base64=image_b64,
        )
    except ProviderAuthenticationError:
        result = _heuristic_fallback(image)
        result["source"] = "heuristic_fallback_OPENROUTER_AUTH_FAILED"
        return result
    parsed = _parse_json_response(raw)

    country = str(parsed.get("country", "unknown")).strip().lower()
    if country not in VALID_COUNTRIES:
        country = "unknown"
    denomination = str(parsed.get("denomination", "unknown")).strip()
    try:
        confidence = float(parsed.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    return {
        "country": country,
        "denomination": denomination,
        "confidence": round(confidence, 2),
        "source": "vision_llm:openai/gpt-4o-mini",
    }


class VisionAgent:
    name = "VisionAgent"

    def handle(self, message: AgentMessage, bus: MessageBus, image: Image.Image) -> AgentMessage:
        assert message.intent == "IDENTIFY_CURRENCY"

        tool_result = currency_classifier_tool(image)

        reply = AgentMessage(
            sender=self.name,
            receiver=message.sender,
            intent="CURRENCY_IDENTIFIED",
            payload=tool_result,
            in_reply_to=message.trace_id,
        )
        bus.send(reply)
        return reply
