# Sourcing a Real Banknote Image Dataset

This repo intentionally does **not** bundle a real banknote image dataset:
currency imagery carries usage/reproduction restrictions that vary by country
(central banks generally permit editorial/educational use but restrict
reproduction that could facilitate counterfeiting), and no dataset was
available to fetch automatically in this environment. The `VisionAgent`
therefore ships with a clearly labelled placeholder classifier
(`agents/vision_agent.py`) so the full agent pipeline still runs end-to-end.

To make the vision step "real," pick ONE of the following:

## Option A — Public image dataset + train a small classifier
Good public starting points (verify each dataset's license before use):
- Kaggle: "Indian Currency Notes" datasets (several exist under this name)
- Kaggle: "Bangladeshi Money / Taka" and similar per-country note datasets
- Kaggle: "Currency Recognition" multi-country compilations

Steps:
1. Download a dataset covering the country/countries you need.
2. Fine-tune a small CNN (e.g. MobileNetV2 via `torchvision`/`keras`) to
   classify (country, denomination) from an input image.
3. Replace the body of `currency_classifier_tool()` in
   `agents/vision_agent.py` with a call to your trained model, keeping the
   same return shape: `{"country": str, "denomination": str, "confidence": float}`.

## Option B — Vision-capable LLM via OpenRouter (no training required)
Send the uploaded image (base64) to a vision-capable OpenRouter model (e.g.
`openai/gpt-4o` or `anthropic/claude-3.5-sonnet`) with a prompt asking it to
identify the country and denomination and return JSON. This requires no
dataset or training step at all — only an OpenRouter API call inside
`currency_classifier_tool()`. This is the fastest path to a fully "real"
vision step for a coursework demo.

## Keeping the RAG knowledge base separate
Note the RAG knowledge base (`data/knowledge_base/`, used by `ResearchAgent`)
is independent of whichever vision approach you choose — it is plain-text
currency facts, not images, and is already populated with 25 original
documents across 5 countries (see README for details). You can freely expand
it with more countries/topics without touching the vision pipeline.
