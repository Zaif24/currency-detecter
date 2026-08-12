# Asian Currency Intelligence Assistant

**IT41043 — Intelligent Systems (Agentic AI) — Assignment Submission**

A multi-agent AI system that identifies Asian banknotes from an image and
answers questions about them (history, security features, travel tips) using
a retrieval-augmented generation pipeline grounded in a domain-specific
knowledge base. Built to satisfy the Option A / Option B brief as a hybrid:
a practical tool (currency identification) supported by structured,
citable research content.

**Live app:** _add your Streamlit Cloud URL here after deployment_
**Repo:** _add your GitHub repo URL here_

---

## 1. Problem & Project Type

Travelers, students, and researchers dealing with multiple Asian currencies
often can't quickly tell which country/denomination a note belongs to, or
recall its security features, history, or cultural handling norms. This
project (**Option A — real-world problem**) builds an agentic assistant that
takes a photo of a banknote and returns a grounded, cited explanation.

---

## 2. System Architecture

See [`docs/architecture_diagram.md`](docs/architecture_diagram.md) for the
full component diagram and [`docs/sequence_diagram.md`](docs/sequence_diagram.md)
for the agent-to-agent message flow.

```
User (Streamlit)
   └─> OrchestratorAgent  (router, planner, orchestrator-worker, reflection)
         ├─> VisionAgent        (tool-use pattern)      -> currency_classifier_tool()
         ├─> ResearchAgent      (ReAct pattern)          -> RAG pipeline (FAISS)
         ├─> ROUTER model       (Groq / Llama 3.1 8B)
         ├─> SYNTHESIS model    (OpenRouter / DeepSeek V4 Flash)
         └─> REFLECTION model   (Groq / Llama 3.1 8B)
```

---

## 3. Agentic Design Patterns (≥3 required — 5 implemented)

| # | Pattern | Where it appears |
|---|---------|-------------------|
| 1 | **Router** | `agents/orchestrator.py :: route_intent()` — classifies the user request into `IDENTIFY_ONLY` / `IDENTIFY_AND_EXPLAIN` / `COMPARE` using the cheap Groq model before any other work happens. |
| 2 | **Planning / task decomposition** | `agents/orchestrator.py :: plan()` — turns the routed label into an explicit ordered subtask list (e.g. `["IDENTIFY_CURRENCY", "RETRIEVE_CONTEXT", "SYNTHESIZE_ANSWER", "REFLECT_AND_REVISE"]`) before execution. |
| 3 | **Orchestrator–worker** | `agents/orchestrator.py :: run()` — the orchestrator dispatches structured `AgentMessage`s to the `VisionAgent` and `ResearchAgent` "workers" and collects their replies. |
| 4 | **Tool-use** | `agents/vision_agent.py :: VisionAgent.handle()` — calls the external `currency_classifier_tool()` and interprets its structured JSON-like output rather than reasoning about pixels itself. |
| 5 | **ReAct (reason → act → observe)** | `agents/research_agent.py :: ResearchAgent.handle()` — loops: reasons about which sub-question to ask next, acts by calling `retrieve()`, observes the returned chunks, and stops early once enough distinct context is gathered (max 3 steps). |
| 6 | **Reflection / self-critique** | `agents/orchestrator.py :: reflect_and_revise()` — a second, cheap model pass checks the draft synthesis answer against the retrieved context for unsupported claims and rewrites it if needed. |

---

## 4. Agent-to-Agent Communication

Two primary agents — `VisionAgent` and `ResearchAgent` — communicate with the
`OrchestratorAgent` via a custom, structured message protocol defined in
[`agents/protocol.py`](agents/protocol.py), inspired by MCP/A2A conventions:
every message is a typed `AgentMessage` (sender, receiver, intent, payload,
trace_id, timestamp, in_reply_to) rather than a free-text string. All
messages are recorded on a shared `MessageBus` and are visible in the
Streamlit UI under **"Agent-to-agent message trace"** for full transparency.

See [`docs/sequence_diagram.md`](docs/sequence_diagram.md) for the full
message flow diagram.

---

## 5. Model Selection Strategy

Two different models are deliberately used for two different sub-tasks —
**not** the same model for everything:

| Sub-task | Model (Provider) | Why chosen |
|---|---|---|
| Intent routing, reflection/self-critique | **Llama 3.1 8B Instant (Groq)** | Groq's LPU inference gives sub-second latency and near-zero per-token cost. Routing and critique are short classification/checking jobs that don't need deep reasoning — a larger model here would only add latency and cost without improving the outcome. |
| Banknote image classification (country + denomination) | **GPT-4o-mini (OpenRouter)** | A narrow, high-volume "look at one image, return structured JSON" job. GPT-4o-mini has strong OCR/vision grounding at a fraction of the cost/latency of a larger multimodal model — no need to pay for deep reasoning here, just accurate visual reading. |
| Final answer synthesis (combining vision result + retrieved RAG context into fluent, cited prose) | **Claude 3.5 Sonnet-class (OpenRouter)** | This is the answer the user actually reads: it must weigh multiple retrieved facts, avoid unsupported claims, and produce well-structured writing. Higher reasoning/writing quality justifies the higher cost and latency specifically for this one call per request. |

Three models across two providers are used, each deliberately matched to its
sub-task's actual difficulty rather than defaulting to one model everywhere.

Configuration lives in one place, [`models/model_router.py`](models/model_router.py)
(`MODEL_CONFIG` dict), so models can be swapped without touching agent code.
If no API key is configured, calls return a clearly labelled demo-mode string
so the app remains runnable for screenshots/testing without live keys.

---

## 6. RAG Integration

**Corpus:** 25 original markdown documents across 5 countries (Sri Lanka,
India, Japan, China, Thailand) × 5 topics each (overview, denominations,
security features, history, travel tips) — see `data/knowledge_base/`.
Generated by [`scripts/generate_kb.py`](scripts/generate_kb.py).

**Chunking strategy:** paragraph-aware fixed-size chunking (`rag/chunking.py`),
packing paragraphs into ≤800-character chunks with 100-character overlap
between chunks. Most of this project's source documents are short enough to
remain a single coherent chunk, which is intentional — further splitting a
150–300 word single-topic document would hurt retrieval precision, not help it.

**Embedding model:** `sentence-transformers/all-MiniLM-L6-v2` (384-dim,
~80MB, free, runs on CPU) is the primary embedder. Because this repo was
partly developed in a network-restricted sandbox with no access to
huggingface.co, [`rag/embedding_backend.py`](rag/embedding_backend.py)
implements an **automatic fallback to a TF-IDF + SVD embedder** (scikit-learn)
so the pipeline still runs end-to-end offline. Streamlit Cloud has outbound
internet access, so the primary MiniLM backend is expected to load there;
whichever backend built the index is recorded in `vectorstore/embedder.json`
so this is never silently ambiguous.

**Vector store:** FAISS (`IndexFlatIP` over L2-normalized vectors = cosine
similarity), chosen over Chroma for this project because it's a single
dependency with no external server process, and persists trivially as a flat
file for Streamlit Cloud's ephemeral filesystem.

**Retrieval evaluation:** 5 sample queries were run against the built index
and manually assessed for relevance — see
[`docs/retrieval_evaluation.md`](docs/retrieval_evaluation.md) for the full
results, scores, and commentary (including an honest discussion of where
TF-IDF retrieval underperformed and why the semantic embedder is expected to
fix it).

---

## 7. Vision Component

`agents/vision_agent.py :: currency_classifier_tool()` sends the uploaded
photo to a real multimodal LLM (GPT-4o-mini via OpenRouter, `role="VISION"`
in `models/model_router.py`) with a strict JSON-output prompt, and parses the
response into `{country, denomination, confidence}`. No training or bundled
image dataset is required — this was chosen over training a CNN specifically
*because* no licensed banknote dataset ships with this repo (see
[`scripts/download_datasets.md`](scripts/download_datasets.md) for why, and
for the CNN-training alternative if you'd rather go that route).

**Fallback:** if `OPENROUTER_API_KEY` is not configured, the tool falls back
to a rough colour-heuristic (dominant-colour matching against Asian
banknotes' real colour-coding) purely so the pipeline still *runs* offline.
This fallback is **not reliable on real photos** — it averages colour across
the whole image (hand, background, lighting included), which is why an
earlier version of this repo misclassified a real Rs.1000 note as a Thai
฿20. It exists only so the app doesn't crash without a key; the UI clearly
labels which path produced a given result (`vision_llm:...` vs
`heuristic_fallback_NO_API_KEY`), and `country="unknown"` (low confidence) is
returned rather than a confident guess when the model genuinely can't tell.

---

## 8. Repository Structure

```
asian-currency-agent/
├── app.py                        # Streamlit entrypoint
├── agents/
│   ├── protocol.py                # A2A message schema + MessageBus
│   ├── vision_agent.py            # Tool-use pattern
│   ├── research_agent.py          # ReAct pattern
│   └── orchestrator.py            # Router, planning, orchestrator-worker, reflection
├── rag/
│   ├── chunking.py
│   ├── embedding_backend.py       # MiniLM primary / TF-IDF fallback
│   ├── ingest.py                  # Build the FAISS index
│   └── retriever.py
├── models/
│   └── model_router.py            # Groq + OpenRouter wrapper, MODEL_CONFIG
├── data/knowledge_base/           # 25 docs, 5 countries (the RAG corpus)
├── vectorstore/                   # FAISS index + metadata (generated, gitignored)
├── docs/
│   ├── architecture_diagram.md
│   ├── sequence_diagram.md
│   └── retrieval_evaluation.md
├── scripts/
│   ├── generate_kb.py
│   ├── eval_retrieval.py
│   └── download_datasets.md
├── tests/test_retrieval.py
├── .streamlit/secrets.toml.example
├── requirements.txt
└── .gitignore
```

---

## 9. Setup & Local Run

```bash
git clone <your-repo-url>
cd asian-currency-agent
pip install -r requirements.txt

# 1. Build the vector store (one-time)
python -m rag.ingest

# 2. (optional) run the retrieval evaluation
PYTHONPATH=. python3 scripts/eval_retrieval.py

# 3. add your API keys for live model calls
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# edit .streamlit/secrets.toml with your GROQ_API_KEY / OPENROUTER_API_KEY

# 4. run tests
PYTHONPATH=. pytest tests/ -v

# 5. launch the app
streamlit run app.py
```

## 10. Deployment (Streamlit Community Cloud)

1. Push this repo to GitHub (secrets are excluded via `.gitignore` — verify
   `.streamlit/secrets.toml` is **not** committed).
2. On [share.streamlit.io](https://share.streamlit.io), create a new app
   pointing at this repo, branch `main`, file `app.py`.
3. In the app's **Settings → Secrets**, paste:
   ```toml
   GROQ_API_KEY = "..."
   OPENROUTER_API_KEY = "..."
   ```
4. Deploy. On first boot, the app needs the vector store to exist — either
   commit a pre-built `vectorstore/` (small enough here — 25 chunks) or add a
   one-line boot check that runs `rag.ingest.build_index()` if
   `vectorstore/index.faiss` is missing. Keep the app live for at least two
   weeks after the submission deadline.

## 11. Secrets Management

- API keys are read via `st.secrets` (Streamlit Cloud) and copied into
  `os.environ` at app startup (`app.py`), never hardcoded.
- `.streamlit/secrets.toml` (the real file with actual keys) is listed in
  `.gitignore` and must never be committed — only the `.example` template is
  version-controlled.
- If no keys are present, `models/model_router.py` returns a clearly labelled
  demo-mode string instead of failing, so the repo is still inspectable/runnable
  without live credentials.
