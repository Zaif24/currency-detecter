"""
Streamlit app — Asian Currency Intelligence Assistant.
Entry point for Streamlit Community Cloud deployment.
"""
import os
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Asian Currency Intelligence Assistant", page_icon="💱", layout="wide")

# --- secrets -> env vars (Streamlit Cloud injects st.secrets; local dev can use env vars) ---
for key in ("GROQ_API_KEY", "OPENROUTER_API_KEY"):
    if key in st.secrets:
        os.environ[key] = st.secrets[key]

# Load agents only after Streamlit secrets have been copied into the process
# environment, so provider configuration cannot see stale credentials.
from agents.orchestrator import OrchestratorAgent
from agents.protocol import MessageBus

st.title("💱 Asian Currency Intelligence Assistant")


if not os.environ.get("GROQ_API_KEY") or not os.environ.get("OPENROUTER_API_KEY"):
    st.info(
        "Running in **demo mode** — no GROQ_API_KEY / OPENROUTER_API_KEY found in "
        "Streamlit secrets. Model calls will return placeholder text. Add your keys "
        "in `.streamlit/secrets.toml` (local) or the app's Secrets settings (Cloud) "
        "to enable live responses.",
        icon="ℹ️",
    )

col1, col2 = st.columns([1, 1.4])

with col1:
    st.subheader("1. Upload a banknote image")
    uploaded = st.file_uploader("Image file", type=["png", "jpg", "jpeg"])
    user_query = st.text_input(
        "2. What do you want to know?",
        placeholder="e.g. What security features does this note have?",
    )
    run_btn = st.button("Run agents", type="primary", disabled=uploaded is None)

    if uploaded:
        st.image(uploaded, caption="Uploaded image", use_container_width=True)

with col2:
    st.subheader("Result")
    if run_btn and uploaded:
        image = Image.open(uploaded)
        bus = MessageBus()
        orchestrator = OrchestratorAgent()

        with st.spinner("Running multi-agent pipeline..."):
            result = orchestrator.run(image, user_query or "Identify this currency and explain it.", bus)

        if result["country"] == "unknown":
            st.warning(
                f"Vision model could not confidently identify this note "
                f" Try a clearer, "
                f"well-lit, front-facing photo of the whole note."
            )
        else:
            st.success(f"**{result['denomination']} — {result['country'].replace('_', ' ').title()}** "
                       f"(confidence: {result['confidence']:.0%})")

        vs = result.get("vision_source", "")
        if vs.startswith("heuristic_fallback"):
            if vs.endswith("AUTH_FAILED"):
                st.error(
                    "OpenRouter rejected `OPENROUTER_API_KEY` (HTTP 401). "
                    "The offline fallback was used; replace the key in "
                    "`.streamlit/secrets.toml` and restart the app."
                )
            st.caption("⚠️ No OPENROUTER_API_KEY set — used the offline colour-heuristic "
                       "fallback, not the real vision model. Results are unreliable on real photos.")
        elif vs.startswith("vision_llm"):
            st.caption(f"✅ Identified via real vision model: `{vs.split(':', 1)[1]}`")

        st.markdown("#### Answer")
        st.write(result["answer"])

    else:
        st.write("Upload an image and click **Run agents** to see the result here.")

st.divider()
