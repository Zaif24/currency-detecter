"""
OrchestratorAgent — the top-level agent the Streamlit app talks to.

Design patterns demonstrated here:

1. ROUTER
   `route_intent()` classifies the user's request into one of:
     IDENTIFY_ONLY | IDENTIFY_AND_EXPLAIN | COMPARE
   using the cheap/fast Groq model (see models/model_router.py, role=ROUTER).

2. PLANNING / TASK DECOMPOSITION
   `plan()` turns the routed intent into an ordered list of subtasks (e.g.
   "call VisionAgent", "call ResearchAgent", "synthesize answer") before any
   execution happens.

3. ORCHESTRATOR-WORKER
   `run()` executes the plan by dispatching structured AgentMessages to the
   VisionAgent and ResearchAgent "workers" via the shared MessageBus (see
   agents/protocol.py), collecting their replies, and passing them to the
   SYNTHESIS model to produce the final answer.

4. REFLECTION / SELF-CRITIQUE
   `reflect_and_revise()` runs a second, cheap model pass that critiques the
   draft synthesis answer against the retrieved context (checking for
   unsupported claims / missing grounding) and revises it if needed, before
   it is returned to the user.

(A fifth pattern, ReAct, is implemented inside ResearchAgent — see
agents/research_agent.py — and a sixth, tool-use, inside VisionAgent — see
agents/vision_agent.py.)
"""
from PIL import Image

from agents.protocol import AgentMessage, MessageBus
from agents.vision_agent import VisionAgent
from agents.research_agent import ResearchAgent
from models.model_router import call_model


class OrchestratorAgent:
    name = "OrchestratorAgent"

    def __init__(self):
        self.vision_agent = VisionAgent()
        self.research_agent = ResearchAgent()

    # ---------- 1. ROUTER ----------
    def route_intent(self, user_query: str, bus: MessageBus) -> str:
        system = (
            "Classify the user's request into exactly one label: "
            "IDENTIFY_ONLY, IDENTIFY_AND_EXPLAIN, or COMPARE. "
            "Reply with only the label, nothing else."
        )
        label = call_model("ROUTER", system, user_query, max_tokens=10).strip()
        valid = {"IDENTIFY_ONLY", "IDENTIFY_AND_EXPLAIN", "COMPARE"}
        if label not in valid:
            label = "IDENTIFY_AND_EXPLAIN"  # safe default
        bus.send(AgentMessage(
            sender="RouterModel(Groq/Llama-3.1-8B)",
            receiver=self.name,
            intent="ROUTE_DECISION",
            payload={"label": label, "user_query": user_query},
        ))
        return label

    # ---------- 2. PLANNING / TASK DECOMPOSITION ----------
    def plan(self, label: str) -> list:
        base_plan = ["IDENTIFY_CURRENCY"]
        if label in ("IDENTIFY_AND_EXPLAIN", "COMPARE"):
            base_plan.append("RETRIEVE_CONTEXT")
            base_plan.append("SYNTHESIZE_ANSWER")
            base_plan.append("REFLECT_AND_REVISE")
        else:
            base_plan.append("SYNTHESIZE_ANSWER")  # short identify-only answer
        return base_plan

    # ---------- 3. ORCHESTRATOR-WORKER ----------
    def run(self, image: Image.Image, user_query: str, bus: MessageBus) -> dict:
        label = self.route_intent(user_query, bus)
        subtasks = self.plan(label)

        # --- Worker 1: VisionAgent ---
        vision_request = AgentMessage(
            sender=self.name, receiver=self.vision_agent.name,
            intent="IDENTIFY_CURRENCY", payload={"user_query": user_query},
        )
        bus.send(vision_request)
        vision_reply = self.vision_agent.handle(vision_request, bus, image)
        country = vision_reply.payload["country"]
        denomination = vision_reply.payload["denomination"]
        confidence = vision_reply.payload["confidence"]
        vision_source = vision_reply.payload.get("source", "unknown")

        chunks = []
        # Skip RAG retrieval if the vision step couldn't confidently identify
        # a supported country — there's nothing meaningful to filter/retrieve.
        if "RETRIEVE_CONTEXT" in subtasks and country != "unknown":
            # --- Worker 2: ResearchAgent ---
            research_request = AgentMessage(
                sender=self.name, receiver=self.research_agent.name,
                intent="RETRIEVE_CONTEXT",
                payload={"country": country, "user_query": user_query},
            )
            bus.send(research_request)
            research_reply = self.research_agent.handle(research_request, bus)
            chunks = research_reply.payload["chunks"]

        # --- Synthesis ---
        draft_answer = self._synthesize(country, denomination, confidence, chunks, user_query, label, bus)

        # --- 4. Reflection ---
        final_answer = draft_answer
        if "REFLECT_AND_REVISE" in subtasks:
            final_answer = self.reflect_and_revise(draft_answer, chunks, bus)

        return {
            "country": country,
            "denomination": denomination,
            "confidence": confidence,
            "vision_source": vision_source,
            "route_label": label,
            "plan": subtasks,
            "retrieved_chunks": chunks,
            "answer": final_answer,
        }

    def _synthesize(self, country, denomination, confidence, chunks, user_query, label, bus: MessageBus) -> str:
        context_block = "\n\n".join(
            f"[Source: {c['source']}]\n{c['text']}" for c in chunks
        ) or "(no retrieved context — identification only)"

        system = (
            "You are a helpful currency-identification assistant for Asian banknotes. "
            "Use ONLY the provided context for factual claims; do not invent facts. "
            "Cite the source filename in brackets after claims drawn from context. "
            "Be concise and well-organized."
        )
        user = (
            f"Detected currency: {denomination} ({country}), confidence {confidence}.\n"
            f"User request: {user_query}\n\nRetrieved context:\n{context_block}\n\n"
            f"Write the final answer for the user."
        )
        answer = call_model("SYNTHESIS", system, user, max_tokens=500)
        bus.send(AgentMessage(
            sender="SynthesisModel(OpenRouter/Claude-3.5-Sonnet)",
            receiver=self.name,
            intent="DRAFT_ANSWER",
            payload={"answer": answer},
        ))
        return answer

    # ---------- 4. REFLECTION / SELF-CRITIQUE ----------
    def reflect_and_revise(self, draft_answer: str, chunks: list, bus: MessageBus) -> str:
        context_block = "\n\n".join(c["text"] for c in chunks) or "(none)"
        system = (
            "You are a strict fact-checking critic. Compare the DRAFT ANSWER to the "
            "CONTEXT. If the draft contains claims not supported by the context, or is "
            "unclear, rewrite it to be fully grounded and clear. If it is already fine, "
            "return it unchanged. Reply with ONLY the final answer text, no commentary."
        )
        user = f"CONTEXT:\n{context_block}\n\nDRAFT ANSWER:\n{draft_answer}"
        revised = call_model("REFLECTION", system, user, max_tokens=500)
        bus.send(AgentMessage(
            sender="ReflectionModel(Groq/Llama-3.1-8B)",
            receiver=self.name,
            intent="REVISED_ANSWER",
            payload={"revised_answer": revised},
        ))
        return revised
