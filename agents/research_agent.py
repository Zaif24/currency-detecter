"""
ResearchAgent — retrieves grounding context from the domain knowledge base
for a given currency, using the RAG pipeline in rag/retriever.py.

Design pattern demonstrated: ReAct (Reason -> Act -> Observe loop).
  The agent reasons about what sub-questions it needs answered (e.g. security
  features, history, travel tips), ACTS by issuing a retrieval call for each
  sub-question, OBSERVES the returned chunks, and stops once it has enough
  distinct, relevant context (or hits max_steps). This is a lightweight,
  transparent ReAct loop rather than a full LLM-driven ReAct agent, to keep
  the retrieval step cheap, deterministic, and free of extra model calls.
"""
from typing import List, Dict

from rag.retriever import retrieve
from agents.protocol import AgentMessage, MessageBus

SUB_QUESTION_TEMPLATES = [
    "overview and denomination of {country} currency",
    "security features of {country} banknotes",
    "history of the {country} currency",
    "travel tips for using {country} currency",
]


class ResearchAgent:
    name = "ResearchAgent"

    def handle(self, message: AgentMessage, bus: MessageBus, max_steps: int = 3) -> AgentMessage:
        assert message.intent == "RETRIEVE_CONTEXT"
        country = message.payload["country"]
        user_query = message.payload.get("user_query", "")

        collected: List[Dict] = []
        seen_sources = set()
        steps_taken = 0

        # REASON: prioritize the user's own query first, then fall back to
        # generic sub-questions to fill in gaps if the user query was narrow.
        candidate_queries = [user_query] if user_query else []
        candidate_queries += [t.format(country=country) for t in SUB_QUESTION_TEMPLATES]

        for q in candidate_queries:
            if steps_taken >= max_steps:
                break
            # ACT
            results = retrieve(q, k=2, country_filter=country)
            steps_taken += 1
            # OBSERVE
            new_results = [r for r in results if r["source"] not in seen_sources]
            for r in new_results:
                seen_sources.add(r["source"])
                collected.append(r)
            if len(collected) >= 4:  # enough grounding context collected
                break

        reply = AgentMessage(
            sender=self.name,
            receiver=message.sender,
            intent="CONTEXT_RETRIEVED",
            payload={"chunks": collected, "steps_taken": steps_taken, "country": country},
            in_reply_to=message.trace_id,
        )
        bus.send(reply)
        return reply
