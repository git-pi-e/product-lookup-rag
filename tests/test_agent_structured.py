from app.services.agent import agent_run_structured, AgentServices


def test_agent_structured_smoke(monkeypatch):
    # Stub AgentServices methods to avoid DB/LLM calls
    def fake_tool_query(self, prompt):
        return [
            {"id": 123, "name": "Red Shirt", "retrieval_method": "entities_server_gds"}
        ]

    def fake_tool_similarity_search(self, prompt):
        return []

    def fake_tool_similar_items(self, ids_json):
        return [{"id": 456, "name": "Red T-Shirt"}]

    monkeypatch.setattr(AgentServices, "tool_query", fake_tool_query)
    monkeypatch.setattr(
        AgentServices, "tool_similarity_search", fake_tool_similarity_search
    )
    monkeypatch.setattr(AgentServices, "tool_similar_items", fake_tool_similar_items)

    out = agent_run_structured("Suggest some red clothes for adults")
    assert out["prompt"].startswith("Suggest some red clothes")
    assert "Red Shirt (123)" in out["final_answer"]
    assert "Red T-Shirt (456)" in out["final_answer"] or "Red T-Shirt" in str(
        out["steps"]
    )
