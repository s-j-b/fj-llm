"""Isolate every test's cost-log writes from the real default path
(~/.local/share/fj_llm/costs.jsonl) -- without this, any test that runs a
successful query() writes fake entries into that real, user-facing file."""

import pytest


@pytest.fixture(autouse=True)
def isolate_cost_log(tmp_path, monkeypatch):
    monkeypatch.setenv("FJ_LLM_COST_LOG", str(tmp_path / "costs.jsonl"))
