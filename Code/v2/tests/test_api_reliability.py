import json

import pytest

from Code.v2.drafter import Drafter
from Code.v2.eval.run import _with_backoff
from Code.v2.router import Router
from Code.v2.verifier import Verifier


class FakeCompletions:
    def __init__(self, contents):
        self.contents = iter(contents)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        content = next(self.contents)
        message = type("Message", (), {"content": content})()
        return type("Response", (), {"choices": [type("Choice", (), {"message": message})()]})()


class FakeClient:
    def __init__(self, contents):
        self.completions = FakeCompletions(contents)
        self.chat = type("Chat", (), {"completions": self.completions})()


def test_router_drafter_and_verifier_set_120_second_timeouts():
    router_client = FakeClient([json.dumps({"label": "SIMPLE", "confidence": 1})])
    Router(client=router_client).classify("What is turmeric?")
    assert router_client.completions.calls[0]["timeout"] == 120

    draft_client = FakeClient([json.dumps({"answer": "I don't know.", "claims": []})])
    Drafter(client=draft_client).draft("What is turmeric?", [], "SIMPLE")
    assert draft_client.completions.calls[0]["timeout"] == 120

    verifier_client = FakeClient([json.dumps([]), json.dumps([])])
    Verifier(client=verifier_client).verify({"answer": "I don't know.", "claims": []}, [])
    assert [call["timeout"] for call in verifier_client.completions.calls] == [120, 120]


def test_backoff_has_five_retry_cap_and_clear_error():
    calls = []

    def always_rate_limited():
        calls.append(1)
        raise RuntimeError("HTTP 429 rate limit")

    with pytest.raises(RuntimeError, match="5 retries"):
        _with_backoff(always_rate_limited, retries=100, base_seconds=0)
    assert len(calls) == 6
