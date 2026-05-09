"""Test: verify trace context propagation to LLM gateway via LangGraph execution.

Constructs a minimal LangGraph graph with a ChatModel node, instruments httpx + threading,
and verifies that the LLM HTTP request carries traceparent header with correct trace_id.

Intercepts at httpcore.ConnectionPool.handle_request level — this is AFTER OTel's
httpx instrumentation injects traceparent into the request, but BEFORE actual network I/O.
"""

import threading
from typing import Annotated
from unittest.mock import patch

import httpcore
import httpx
import pytest
from aidev_agent.packages.langchain_core.models.llm_gateway import ChatModel
from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from opentelemetry import trace
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.threading import ThreadingInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from typing_extensions import TypedDict


class GraphState(TypedDict):
    messages: Annotated[list, add_messages]


FAKE_COMPLETION = (
    b'{"id":"x","object":"chat.completion","choices":'
    b'[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],'
    b'"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}'
)


@pytest.fixture()
def otel_env():
    """Set up OTel TracerProvider and instrument httpx + threading."""
    provider = TracerProvider()
    trace.set_tracer_provider(provider)
    HTTPXClientInstrumentor().instrument()
    ThreadingInstrumentor().instrument()

    yield provider

    HTTPXClientInstrumentor().uninstrument()
    ThreadingInstrumentor().uninstrument()
    provider.shutdown()


def _make_httpcore_interceptor(captured: dict):
    """Create an httpcore-level interceptor that captures headers."""

    def intercept(self, request):
        captured.update({k.decode(): v.decode() for k, v in request.headers})
        return httpcore.Response(status=200, content=FAKE_COMPLETION, headers=[(b"content-type", b"application/json")])

    return intercept


def test_same_thread_traceparent(otel_env):
    """Baseline: httpx in same thread as span correctly injects traceparent."""
    captured = {}
    tracer = trace.get_tracer("test")

    with tracer.start_as_current_span("root") as span:
        trace_id = format(span.get_span_context().trace_id, "032x")
        with (
            patch.object(httpcore.ConnectionPool, "handle_request", _make_httpcore_interceptor(captured)),
            httpx.Client(base_url="http://fake") as c,
        ):
            c.post("/v1/completions", json={})

    assert "traceparent" in captured
    assert captured["traceparent"].split("-")[1] == trace_id


def test_child_thread_traceparent(otel_env):
    """ThreadingInstrumentor propagates context to child threads."""
    captured = {}
    tracer = trace.get_tracer("test")

    with tracer.start_as_current_span("root") as span:
        trace_id = format(span.get_span_context().trace_id, "032x")

        def work():
            with (
                patch.object(httpcore.ConnectionPool, "handle_request", _make_httpcore_interceptor(captured)),
                httpx.Client(base_url="http://fake") as c,
            ):
                c.post("/v1/completions", json={})

        t = threading.Thread(target=work)
        t.start()
        t.join()

    assert "traceparent" in captured
    assert captured["traceparent"].split("-")[1] == trace_id


def test_langgraph_chatmodel_traceparent(otel_env):
    """LangGraph graph calling ChatModel propagates traceparent to LLM gateway."""
    captured = {}
    model = ChatModel.get_setup_instance(model="gpt-4", base_url="http://fake-gateway/v1")

    def model_node(state: GraphState):
        return {"messages": [model.invoke(state["messages"])]}

    graph_builder = StateGraph(GraphState)
    graph_builder.add_node("model", model_node)
    graph_builder.add_edge(START, "model")
    graph_builder.add_edge("model", END)
    graph = graph_builder.compile()

    tracer = trace.get_tracer("agent")
    with tracer.start_as_current_span("agent-execute") as span:
        trace_id = format(span.get_span_context().trace_id, "032x")
        with patch.object(httpcore.ConnectionPool, "handle_request", _make_httpcore_interceptor(captured)):
            result = graph.invoke({"messages": [HumanMessage(content="hello")]})

    assert result["messages"][-1].content == "hi"
    assert "traceparent" in captured, f"No traceparent. Headers: {list(captured.keys())}"
    parts = captured["traceparent"].split("-")
    assert parts[1] == trace_id, f"trace_id mismatch: got {parts[1]}, want {trace_id}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
