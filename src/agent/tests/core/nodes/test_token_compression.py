# -*- coding: utf-8 -*-
"""测试 Token 压缩中间件。"""

from unittest.mock import MagicMock, patch

from aidev_agent.core.nodes.model.pydantic_models import ProcessorContext
from aidev_agent.core.nodes.model.token_compression import (
    BaseCompressionMiddleware,
    ChatHistoryCompressionMiddleware,
    CompressionState,
    KnowledgeCompressionMiddleware,
    TokenCompressionMiddleware,
    ToolOutputCompressionMiddleware,
    _ensure_compression_state,
)
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


# ============================================================================
# CompressionState 测试
# ============================================================================
class TestCompressionState:
    def test_default_values(self):
        state = CompressionState()
        assert state.knowledge_hash is None
        assert state.knowledge_cache is None
        assert state.knowledge_compressed is False
        assert state.tool_output_compressed is False
        assert state.tool_output_compressed_ids == set()
        assert state.chat_history_removed == 0
        assert state.chat_history_baseline == 0

    def test_from_legacy_with_old_keys(self):
        raw = {
            "context_compressed": True,
            "chat_history_compression_count": 5,
            "knowledge_hash": "abc123",
        }
        state = CompressionState.from_legacy(raw)
        assert state.knowledge_compressed is True
        assert state.chat_history_removed == 5
        assert state.knowledge_hash == "abc123"

    def test_from_legacy_with_new_keys(self):
        raw = {
            "knowledge_compressed": True,
            "chat_history_removed": 3,
            "tool_output_compressed": True,
            "tool_output_compressed_ids": ["id1", "id2"],
        }
        state = CompressionState.from_legacy(raw)
        assert state.knowledge_compressed is True
        assert state.chat_history_removed == 3
        assert state.tool_output_compressed is True
        assert state.tool_output_compressed_ids == {"id1", "id2"}


class TestEnsureCompressionState:
    def test_creates_new_state_if_missing(self):
        metadata = {}
        state = _ensure_compression_state(metadata)
        assert isinstance(state, CompressionState)
        assert metadata["_compression_state"] is state

    def test_returns_existing_state(self):
        existing = CompressionState(knowledge_compressed=True)
        metadata = {"_compression_state": existing}
        state = _ensure_compression_state(metadata)
        assert state is existing

    def test_migrates_legacy_dict(self):
        metadata = {"_compression_state": {"context_compressed": True}}
        state = _ensure_compression_state(metadata)
        assert isinstance(state, CompressionState)
        assert state.knowledge_compressed is True


# ============================================================================
# BaseCompressionMiddleware 测试
# ============================================================================
class TestBaseCompressionMiddleware:
    def test_is_overflow_returns_false_when_no_limit(self):
        ctx = ProcessorContext(state={}, config={}, token_limit=None)
        assert BaseCompressionMiddleware._is_overflow(ctx) is False

    def test_is_overflow_returns_false_when_under_limit(self):
        mock_llm = MagicMock()
        mock_llm.get_num_tokens_from_messages.return_value = 500

        mock_template = MagicMock()
        mock_template._format_prompt_with_error_handling.return_value = MagicMock(messages=[])

        ctx = ProcessorContext(
            state={},
            config={},
            llm=mock_llm,
            chat_prompt_template=mock_template,
            token_limit=1000,
            token_margin=100,
            variables={},
        )
        assert BaseCompressionMiddleware._is_overflow(ctx) is False

    def test_is_overflow_returns_true_when_over_limit(self):
        mock_llm = MagicMock()
        mock_llm.get_num_tokens_from_messages.return_value = 950

        mock_template = MagicMock()
        mock_template._format_prompt_with_error_handling.return_value = MagicMock(messages=[])

        ctx = ProcessorContext(
            state={},
            config={},
            llm=mock_llm,
            chat_prompt_template=mock_template,
            token_limit=1000,
            token_margin=100,
            variables={},
        )
        # 950 > 1000 - 100 = 900
        assert BaseCompressionMiddleware._is_overflow(ctx) is True


# ============================================================================
# KnowledgeCompressionMiddleware 测试
# ============================================================================
class TestKnowledgeCompressionMiddleware:
    def test_skips_when_no_context(self):
        middleware = KnowledgeCompressionMiddleware()
        ctx = ProcessorContext(
            state={},
            config={},
            llm=MagicMock(),
            chat_prompt_template=MagicMock(),
            token_limit=1000,
            variables={},  # no context
        )

        next_called = []
        middleware(ctx, lambda: next_called.append(True))

        assert len(next_called) == 1

    def test_skips_when_not_overflow(self):
        mock_llm = MagicMock()
        mock_llm.get_num_tokens_from_messages.return_value = 500

        mock_template = MagicMock()
        mock_template._format_prompt_with_error_handling.return_value = MagicMock(messages=[])

        middleware = KnowledgeCompressionMiddleware()
        ctx = ProcessorContext(
            state={},
            config={},
            llm=mock_llm,
            chat_prompt_template=mock_template,
            token_limit=1000,
            token_margin=100,
            variables={"context": "some context"},
        )

        next_called = []
        middleware(ctx, lambda: next_called.append(True))

        assert len(next_called) == 1
        assert ctx.variables["context"] == "some context"

    @patch("aidev_agent.core.nodes.model.token_compression.conditional_dispatch_custom_event")
    def test_compresses_when_overflow_and_func_provided(self, mock_dispatch):
        mock_llm = MagicMock()
        mock_llm.get_num_tokens_from_messages.return_value = 950

        mock_template = MagicMock()
        mock_template._format_prompt_with_error_handling.return_value = MagicMock(messages=[])

        compressor_func = MagicMock(return_value="compressed context")

        middleware = KnowledgeCompressionMiddleware()
        ctx = ProcessorContext(
            state={},
            config={},
            llm=mock_llm,
            chat_prompt_template=mock_template,
            token_limit=1000,
            token_margin=100,
            variables={"context": "original context", "query": "test query"},
            metadata={
                "knowledge_compressor_func": compressor_func,
                "provided_chat_history": [],
            },
        )

        next_called = []
        middleware(ctx, lambda: next_called.append(True))

        assert len(next_called) == 1
        assert ctx.variables["context"] == "compressed context"
        compressor_func.assert_called_once()

    def test_uses_cache_on_repeated_call(self):
        mock_llm = MagicMock()
        mock_llm.get_num_tokens_from_messages.return_value = 500  # not overflow

        mock_template = MagicMock()
        mock_template._format_prompt_with_error_handling.return_value = MagicMock(messages=[])

        middleware = KnowledgeCompressionMiddleware()

        # Pre-populate compression state with cache
        state = CompressionState(
            knowledge_hash="abc123",
            knowledge_cache="cached context",
            knowledge_compressed=True,
        )

        ctx = ProcessorContext(
            state={},
            config={},
            llm=mock_llm,
            chat_prompt_template=mock_template,
            token_limit=1000,
            variables={"context": "original context"},
            metadata={"_compression_state": state},
        )

        # Mock the hash to match
        with patch.object(KnowledgeCompressionMiddleware, "_compute_hash", return_value="abc123"):
            next_called = []
            middleware(ctx, lambda: next_called.append(True))

        assert ctx.variables["context"] == "cached context"


# ============================================================================
# ChatHistoryCompressionMiddleware 测试
# ============================================================================
class TestChatHistoryCompressionMiddleware:
    def test_skips_when_no_chat_history(self):
        middleware = ChatHistoryCompressionMiddleware()
        ctx = ProcessorContext(
            state={},
            config={},
            llm=MagicMock(),
            chat_prompt_template=MagicMock(),
            token_limit=1000,
            variables={},
        )

        next_called = []
        middleware(ctx, lambda: next_called.append(True))

        assert len(next_called) == 1

    @patch("aidev_agent.core.nodes.model.token_compression.conditional_dispatch_custom_event")
    def test_removes_messages_when_overflow(self, mock_dispatch):
        # First call returns overflow, subsequent calls return under limit
        call_count = [0]

        def mock_get_tokens(*args):
            call_count[0] += 1
            if call_count[0] <= 2:
                return 950  # overflow
            return 500  # under limit

        mock_llm = MagicMock()
        mock_llm.get_num_tokens_from_messages.side_effect = mock_get_tokens

        mock_template = MagicMock()
        mock_template._format_prompt_with_error_handling.return_value = MagicMock(messages=[])

        middleware = ChatHistoryCompressionMiddleware()
        ctx = ProcessorContext(
            state={"messages": []},
            config={},
            llm=mock_llm,
            chat_prompt_template=mock_template,
            token_limit=1000,
            token_margin=100,
            variables={
                "chat_history": [
                    HumanMessage(content="msg1"),
                    AIMessage(content="msg2"),
                    HumanMessage(content="msg3"),
                ]
            },
        )

        next_called = []
        middleware(ctx, lambda: next_called.append(True))

        assert len(next_called) == 1
        # Should have removed some messages
        state = _ensure_compression_state(ctx.metadata)
        assert state.chat_history_removed > 0

    def test_applies_previous_removal_count(self):
        mock_llm = MagicMock()
        mock_llm.get_num_tokens_from_messages.return_value = 500

        mock_template = MagicMock()
        mock_template._format_prompt_with_error_handling.return_value = MagicMock(messages=[])

        state = CompressionState(chat_history_removed=1)

        middleware = ChatHistoryCompressionMiddleware()
        ctx = ProcessorContext(
            state={"messages": []},
            config={},
            llm=mock_llm,
            chat_prompt_template=mock_template,
            token_limit=1000,
            variables={
                "chat_history": [
                    HumanMessage(content="msg1"),
                    AIMessage(content="msg2"),
                    HumanMessage(content="msg3"),
                ]
            },
            metadata={"_compression_state": state},
        )

        next_called = []
        middleware(ctx, lambda: next_called.append(True))

        # Should skip first message due to previous removal
        assert len(ctx.variables["chat_history"]) == 2


# ============================================================================
# ToolOutputCompressionMiddleware 测试
# ============================================================================
class TestToolOutputCompressionMiddleware:
    def test_tool_output_len_calculation(self):
        messages = [
            HumanMessage(content="hello"),
            ToolMessage(content="result1", tool_call_id="1"),
            ToolMessage(content="result2", tool_call_id="2"),
            AIMessage(content="response"),
        ]
        length = ToolOutputCompressionMiddleware._tool_output_len(messages)
        assert length == len("result1") + len("result2")

    def test_get_query_from_variables(self):
        ctx = ProcessorContext(
            state={},
            config={},
            variables={"query": "test query"},
        )
        query = ToolOutputCompressionMiddleware._get_query(ctx)
        assert query == "test query"

    def test_get_query_from_state_input(self):
        ctx = ProcessorContext(
            state={"input": "state input"},
            config={},
            variables={},
        )
        query = ToolOutputCompressionMiddleware._get_query(ctx)
        assert query == "state input"

    def test_skips_when_no_llm(self):
        middleware = ToolOutputCompressionMiddleware()
        ctx = ProcessorContext(
            state={},
            config={},
            llm=None,
            metadata={"tool_messages": [ToolMessage(content="x" * 10000, tool_call_id="1")]},
        )

        next_called = []
        middleware(ctx, lambda: next_called.append(True))

        assert len(next_called) == 1

    def test_skips_when_under_threshold(self):
        middleware = ToolOutputCompressionMiddleware(tool_output_compress_thrd=5000)
        ctx = ProcessorContext(
            state={},
            config={},
            llm=MagicMock(),
            metadata={"tool_messages": [ToolMessage(content="short", tool_call_id="1")]},
        )

        next_called = []
        middleware(ctx, lambda: next_called.append(True))

        assert len(next_called) == 1

    @patch("aidev_agent.core.nodes.model.token_compression.conditional_dispatch_custom_event")
    @patch("aidev_agent.core.nodes.model.token_compression._compression_executor")
    def test_compress_if_too_long(self, mock_executor, mock_dispatch):
        # Mock the executor to return compressed content
        mock_future = MagicMock()
        mock_future.result.return_value = "compressed"
        mock_executor.submit.return_value = mock_future

        # Make as_completed return the future immediately
        with patch("concurrent.futures.as_completed", return_value=[mock_future]):
            middleware = ToolOutputCompressionMiddleware(tool_output_compress_thrd=100)
            ctx = ProcessorContext(
                state={},
                config={},
                llm=MagicMock(),
                variables={"query": "test"},
                metadata={
                    "tool_messages": [ToolMessage(content="x" * 200, tool_call_id="1", name="test_tool")],
                    "provided_chat_history": [],
                },
            )

            result = middleware.compress_if_too_long(ctx)

        assert result is True
        state = _ensure_compression_state(ctx.metadata)
        assert state.tool_output_compressed is True
        assert "1" in state.tool_output_compressed_ids

    def test_skips_already_compressed(self):
        middleware = ToolOutputCompressionMiddleware(tool_output_compress_thrd=100)

        state = CompressionState(
            tool_output_compressed=True,
            tool_output_compressed_ids={"1"},
        )

        ctx = ProcessorContext(
            state={},
            config={},
            llm=MagicMock(),
            metadata={
                "tool_messages": [ToolMessage(content="x" * 200, tool_call_id="1", name="test_tool")],
                "_compression_state": state,
            },
        )

        result = middleware.compress_if_too_long(ctx)
        # Should not compress because tool_call_id "1" is already in compressed_ids
        assert result is False

    def test_compress_for_token_overflow_disabled(self):
        middleware = ToolOutputCompressionMiddleware(enable_token_overflow_compression=False)
        ctx = ProcessorContext(
            state={},
            config={},
            llm=MagicMock(),
            metadata={"tool_messages": [ToolMessage(content="x" * 200, tool_call_id="1")]},
        )

        result = middleware.compress_for_token_overflow(ctx)
        assert result is False


# ============================================================================
# TokenCompressionMiddleware 集成测试
# ============================================================================
class TestTokenCompressionMiddleware:
    def test_calls_next(self):
        middleware = TokenCompressionMiddleware()
        ctx = ProcessorContext(
            state={},
            config={},
            llm=None,
            variables={},
        )

        next_called = []
        middleware(ctx, lambda: next_called.append(True))

        assert len(next_called) == 1

    @patch("aidev_agent.core.nodes.model.token_compression.conditional_dispatch_custom_event")
    def test_compression_priority_order(self, mock_dispatch):
        """测试压缩优先级顺序：长度触发的工具压缩 -> 知识库 -> token触发的工具压缩 -> 聊天历史"""
        # This is a simplified integration test
        call_order = []

        def mock_compress_if_too_long(self, ctx):
            call_order.append("tool_length")
            return False

        def mock_compress_for_token(self, ctx):
            call_order.append("tool_token")
            return False

        with patch.object(ToolOutputCompressionMiddleware, "compress_if_too_long", mock_compress_if_too_long):
            with patch.object(ToolOutputCompressionMiddleware, "compress_for_token_overflow", mock_compress_for_token):
                middleware = TokenCompressionMiddleware()

                mock_llm = MagicMock()
                mock_llm.get_num_tokens_from_messages.return_value = 500  # not overflow

                mock_template = MagicMock()
                mock_template._format_prompt_with_error_handling.return_value = MagicMock(messages=[])

                ctx = ProcessorContext(
                    state={},
                    config={},
                    llm=mock_llm,
                    chat_prompt_template=mock_template,
                    token_limit=1000,
                    variables={},
                )

                next_called = []
                middleware(ctx, lambda: next_called.append(True))

        assert "tool_length" in call_order
        assert len(next_called) == 1


# ============================================================================
# Prompt 模板测试
# ============================================================================
class TestPromptTemplates:
    def test_common_compressor_prompt_renders(self):
        from aidev_agent.core.nodes.model.token_compression import _COMMON_COMPRESSOR_USR_PROMPT

        result = _COMMON_COMPRESSOR_USR_PROMPT.render(content="test content")
        assert "test content" in result

    def test_specific_compressor_sys_prompt_renders(self):
        from aidev_agent.core.nodes.model.token_compression import _SPECIFIC_COMPRESSOR_SYS_PROMPT

        result = _SPECIFIC_COMPRESSOR_SYS_PROMPT.render(candidate_tool_name="my_tool")
        assert "my_tool" in result

    def test_specific_compressor_usr_prompt_renders(self):
        from aidev_agent.core.nodes.model.token_compression import _SPECIFIC_COMPRESSOR_USR_PROMPT

        result = _SPECIFIC_COMPRESSOR_USR_PROMPT.render(
            provided_chat_history="[history]",
            candidate_tool_name="my_tool",
            candidate_tool_result="result data",
            query="user question",
        )
        assert "[history]" in result
        assert "my_tool" in result
        assert "result data" in result
        assert "user question" in result


# ============================================================================
# 向后兼容性测试
# ============================================================================
class TestBackwardCompatibility:
    def test_token_overflow_middleware_alias(self):
        from aidev_agent.core.nodes.model.token_compression import TokenOverflowMiddleware

        assert TokenOverflowMiddleware is TokenCompressionMiddleware
