# -*- coding: utf-8 -*-
"""compress_log CustomEvent 路径验证测试

验证 compress_log 事件走 CustomEvent 路径而非 TextMessage 三元组：
1. COMPRESS_LOG 枚举值正确
2. AidevAGUIAgent.run() 中拦截并展开为 TextMessage 三元组输出到 SSE
3. BaseSessionWriter 不处理 COMPRESS_LOG 事件（不触发 DB 写入）
4. token_compression._dispatch_log 使用 COMPRESS_LOG.value 作为事件名
"""

import inspect

from ag_ui.core import CustomEvent, EventType
from aidev_agent.core.ag_ui.aidev_agent import AidevAGUIAgent
from aidev_agent.core.ag_ui.types import CustomMessageType
from aidev_agent.core.nodes.model.token_compression import BaseCompressionMiddleware
from aidev_agent.services.event_handlers.base import BaseSessionWriter


class TestCompressLogEnumValue:
    """Test 1: CustomMessageType.COMPRESS_LOG 枚举值"""

    def test_compress_log_enum_value(self):
        """COMPRESS_LOG 枚举值应为 'compress_log'"""
        assert CustomMessageType.COMPRESS_LOG.value == "compress_log"

    def test_compress_log_is_member_of_custom_message_type(self):
        """COMPRESS_LOG 应该是 CustomMessageType 的成员"""
        assert "COMPRESS_LOG" in [e.name for e in CustomMessageType]


class TestBaseSessionWriterIgnoresCompressLog:
    """Test 3: BaseSessionWriter 不处理 COMPRESS_LOG 事件"""

    def test_base_session_writer_ignores_compress_log(self):
        """BaseSessionWriter 不应处理 COMPRESS_LOG 事件，不触发 DB 写入"""
        event = CustomEvent(
            type=EventType.CUSTOM,
            name=CustomMessageType.COMPRESS_LOG.value,
            value={"compress_log": "test compression log content"},
        )

        # BaseSessionWriter 是抽象类，使用其文档中推荐的 MyWriter 子类方式
        from aidev_agent.services.event_handlers.base import BaseSessionWriter

        class MyWriter(BaseSessionWriter):
            def _do_create_content(self, messages, content):  # noqa: PLR6301
                return ""

        writer = MyWriter.__new__(MyWriter)
        writer._written_message_ids = set()
        writer._streaming_messages = {}

        # 调用 _dispatch_custom_event_direct — 不应有任何副作用
        writer._dispatch_custom_event_direct(event)

        # 验证 _streaming_messages 未变
        assert len(writer._streaming_messages) == 0

    def test_dispatch_custom_event_direct_has_no_compress_log_handler(self):
        """_dispatch_custom_event_direct 方法中不应包含 COMPRESS_LOG 分支"""
        source = inspect.getsource(BaseSessionWriter._dispatch_custom_event_direct)
        assert "compress_log" not in source
        assert "COMPRESS_LOG" not in source


class TestDispatchLogUsesCompressLogEventName:
    """Test 4: token_compression._dispatch_log 使用 COMPRESS_LOG 事件名"""

    def test_dispatch_log_uses_compress_log_event_name(self):
        """_dispatch_log 应使用 CustomMessageType.COMPRESS_LOG.value 作为事件名"""
        source = inspect.getsource(BaseCompressionMiddleware._dispatch_log)
        assert "COMPRESS_LOG" in source
        assert '"custom_event"' not in source

    def test_dispatch_log_imports_custom_message_type(self):
        """token_compression 模块应导入 CustomMessageType"""
        import aidev_agent.core.nodes.model.token_compression as mod

        source = inspect.getsource(mod)
        assert "CustomMessageType" in source


class TestAidevAgentRunInterceptsCompressLog:
    """Test 2: AidevAGUIAgent.run() 中 compress_log CustomEvent 被拦截并展开为 TextMessage 三元组"""

    def test_run_method_has_compress_log_interception(self):
        """AidevAGUIAgent.run() 方法应包含 COMPRESS_LOG 拦截逻辑"""
        source = inspect.getsource(AidevAGUIAgent.run)
        assert "COMPRESS_LOG" in source

    def test_run_method_generates_text_message_triplet(self):
        """AidevAGUIAgent.run() 中 compress_log 拦截应生成 TextMessage 三元组"""
        source = inspect.getsource(AidevAGUIAgent.run)
        assert "TextMessageStartEvent" in source
        assert "TextMessageContentEvent" in source
        assert "TextMessageEndEvent" in source

    def test_run_method_uses_event_encoder_encode(self):
        """AidevAGUIAgent.run() 中 compress_log 三元组应使用 event_encoder.encode 直接输出"""
        source = inspect.getsource(AidevAGUIAgent.run)
        # 在 compress_log 分支中应使用 event_encoder.encode
        assert "event_encoder.encode" in source
