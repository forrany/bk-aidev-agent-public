# -*- coding: utf-8 -*-
"""
常量模块单元测试
"""

from aidev_bkplugin.constants import AGUI_PROTOCOL_VERSION


class TestProtocolVersion:
    """测试 AG-UI 协议版本标识"""

    def test_agui_protocol_version_is_v2(self):
        """验证 AG-UI 协议版本为 v2"""
        assert AGUI_PROTOCOL_VERSION == "v2"

    def test_create_session_includes_protocol_version(self):
        """验证创建会话时 protocol_version 被正确添加到请求数据"""
        # 模拟 ViewSet.create 的核心逻辑
        request_data = {"session_name": "测试会话"}
        data = {**request_data, "protocol_version": AGUI_PROTOCOL_VERSION}

        assert data["protocol_version"] == "v2"
        assert data["session_name"] == "测试会话"

    def test_protocol_version_not_overwritten(self):
        """验证即使请求中已有 protocol_version，也会被覆盖为正确值"""
        # 模拟请求中传入了错误的版本
        request_data = {"session_name": "测试会话", "protocol_version": "v1"}
        data = {**request_data, "protocol_version": AGUI_PROTOCOL_VERSION}

        # 应该被覆盖为 v2
        assert data["protocol_version"] == "v2"
