# -*- coding: utf-8 -*-
"""BkAgentApi 客户端单元测试。"""

from unittest.mock import MagicMock, patch

from aidev_agent.api.bk_agent import BkAgentApi, Client


class TestClientOperations:
    """Client 类的 bind_property 操作测试。"""

    def test_client_has_ping(self) -> None:
        """Client 类有 ping 属性。"""
        assert hasattr(Client, "ping")

    def test_client_has_private_chat_completion(self) -> None:
        """Client 类有 private_chat_completion 属性。"""
        assert hasattr(Client, "private_chat_completion")

    def test_client_has_openapi_chat_completion(self) -> None:
        """Client 类有 openapi_chat_completion 属性。"""
        assert hasattr(Client, "openapi_chat_completion")

    def test_client_has_create_session(self) -> None:
        """Client 类有 create_session 属性。"""
        assert hasattr(Client, "create_session")

    def test_client_has_save_session_content(self) -> None:
        """Client 类有 save_session_content 属性。"""
        assert hasattr(Client, "save_session_content")


class TestBkAgentApiGetClient:
    """BkAgentApi.get_client() 方法测试。"""

    @patch("aidev_agent.api.bk_agent._get_client_by_settings")
    def test_get_client_returns_client_instance(self, mock_get_client: MagicMock) -> None:
        """get_client() 返回 Client 实例。"""
        mock_client = MagicMock(spec=Client)
        mock_get_client.return_value = mock_client

        result = BkAgentApi.get_client(agent_code="test_agent")

        assert result is mock_client
        mock_get_client.assert_called_once()

    @patch("aidev_agent.api.bk_agent._get_client_by_settings")
    def test_get_client_endpoint_contains_bp_prefix(self, mock_get_client: MagicMock) -> None:
        """get_client(agent_code='my_agent') 的 endpoint 包含 bp-my_agent。"""
        mock_client = MagicMock(spec=Client)
        mock_get_client.return_value = mock_client

        BkAgentApi.get_client(agent_code="my_agent")

        call_kwargs = mock_get_client.call_args
        assert "bp-my_agent" in call_kwargs.kwargs["endpoint"]

    @patch("aidev_agent.api.bk_agent._get_client_by_settings")
    def test_get_client_with_access_token_sets_bkapi_authorization_header(self, mock_get_client: MagicMock) -> None:
        """get_client(access_token='token123') 调用 update_bkapi_authorization。"""
        mock_client = MagicMock()
        mock_client.session.headers = {}
        mock_get_client.return_value = mock_client

        BkAgentApi.get_client(agent_code="test", access_token="token123")

        mock_client.update_bkapi_authorization.assert_called_once_with(
            access_token="token123",
        )

    @patch("aidev_agent.api.bk_agent._get_client_by_settings")
    def test_get_client_no_access_token_skips_bkapi_authorization_header(self, mock_get_client: MagicMock) -> None:
        """get_client(access_token='') 不设置 X-Bkapi-Authorization header。"""
        mock_client = MagicMock()
        mock_client.session.headers = {}
        mock_get_client.return_value = mock_client

        BkAgentApi.get_client(agent_code="test", access_token="")

        assert "X-Bkapi-Authorization" not in mock_client.session.headers

    @patch("aidev_agent.api.bk_agent._get_client_by_settings")
    def test_get_client_default_credentials(self, mock_get_client: MagicMock) -> None:
        """get_client() 默认使用 settings.APP_CODE 和 settings.SECRET_KEY。"""
        mock_client = MagicMock(spec=Client)
        mock_get_client.return_value = mock_client

        BkAgentApi.get_client(agent_code="test")

        call_kwargs = mock_get_client.call_args
        assert "bk_app_code" in call_kwargs.kwargs
        assert "bk_app_secret" in call_kwargs.kwargs

    @patch("aidev_agent.api.bk_agent._get_client_by_settings")
    def test_get_client_custom_credentials(self, mock_get_client: MagicMock) -> None:
        """get_client(app_code=..., app_secret=...) 使用自定义凭据。"""
        mock_client = MagicMock(spec=Client)
        mock_get_client.return_value = mock_client

        BkAgentApi.get_client(agent_code="test", app_code="custom_code", app_secret="custom_secret")

        call_kwargs = mock_get_client.call_args
        assert call_kwargs.kwargs["bk_app_code"] == "custom_code"
        assert call_kwargs.kwargs["bk_app_secret"] == "custom_secret"


class TestBkAgentApiValidateEndpoint:
    """BkAgentApi.get_client(validate_endpoint=True) 端点校验测试。"""

    @patch("aidev_agent.api.bk_agent._get_client_by_settings")
    def test_validate_endpoint_clears_bare_url(self, mock_get_client: MagicMock) -> None:
        """validate_endpoint=True 时，不含环境的默认平台 URL 被清空并自动构建。"""
        mock_client = MagicMock(spec=Client)
        mock_get_client.return_value = mock_client

        # 模拟一个不含 stage 的平台默认 URL（与 bare_suffix 匹配）
        with (
            patch("aidev_agent.api.bk_agent.get_endpoint", return_value="http://auto/bp-test/prod"),
            patch("aidev_agent.api.bk_agent.APIGW_URL_FORMAT", "http://host/{api_name}/{stage}"),
        ):
            BkAgentApi.get_client(
                agent_code="test",
                endpoint="http://host/bp-test",  # 不含 stage，与 bare_suffix 一致
                validate_endpoint=True,
            )

        # endpoint 应被清空并由 get_endpoint 自动构建
        call_kwargs = mock_get_client.call_args
        assert call_kwargs.kwargs["endpoint"] == "http://auto/bp-test/prod"

    @patch("aidev_agent.api.bk_agent._get_client_by_settings")
    def test_validate_endpoint_keeps_valid_url(self, mock_get_client: MagicMock) -> None:
        """validate_endpoint=True 时，含环境的有效 URL 被保留。"""
        mock_client = MagicMock(spec=Client)
        mock_get_client.return_value = mock_client

        with patch("aidev_agent.api.bk_agent.APIGW_URL_FORMAT", "http://host/{api_name}/{stage}"):
            BkAgentApi.get_client(
                agent_code="test",
                endpoint="http://host/bp-test/stag",
                validate_endpoint=True,
            )

        call_kwargs = mock_get_client.call_args
        assert call_kwargs.kwargs["endpoint"] == "http://host/bp-test/stag"

    @patch("aidev_agent.api.bk_agent._get_client_by_settings")
    def test_validate_endpoint_false_keeps_any_url(self, mock_get_client: MagicMock) -> None:
        """validate_endpoint=False（默认）时，任何 endpoint 都原样传递。"""
        mock_client = MagicMock(spec=Client)
        mock_get_client.return_value = mock_client

        BkAgentApi.get_client(
            agent_code="test",
            endpoint="http://host/bp-test/",
            validate_endpoint=False,
        )

        call_kwargs = mock_get_client.call_args
        assert call_kwargs.kwargs["endpoint"] == "http://host/bp-test/"

    @patch("aidev_agent.api.bk_agent._get_client_by_settings")
    def test_validate_endpoint_with_empty_endpoint_noop(self, mock_get_client: MagicMock) -> None:
        """validate_endpoint=True 但 endpoint 为空时，不触发校验，正常自动构建。"""
        mock_client = MagicMock(spec=Client)
        mock_get_client.return_value = mock_client

        BkAgentApi.get_client(
            agent_code="test",
            endpoint="",
            validate_endpoint=True,
        )

        # get_endpoint 应被调用以自动构建 endpoint
        call_kwargs = mock_get_client.call_args
        assert "bp-test" in call_kwargs.kwargs["endpoint"]
