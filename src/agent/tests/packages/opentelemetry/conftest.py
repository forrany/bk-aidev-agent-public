# -*- coding: utf-8 -*-
"""OpenTelemetry 测试目录的 conftest。

OTel 是 ``aidev-agent`` 的可选 extras（``pip install aidev-agent[opentelemetry]``），
未安装或版本组合不一致时整体跳过本目录。

注意：不能在 conftest.py 里使用 allow_module_level=True 的 skip/importorskip，
      因为 conftest 中的模块级 skip 会向父目录传播，导致整个 tests/ session 被跳过。
      改为在父级 conftest 中用 pytest_ignore_collect 忽略本目录。
"""
