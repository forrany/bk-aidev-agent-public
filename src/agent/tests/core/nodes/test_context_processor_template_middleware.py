# -*- coding: utf-8 -*-
"""测试 ContextProcessor 模板选择中间件。"""

from aidev_agent.core.graphs.react.prompts import DEFAULT_QA_PROMPT_TEMPLATES, general_qa_prompt_structured_chat
from aidev_agent.core.nodes.model.basic_middleware import BaseTemplateMiddleware, DecisionBasedTemplateMiddleware
from aidev_agent.core.nodes.model.context_processor import MiddlewarePipeline
from aidev_agent.core.nodes.model.pydantic_models import ProcessorContext
from aidev_agent.enums import Decision


class TestTemplateMiddleware:
    def test_fallback_template_structured_is_deepcopy(self):
        pipeline = MiddlewarePipeline()
        pipeline.use(BaseTemplateMiddleware(use_structured_response=True, prefix=None, role_prompt=""))
        pipeline.use(
            DecisionBasedTemplateMiddleware(
                chat_prompt_templates=DEFAULT_QA_PROMPT_TEMPLATES,
                use_structured_response=True,
                enable_query_clarification=True,
            )
        )

        ctx = ProcessorContext(state={"decision": "UNKNOWN"}, config={})
        pipeline.execute(ctx)

        assert ctx.prompt_template is not None
        # fallback 使用 deepcopy，避免运行时污染原模板
        assert ctx.prompt_template is not general_qa_prompt_structured_chat

    def test_decision_general_qa_structured_uses_default_template(self):
        pipeline = MiddlewarePipeline()
        pipeline.use(BaseTemplateMiddleware(use_structured_response=True, prefix=None, role_prompt=""))
        pipeline.use(
            DecisionBasedTemplateMiddleware(
                chat_prompt_templates=DEFAULT_QA_PROMPT_TEMPLATES,
                use_structured_response=True,
                enable_query_clarification=True,
            )
        )

        ctx = ProcessorContext(state={"decision": Decision.GENERAL_QA}, config={})
        pipeline.execute(ctx)

        assert ctx.prompt_template is DEFAULT_QA_PROMPT_TEMPLATES["general_qa_prompt_structured_chat"]

    def test_decision_query_clarification_can_fallback_to_private(self):
        pipeline = MiddlewarePipeline()
        pipeline.use(BaseTemplateMiddleware(use_structured_response=False, prefix="P", role_prompt="R"))
        pipeline.use(
            DecisionBasedTemplateMiddleware(
                chat_prompt_templates=DEFAULT_QA_PROMPT_TEMPLATES,
                use_structured_response=False,
                enable_query_clarification=False,
            )
        )

        ctx = ProcessorContext(state={"decision": Decision.QUERY_CLARIFICATION}, config={})
        pipeline.execute(ctx)

        assert ctx.prompt_template is DEFAULT_QA_PROMPT_TEMPLATES["private_qa_prompt_tool_calling"]
