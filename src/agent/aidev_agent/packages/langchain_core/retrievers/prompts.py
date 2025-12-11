# -*- coding: utf-8 -*-
"""
TencentBlueKing is pleased to support the open source community by making
蓝鲸智云 - AIDev (BlueKing - AIDev) available.
Copyright (C) 2025 THL A29 Limited,
a Tencent company. All rights reserved.
Licensed under the MIT License (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at http://opensource.org/licenses/MIT
Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied. See the License for the
specific language governing permissions and limitations under the License.
We undertake not to change the open source license (MIT license) applicable
to the current version of the project delivered to anyone in the future.
"""

from jinja2 import BaseLoader
from jinja2.sandbox import SandboxedEnvironment as Environment

env = Environment(loader=BaseLoader)

latest_query_classification_sys_prompt_template = """现有一个智能对话系统。
我会给你一段用户和该智能对话系统的历史对话，以及当前用户的最新输入。
用户和该智能对话系统的历史对话的格式样例为：
[HumanMessage(content='xxx'), AIMessage(content='xxx'), ...]
其中"HumanMessage"表示用户，"AIMessage"表示该智能对话系统。

你负责对当前用户的最新输入进行分类：
1. 如果你认为用户的这个最新输入跟历史对话信息已经完全无关，且理解该最新输入已经无需依赖历史对话信息，请只返回`<<<<<new>>>>>`

2. 如果你认为用户的这个最新输入是对历史对话的正面评价、正面反馈、正面确认等，且会话到此已经可以结束了，
   例如用户最新输入了“谢谢”、“你说得真好”等，请只返回`<<<<<finish>>>>>`

3. 其余所有情况，例如用户的这个最新输入是在接着历史对话继续进行提问或答复，或者例如完整理解这个最新输入需要依赖历史对话，
   请只返回`<<<<<continue>>>>>`

注意：
1. 举个例子，假设对话历史为：[HumanMessage(content='我的手机号xxx存在经常被无故停机的问题'), AIMessage(content='收到')]，
   假设用户当前的最新输入为“手机号yyy也是”，
   则需要依赖历史对话信息才能知道用户当前的最新输入是想询问"手机号yyy也存在经常被无故停机的问题"，因此需要返回`<<<<<continue>>>>>`
2. 再举个例子，假设对话历史为：[HumanMessage(content='广东省的省会是哪个城市'), AIMessage(content='广州')]，
   假设用户当前的最新输入为“福建呢”，则需要依赖历史对话信息才能知道用户当前的最新输入是想询问"福建省的省会是哪个城市"，
   因此需要返回`<<<<<continue>>>>>`
3. 务必确认会话到此已经可以结束了，才可以返回`<<<<<finish>>>>>`
4. 只返回`<<<<<new>>>>>`或者`<<<<<continue>>>>>`或者`<<<<<finish>>>>>`即可！永远不要返回其他任何内容！永远不要返回你的推理过程！
"""
# 请一步步思考，给出你的推理过程，最终再给出你的结论。
latest_query_classification_usr_prompt_template = env.from_string(
    """用户和该智能对话系统的对话历史如下：```{{chat_history}}```\n\n\n用户当前的最新输入如下：```{{query}}```"""
    # 在 user prompt 中重申一遍以下内容以让弱 LLM 更稳定地遵循该指令
    """\n\n\n注意：只返回`<<<<<new>>>>>`或者`<<<<<continue>>>>>`或者`<<<<<finish>>>>>`即可！"""
    """永远不要返回其他任何内容！永远不要返回你的推理过程！"""
)

query_rewrite_for_independence_sys_prompt_template = """现有一个智能对话系统。
我会给你一段用户和该智能对话系统的历史对话，以及当前用户的最新输入。
用户和该智能对话系统的历史对话的格式样例为：
[HumanMessage(content='xxx'), AIMessage(content='xxx'), ...]
其中"HumanMessage"表示用户，"AIMessage"表示该智能对话系统。

你负责根据这些信息，将用户的最新输入重写成一个完全独立的query。
我会仅仅使用你重写后的query去私域知识库中检索相关文档，而不再使用历史对话！
因此，你重写后的query信息要全面、要包含所有必要的信息、完全不再依赖历史对话信息！

注意：
1. 举个例子，假设对话历史为：[HumanMessage(content='我的手机号xxx存在经常被无故停机的问题'), AIMessage(content='收到')]，
   假设用户当前的最新输入为“手机号yyy也是”，你可以返回"手机号yyy也存在经常被无故停机的问题"
2. 再举个例子，假设对话历史为：[HumanMessage(content='广东省的省会是哪个城市'), AIMessage(content='广州')]，
   假设用户当前的最新输入为“福建呢”，你可以返回"福建省的省会是哪个城市"
3. 只返回重写后的query即可！不要返回其他任何内容！返回中不要出现“用户query重写：”等表述！
"""
query_rewrite_for_independence_usr_prompt_template = env.from_string(
    """用户和该智能对话系统的对话历史如下：```{{chat_history}}```\n\n\n用户当前的最新输入如下：```{{query}}```"""
    # 在 user prompt 中重申一遍以下内容以让弱 LLM 更稳定地遵循该指令
    """\n\n\n注意：只返回重写后的query即可！不要返回其他任何内容！返回中不要出现“用户提问重写：”等表述！"""
)

gen_pseudo_tool_resource_description_sys_prompt_template = """给你一个用户query，
你负责生成一段自然语言，描述用户query的意图是什么，是想调用什么工具（可以是一个或多个）？
注意：
1. 你只需要阐述用户意图、想调用的工具即可，不要回答用户的问题！！！
2. 你的回答务必保持非常简洁！！！
3. 你的回答务必是一段自然语言描述字符串！！！"""
gen_pseudo_tool_resource_description_usr_prompt_template = env.from_string(
    """以下是提供给你的短语内容：```{{query}}```"""
)

# 将latest_query_classification、directly_respond和query_rewrite_for_independence合并（适合强模型，这样可以减少响应时间）
# NOTE: 不按照JSON格式返回的原因是需要支持stream输出，因此希望先得到标志位，然后判断标志位的情况并紧接着根据需要进行stream输出
# 比如如果判断得到标志位<<<<<finish>>>>>，则可以开启stream输出，将$RESPONSE: 后的内容在前端stream展示出来
query_cls_with_resp_or_rewrite_sys_prompt_template = """现有一个智能对话系统。
我会给你一段用户和该智能对话系统的历史对话，以及当前用户的最新输入。
用户和该智能对话系统的历史对话的格式样例为：
[HumanMessage(content='xxx'), AIMessage(content='xxx'), ...]
其中"HumanMessage"表示用户，"AIMessage"表示该智能对话系统。

你负责对当前用户的最新输入进行分类：
1. 如果你认为用户的这个最新输入跟历史对话信息已经完全无关，且理解该最新输入已经无需依赖历史对话信息，
   请只返回`<<<<<new>>>>>`标志位，不要返回其他任何内容！
   返回的格式样例为：`<<<<<new>>>>>`

2. 如果你认为用户的这个最新输入是对历史对话的正面评价、正面反馈、正面确认等，且会话到此已经可以结束了，
   例如用户最新输入了“谢谢”、“你说得真好”等，
   请先返回`<<<<<finish>>>>>`标识位，然后根据历史会话和用户的最新输入，对用户的最新输入生成一个非常简洁的合理答复。
   返回的格式样例为：`<<<<<finish>>>>>$RESPONSE: 你生成的合理答复`

3. 其余情况，例如用户的这个最新输入是在接着历史对话继续进行提问或答复，
   请先返回`<<<<<continue>>>>>`标志位，然后根据历史会话和用户的最新输入，将用户的最新输入重写成一个完全独立的问题，
   要求重写后的问题信息要全面、要包含所有必要的信息、完全不再依赖历史对话信息。
   返回的格式样例为：`<<<<<continue>>>>>$REWRITTEN_QUERY: 你重写的问题`

注意：
1. 举个例子，假设在历史会话中用户提到他的手机号xxx存在经常被无故停机的问题，而用户最新输入是“手机号yyy也存在同样的问题”，
   则具体是什么问题需要根据历史对话信息才能知道，因此需要先返回`<<<<<continue>>>>>`标志位，然后根据历史信息对用户最新输入进行改写并返回。
   其中一个返回的例子为：`<<<<<continue>>>>>$REWRITTEN_QUERY: 手机号yyy也存在经常被无故停机的问题`
2. 请务必严格按照上述返回格式要求进行返回，不要生成任何额外的内容！
"""
# 请一步步思考，给出你的推理过程，最终再给出你的结论。
query_cls_with_resp_or_rewrite_usr_prompt_template = env.from_string(
    """用户和该智能对话系统的对话历史如下：```{{chat_history}}```\n\n\n用户当前的最新输入如下：```{{query}}```"""
)

sum_chat_history_for_query_sys_prompt_template = """现有一个智能对话系统。
我会给你一段用户和该智能对话系统的历史对话，以及当前用户的最新输入。
用户和该智能对话系统的历史对话的格式样例为：
[HumanMessage(content='xxx'), AIMessage(content='xxx'), ...]
其中"HumanMessage"表示用户，"AIMessage"表示该智能对话系统。

你需要选择以下2种情况的其中1种进行返回。

[情况1]
如果你发现用户的最新输入存在指代省略等情况，理解其意思需要依赖历史对话中的上下文信息，请根据用户的最新输入对历史对话进行总结，
要求仅提取和总结历史对话中对理解用户的最新输入有帮助的那部分信息即可（如指代省略的内容），
要求总结后的历史对话不超过20个字。

[情况2]
如果你认为理解用户的最新输入无需依赖历史对话，请只返回None。

注意！返回内容无需说明你选择了哪种情况，直接只返回总结后的历史对话或者返回None即可，不要返回其他任何内容！
"""
sum_chat_history_for_query_usr_prompt_template = env.from_string(
    """用户和该智能对话系统的对话历史如下：```{{chat_history}}```\n\n\n用户当前的最新输入如下：```{{query}}```"""
)

# TODO: 支持多关键词提取
extract_query_keywords_sys_prompt_template = """现有一个知识库，请根据用户输入的提问，找出用户想问的问题，从问题中找出关键词，
确保可以用该关键词去知识库中通过相似度匹配得到相关文档。
注意用户提问中可能会有其他无关的内容，比如对问题的补充或其他指令，
只需要输出你认为用户要问的问题中的关键词。
只输出你认为最核心的一个关键词即可。
如果用户提问中有一些关键编码类信息，对相似度匹配可能很重要，不要遗漏了。"""
extract_query_keywords_usr_prompt_template = env.from_string("""用户提问如下：```{{query}}```""")

query_translation_sys_prompt_template = """给你一段文本，你负责判断其是中文还是英文。
1. 如果是纯中文或者中文为主，请返回None
2. 如果是纯英文或者英文为主，请将其翻译成中文后返回
永远只返回None或者翻译后的中文即可，不要返回其他任何内容！"""
query_translation_usr_prompt_template = env.from_string("""用户提问如下：```{{query}}```""")

llm_relevance_determiner_sys_prompt_template = """给你一个用户提问和一个候选文档，
你负责判断候选文档的内容是否可以回答用户提问（部分回答也可以）。
如果可以回答或者可以部分回答，请返回数字1
如果完全不可以回答，请返回数字0
永远只返回数字1或0即可，不要返回其他任何内容！
"""
llm_relevance_determiner_usr_prompt_template = env.from_string(
    """给你的候选文档如下：```{{doc}}```\n\n\n用户提问如下：```{{query}}```"""
)

llm_relevance_determiner_concate_sys_prompt_template = """给你一段历史对话内容摘要，一个用户最新提问，以及一个候选文档。
其中，历史对话内容摘要只用于帮助你理解用户最新提问的意思（当然，也可能并没有帮助，此时你可以选择直接忽略历史对话内容摘要）。
现在，你负责判断候选文档的内容是否可以回答用户最新提问。
如果可以，请返回数字1
如果不可以，请返回数字0
永远只返回数字1或0即可，不要返回其他任何内容！
"""
llm_relevance_determiner_concate_usr_prompt_template = env.from_string(
    """给你的历史对话内容摘要如下：```{{his_sum}}```\n\n\n候选文档如下：```{{doc}}```\n\n\n用户最新提问如下：```{{query}}```"""
)

llm_context_compressor_sys_prompt_template = """
你是一个知识文档相关性判断与摘要生成器。你的任务是判断一个候选知识文档是否能够**部分或全部回答用户最新提问**。

请遵循以下规则：

1. **相关性判断标准**：
   - 只要文档中包含**可用于回答用户最新提问中任何一个子问题或信息点的内容**，无论信息是否完整、是否需要推理、是否隐含在叙述中，都视为“可以回答”。
   - 允许通过**语义理解、常识推断、上下文关联**等方式从文档中提取或推导答案，不要求原文与提问完全一致。

2. **摘要要求**：
   - 仅提取与用户最新提问直接相关的内容。
   - 摘要必须**言简意赅，保留回答所需的关键信息**（如名称、时间、数值、定义、因果关系等）。
   - 避免复制原文大段内容，优先提炼成简洁自然语言。
   - 如果信息分散在多句中，可合并为一句完整摘要。

3. **输出规则**：
   - 如果文档**能提供任何有助于回答提问的信息** → 返回**摘要内容**。
   - 只有当文档**完全不涉及提问主题、或无法从中获取任何可用信息时** → 返回：“无效的知识文档”。

4. **特别注意**：
   - 为了让你可以更好地理解用户最新提问，我还会提供给你一段会话历史以供参考，格式如下：[HumanMessage(content='xxx'), AIMessage(content='xxx'), ...]
     其中"HumanMessage"表示用户历史提问，"AIMessage"表示智能聊天系统的历史回答。
   - 会话历史仅用于帮助理解当前提问的背景和指代，你的判断对象是**用户最新提问**与**候选文档内容**之间的相关性。
   - 知识文档可能是叙述性、多主题或背景性内容，请聚焦其中**与当前问题最相关的片段**。
   - **宁可保留一条模糊但可能相关的信息，也不要轻易判定为“无效”**。

直接返回摘要或“无效的知识文档”，不要输出任何解释、前缀、格式标记或额外说明。
"""
llm_context_compressor_usr_prompt_template = env.from_string(
    "提供给你参考的会话历史内容如下：```{{provided_chat_history}}```"
    "\n\n\n给你的候选文档如下：```{{candidate_context}}```"
    "\n\n\n用户最新提问如下：```{{query}}```"
)


DEFAULT_INTENT_RECOGNITION_PROMPT_TEMPLATES = {k: v for k, v in globals().items() if "prompt_template" in k}
