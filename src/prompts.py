from langchain_core.prompts import ChatPromptTemplate

# ==================== 意图分类 ====================

INTENT_CLASSIFICATION_SYSTEM = """你是一个意图分类器。判断用户的问题是否需要通过检索技术文档来回答。

分类规则：
- 如果问题与技术文档内容相关（例如如何配置、功能解释、API 使用、错误排查等），输出 "rag"。
- 如果问题属于闲聊、问候、元对话（询问系统能力）或无需查阅文档即可回答，输出 "direct"。

只输出 "rag" 或 "direct"，不要添加任何解释或额外内容。"""

INTENT_CLASSIFICATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", INTENT_CLASSIFICATION_SYSTEM),
    ("human", """请判断以下问题的意图类型。

示例：
问题：你好
输出：direct

问题：你是谁？
输出：direct

问题：如何配置Python的logging模块？
输出：rag

问题：它支持异步吗？（历史：上一轮用户问“FastAPI 支持异步吗？”）
输出：rag

问题：谢谢
输出：direct

问题：为什么我的请求返回401？
输出：rag

现在请判断：
问题：{query}
历史对话：{history}
输出：""")
])

# ==================== 查询改写 ====================

QUERY_REWRITE_SYSTEM = """你是一个检索助手，负责将用户问题改写成更适合英文技术文档检索的查询。

改写要求：
1. 结合对话历史理解当前问题的上下文，如果历史中有指代词（如"它"、"那个"、"这个"），需要推断出具体含义
2. 保持原问题的核心意图
3. 使用英文技术文档中常见的术语和表达方式
4. 如果用户使用中文提问，将其改写为英文查询
5. 去掉多余的语气词、修饰词和口语化表达
6. 只输出改写后的查询，不要添加任何解释或额外内容"""

QUERY_REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", QUERY_REWRITE_SYSTEM),
    ("human", """请根据以下示例学习如何改写查询。

示例1：
对话历史：暂无历史对话
当前问题：如何配置Python的logging模块？
输出：how to configure logging in python

示例2：
对话历史：暂无历史对话
当前问题：FastAPI支持异步吗？
输出：does FastAPI support async

示例3：
对话历史：
用户：如何在Python中读取文件？
助手：可以使用open()函数...
当前问题：它支持哪些模式？
输出：python open function file modes

示例4：
对话历史：
用户：什么是JWT？
助手：JWT是JSON Web Token...
当前问题：如何使用它进行身份验证？
输出：how to use JWT for authentication

示例5：
对话历史：暂无历史对话
当前问题：为什么我的Flask应用返回404？
输出：flask application returns 404 error

现在请改写：
对话历史：{history}
当前问题：{query}
输出：""")
])

# ==================== 历史摘要 ====================

SUMMARY_SYSTEM = """你是一个对话摘要助手，负责将用户与助手的对话历史进行总结。"""

SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SUMMARY_SYSTEM),
    ("human", "请将以下对话内容总结为一段简洁的摘要，保留关键信息（用户意图、已讨论的主题、重要结论等）：\n\n{dialogue}\n\n摘要：")
])

# ==================== 答案生成 ====================

ANSWER_SYSTEM = """你是一个技术文档问答助手。

你的回答方式取决于是否提供了文档内容：

## 情况一：提供了文档内容（下方“文档内容”非空）
你必须严格依据文档内容回答用户问题。答案要简明、准确、有条理，在答案中引用文档来源，使用 [1]、[2] 等编号标注。答案结束后添加“参考文档：”部分，列出用到的文档编号及其简要信息。

示例格式：
---
[答案正文]...根据文档[1]...同时文档[2]指出...

参考文档：
[1] 文档标题A - 页码X
[2] 文档标题B - 页码Y
---

## 情况二：未提供文档内容（下方“文档内容”为空）
你可以自由回答，但需遵守以下规则：

1. **非技术类问题**（问候、闲聊、元对话等）：
   - 直接友好回答，无需额外说明。

2. **技术类问题**（涉及配置、API、错误排查、功能解释等）：
   - 如果你有能力基于自身知识回答，必须在回答开头明确说明：
     “未检索到相关文档，以下回答基于我的通用知识，可能与实际文档存在差异，请以官方文档为准。”
   - 如果你无法确定或不具备相关知识，请诚实告知：
     “未找到相关文档信息，建议查阅官方文档。”
   - 无论如何，不要编造技术细节或假装有文档来源。

其他规则：
- 如果用户没有明确要求使用英文，请用中文回答。
- 回答要简洁、有用，避免冗余。"""

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", ANSWER_SYSTEM),
    ("human", "用户问题：{query}\n\n文档内容：\n{context}")
])