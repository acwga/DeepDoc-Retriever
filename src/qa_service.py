from typing import List, Dict, Tuple, Iterator, Optional
from langchain_community.chat_models import ChatTongyi
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.chat_models import ChatOllama
from src.retriever import HybridRetriever, BM25_PKL, EMBEDDING_NPY, META_JSONL
from src.rerank import Reranker
from src.prompts import QUERY_REWRITE_PROMPT, ANSWER_PROMPT, SUMMARY_PROMPT

class QASystem:
    """
    技术文档文档服务
    """
    
    def __init__(self,
                 retrieve_k: int = 30,
                 final_k : int = 5,
                 vector_weight: float = 0.5,
                 bm25_weight: float = 0.5,
                 k: int = 60,
                 context_max_chars: int = 2000,
                 max_window_size: int = 5,
                 summary_trigger: int = 10,
                 summary_llm = None
                 ):
        self.retrieve_k = retrieve_k    # 初始检索候选文档数量
        self.final_k = final_k          # 最终用于生成答案的文档数量
        self.vector_weight = vector_weight  # 向量检索权重
        self.bm25_weight = bm25_weight      # BM25检索权重
        self.k = k  # 平滑参数
        self.context_max_chars = context_max_chars  # 上下文最大字符数限制
        self.max_window_size = max_window_size  # 滑动窗口大小
        self.summary_trigger = summary_trigger  # 触发历史摘要的对话轮数阈值

        self.rewrite_llm = ChatOllama(model="qwen2.5:7b", temperature=0.1)  # 查询改写模型
        self.summary_llm = summary_llm or self.rewrite_llm  # 历史摘要模型, 默认与查询改写相同
        self.answer_llm = ChatTongyi(model="qwen3-max") # 答案生成模型

        self.retriever = HybridRetriever(
            bm25_path=BM25_PKL,
            embed_path=EMBEDDING_NPY,
            meta_path=META_JSONL,
            vector_weight=self.vector_weight,
            bm25_weight=self.bm25_weight,
            k=self.k
        )   # 混合检索器
        self.reranker = Reranker()  # 重排序器

        self.rewrite_prompt = QUERY_REWRITE_PROMPT  # 查询改写提示词
        self.answer_prompt = ANSWER_PROMPT  # 答案生成提示词
        self.summary_prompt = SUMMARY_PROMPT    # 历史摘要提示词

    def _summarize_history(self, history: List[Dict]) -> str:
        """
        对对话历史进行摘要
        """
        if not history:
            return ""
        # 构建对话文本
        dialogue = []
        for msg in history:
            role = "用户" if msg["role"] == "user" else "助手"
            dialogue.append(f"{role}: {msg['content']}")
        dialogue_text = "\n".join(dialogue)
        message = self.summary_prompt.format_messages(dialogue=dialogue_text)
        try:
            response = self.summary_llm.invoke(message)
            return response.content.strip()
        except Exception as e:
            print(f"生成摘要失败: {e}")
            return ""
        
    def _get_contextual_history(self, history: Optional[List[Dict]]) -> str:
        """
        获取带有摘要的上下文历史
        """
        if not history:
            return "暂无历史对话"
        
        # 如果历史长度超过阈值，生成摘要并只保留最近窗口
        if len(history) > self.summary_trigger:
            recent_turns = self.max_window_size * 2
            to_summarize = history[:-recent_turns] if recent_turns < len(history) else []
            recent = history[-recent_turns:] if recent_turns > 0 else []

            # 生成摘要
            summary = self._summarize_history(to_summarize) if to_summarize else ""
            # 组合： 摘要 + 最近对话
            recent_str = self._format_history_for_rewrite(
                history=recent, max_turns=self.max_window_size
            ) if recent else ""
            if summary:
                history_str = f"【历史摘要】{summary}\n\n【最近对话】\n{recent_str}"
            else:
                history_str = recent_str
        else:
            history_str = self._format_history_for_rewrite(
                history=history, max_turns=self.max_window_size
            )
        return history_str

    def _format_history_for_rewrite(self, history: Optional[List[Dict]], max_turns: int = 5) -> str:
        """
        将历史对话格式化为改写用的字符串
        """
        if not history:
            return "暂无历史对话"
        
        recent_history = history[-max_turns * 2:]
        
        formatted = []
        for msg in recent_history:
            role = "用户" if msg["role"] == "user" else "助手"
            formatted.append(f"{role}: {msg['content']}")
        
        return "\n".join(formatted)

    def _rewrite_query(self, query: str, history: Optional[List[Dict]] = None) -> str:
        """
        查询重写
        """
        try:
            history_str = self._get_contextual_history(history)
            messages = self.rewrite_prompt.format_messages(
                query=query, 
                history=history_str
            )
            response = self.rewrite_llm.invoke(messages)
            rewritten = response.content.strip('"').strip("'").strip()
            
            return rewritten if rewritten else query
        
        except Exception as e:
            print(f"查询改写失败: {e}")
            return query
    
    def _retrieve_docs(self, query: str) -> List[Dict]:
        """
        混合检索相关文档
        """
        return self.retriever.search(query, top_k=self.retrieve_k)
    
    def _rerank_docs(self, query: str, candidates: List[Dict]) -> List[Dict]:
        """
        重排序候选文档
        """
        return self.reranker.rerank(query, candidates, top_k=self.final_k)
    
    def _build_context(self, docs: List[Dict]) -> str:
        """
        构建上下文
        """
        if not docs:
            return ""
        
        blocks = []
        total = 0
        for i, doc in enumerate(docs, 1):
            title = doc.get("title", "Unknown")
            page = doc.get("page", "N/A")
            source = doc.get("source", "N/A")
            content = doc.get("content", "")
            block = (
                f"[{i}] title: {title}\n"
                f"source: {source}\n"
                f"page: {page}\n"
                f"content: {content}\n"
            )
            if total + len(block) > self.context_max_chars:
                break
            blocks.append(block)
            total += len(block)
        return "\n".join(blocks)
    
    
    def _generate_answer(self, query: str, docs: List[Dict], 
                         history: Optional[List[Dict]] = None) -> Iterator[AIMessage]:
        """
        生成答案
        """
        context = self._build_context(docs)
        # 构造历史消息
        # 先把系统消息分离出来, 放在最前面
        messages = self.answer_prompt.format_messages(query=query, context=context)
        if history:
            system_msg = messages[0]
            user_msgs = messages[1:]
            # 重构消息列表: 系统消息 + 历史消息 + 当前用户消息
            # 系统消息
            messages = [system_msg]
            # 历史消息
            history_str = self._get_contextual_history(history)
            if history_str.strip():
                messages.append(HumanMessage(content=f"对话历史：\n{history_str}"))
            # 当前消息
            messages.extend(user_msgs)

        return self.answer_llm.stream(messages)
    
    def answer(self, query: str, 
               history: Optional[List[Dict]] = None,
               eval_rerank: bool = False
               ) -> Tuple[Optional[Iterator[AIMessage]], List[Dict]]:
        """
        返回答案生成器
        """
        query = self._rewrite_query(query, history)
        candidates = self._retrieve_docs(query)
        reranked = self._rerank_docs(query, candidates)
        # 如果是评测重排序效果, 则直接返回重排序结果, 不生成答案
        if eval_rerank:
            return None, reranked
        return self._generate_answer(query, reranked, history), reranked

if __name__ == "__main__":
    qa = QASystem()
    test_query = input("请输入测试问题：")
    print("\n--- 答案 ---\n")
    answer_stream, reranked = qa.answer(test_query)
    for msg in answer_stream:
        print(msg.content, end="", flush=True)
    print("\n\n--- 完成 ---\n")