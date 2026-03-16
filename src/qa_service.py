import os
from typing import List, Dict, Iterator, Optional
from langchain_community.chat_models import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage
from src.retriever import HybridRetriever, BM25_PKL, EMBEDDING_NPY, META_JSONL
from src.rerank import Reranker


class QASystem:
    """
    技术文档文档服务
    """
    
    def __init__(self,
                 retrieve_k: int = 30,
                 final_k : int = 5,
                 vector_weight: float = 0.7,
                 bm25_weight: float = 0.3,
                 k: int = 60,
                 context_max_chars: int = 2000
                 ):
        self.retrieve_k = retrieve_k
        self.final_k = final_k
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.k = k
        self.context_max_chars = context_max_chars

        self.llm = ChatTongyi(model="qwen3-max")
        self.retriever = HybridRetriever(
            bm25_path=BM25_PKL,
            embed_path=EMBEDDING_NPY,
            meta_path=META_JSONL,
            vector_weight=self.vector_weight,
            bm25_weight=self.bm25_weight,
            k=self.k
        )
        self.reranker = Reranker()
        self.rewrite_prompt = ChatPromptTemplate.from_messages([
            (
                "system", 
                "你是一个检索助手。将用户问题改写成更适合英文技术文档检索的查询。"
                "要求：\n"
                "1. 保持原问题的核心意图\n"
                "2. 使用英文技术文档中常见的术语\n"
                "3. 去掉多余的语气词和修饰词\n"
                "4. 只输出英文查询，不要解释"
            ),
            ("human", "{query}")
        ])
        self.answer_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """
                你是一个技术文档问答助手。
                请只根据下方提供的文档内容回答用户问题。
                如果内容不足，请明确说明无法回答。

                要求：
                1. 答案简明、准确。
                2. 在答案中引用文档编号，如 [1], [2]。
                3. 答案后添加“参考文档”部分，列出用到的编号。
                4. 若用户没有明确要求，请用中文回答。 """
            ),
            ("human", "用户问题：{query}\n文档内容：{context}")
        ])

    def _rewrite_query(self, query: str) -> str:
        """
        查询重写
        """
        messages = self.rewrite_prompt.format_messages(query=query)
        response = self.llm.invoke(messages)
        return response.content.strip()
    
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
            messages = [system_msg]
            for msg in history:
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    messages.append(AIMessage(content=msg["content"]))
            # 添加当前消息
            messages.extend(user_msgs)

        return self.llm.stream(messages)
    
    def answer(self, query: str, 
               history: Optional[List[Dict]] = None,
               eval_rerank: bool = False
               ) -> tuple[Iterator[AIMessage], List[Dict]]:
        """
        返回答案生成器
        """
        query = self._rewrite_query(query)
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