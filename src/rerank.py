from typing import Dict, List
from sentence_transformers import CrossEncoder

class Reranker:
    def __init__(self,
                 model_name: str = "BAAI/bge-reranker-base",
                 batch_size: int = 32):
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = CrossEncoder(self.model_name)

    def rerank(self, query: str, candidates: List[Dict], top_k: int = 5) -> List[Dict]:
        """
        重排序候选项
        """
        if not candidates:
            return []
        
        # 构建输入对
        pairs = [(query, doc.get("content") or doc.get("text") or "") for doc in candidates]

        # 计算相似度分数
        scores = self.model.predict(pairs, batch_size=self.batch_size)

        # 将分数添加到候选项中
        reranked = []
        for i, doc in enumerate(candidates):
            item = dict(doc)
            item["rerank_score"] = scores[i]
            reranked.append(item)

        # 根据分数排序
        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]
    
if __name__ == "__main__":
    from src.retriever import HybridRetriever, BM25_PKL, EMBEDDING_NPY, META_JSONL

    retriever = HybridRetriever(
        bm25_path=BM25_PKL,
        embed_path=EMBEDDING_NPY,
        meta_path=META_JSONL,
        vector_weight=0.6,
        bm25_weight=0.4
    )

    query = "How to configure logging in python?"
    candidates = retriever.search(query, top_k=20)

    reranker = Reranker()
    results = reranker.rerank(query, candidates, top_k=5)

    print(f"Query: {query}")
    for i, doc in enumerate(results, 1):
        print(f"[{i}] rerank_score: {doc['rerank_score']:.4f}, rrf_score: {doc['rrf_score']:.4f}")
        print(f"    title: {doc.get('title')}, page: {doc.get('page')}, source: {doc.get('source')}")
        print(f"    content: {doc['content'][:100]}...")