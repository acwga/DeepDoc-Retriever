import json
from pathlib import Path
from typing import List, Dict
from src.qa_service import QASystem
from src.config import EVAL_DIR

TEST_JSON = EVAL_DIR / "test_set_final.json"

def load_test_set(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def evaluate_retriever(test_set: List[Dict], qa_system: QASystem, top_k: int = 5):
    total = len(test_set)
    hit_count = 0
    mrr_total = 0.0
    precision_total = 0.0

    for item in test_set:
        query = item["query"]
        # 获取相关文档ID集合
        relevant_docs = {item["relevant_docs"]}
        # 获取模型返回的文档ID列表
        reranked = qa_system.answer(query, eval_rerank=True)[1]
        retrieved_docs = [doc.get("chunk_id") for doc in reranked[:top_k]]

        # 计算命中率
        hit = any(doc_id in relevant_docs for doc_id in retrieved_docs)
        hit_count += int(hit)

        # 计算MRR
        for rank, doc_id in enumerate(retrieved_docs, 1):
            if doc_id in relevant_docs:
                mrr_total += 1.0 / rank
                break

        # 计算Precision@K
        relevant_retrieved = sum(1 for doc_id in retrieved_docs if doc_id in relevant_docs)
        precision_total += relevant_retrieved / top_k

    hit_rate = hit_count / total
    mrr = mrr_total / total
    precision_at_k = precision_total / total

    print(f"命中率: {hit_rate:.4f}")
    print(f"MRR: {mrr:.4f}")
    print(f"Precision@{top_k}: {precision_at_k:.4f}")

if __name__ == "__main__":
    qa = QASystem()
    qa.rewrite_llm
    test_set = load_test_set(TEST_JSON)
    evaluate_retriever(test_set, qa, top_k=5)