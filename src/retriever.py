import re
import numpy as np
import json
import pickle
from pathlib import Path
from typing import List, Dict
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BM25_PKL = PROJECT_ROOT / "data" / "index" / "bm25" / "bm25_index.pkl"
VECTOR_DIR = PROJECT_ROOT / "data" / "index" / "vector"
EMBEDDING_NPY = VECTOR_DIR / "embeddings.npy"
META_JSONL = VECTOR_DIR / "chunk_meta.jsonl"

class HybridRetriever:
    def __init__(self,
                 bm25_path: Path,
                 embed_path: Path,
                 meta_path: Path,
                 model_name: str = "BAAI/bge-small-en-v1.5",
                 vector_weight: float = 0.5,
                 bm25_weight: float = 0.5):
        self.bm25_path = bm25_path
        self.embed_path = embed_path
        self.meta_path = meta_path
        self.model_name = model_name
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        
        self.model = None
        self.bm25 = None
        self.embeddings = None
        self.chunks = []
        self.meta_rows = []

        self._load_all()

    def _load_all(self) -> None:
        # 加载向量模型
        self.model = SentenceTransformer(self.model_name)

        # 加载BM25索引
        if not self.bm25_path.exists():
            raise FileNotFoundError(f"BM25索引文件未找到: {self.bm25_path}")
        with self.bm25_path.open("rb") as f:
            payload = pickle.load(f)
        self.bm25 = payload["bm25"]
        self.chunks = payload["chunks"]

        # 加载向量索引和元数据
        if not self.embed_path.exists():
            raise FileNotFoundError(f"向量索引文件未找到: {self.embed_path}")
        self.embeddings = np.load(self.embed_path).astype(np.float32)

        if not self.meta_path.exists():
            raise FileNotFoundError(f"元数据文件未找到: {self.meta_path}")
        
        with self.meta_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.meta_rows.append(json.loads(line))

        if len(self.meta_rows) != self.embeddings.shape[0]:
            raise RuntimeError(
                f"元数据行数 {len(self.meta_rows)} 与向量数量 {self.embeddings.shape[0]} 不匹配"
            )
        
    def _tokenize_for_bm25(self, text: str) -> List[str]:
        """
        bm25简单分词
        """
        return re.findall(r"[a-zA-Z0-9_]+", text.lower())
    
    def _vector_search(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        向量检索
        """
        # 将查询转换为向量
        q_vec = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )[0].astype(np.float32)

        # 通过点积计算余弦相似度
        sims = self.embeddings @ q_vec
        top_idx = np.argsort(sims)[::-1][:top_k]

        # 获取结果
        results = []
        for i in top_idx:
            row = dict(self.meta_rows[i])
            row["id"] = row["chunk_id"]
            row["content"] = row["text"]
            row["score"] = float(sims[i])
            results.append(row)
        return results
    
    def _bm25_search(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        BM25检索
        """
        # 分词
        q_tokens = self._tokenize_for_bm25(query)
        if not q_tokens:
            return []
        
        # 计算BM25得分
        scores = self.bm25.get_scores(q_tokens)
        top_idx = np.argsort(scores)[::-1][:top_k]

        # 获取结果
        results = []
        for i in top_idx:
            row = dict(self.chunks[i])
            row["id"] = row["chunk_id"]
            row["content"] = row["text"]
            row["score"] = float(scores[i])
            results.append(row)
        return results
    
    def _minmax_normalize(self, results: List[Dict]) -> List[Dict]:
        """
        对得分进行归一化处理
        """
        if not results:
            return results
        
        scores = [r["score"] for r in results]
        s_min, s_max = min(scores), max(scores)

        if s_max == s_min:
            for r in results:
                r["score"] = 1.0
            return results
        
        for r in results:
            r["score"] = (r["score"] - s_min) / (s_max - s_min)
        return results
    
    def _hybrid_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        混合检索
        """
        # 向量检索
        if self.model:
            vector_results = self._vector_search(query, top_k * 2)
            vector_results = self._minmax_normalize(vector_results)
        else:
            vector_results = []

        # BM25检索
        if self.bm25:
            bm25_results = self._bm25_search(query, top_k * 2)
            bm25_results = self._minmax_normalize(bm25_results)
        else:
            bm25_results = []
        
        # 合并结果
        combined = {}

        for doc in vector_results:
            doc_id = doc.get("id", doc["content"][:50])
            combined[doc_id] = {**doc, "vector_score": doc["score"], "bm25_score": 0.0}

        for doc in bm25_results:
            doc_id = doc.get("id", doc["content"][:50])
            if doc_id in combined:
                combined[doc_id]["bm25_score"] = doc["score"]
            else:
                combined[doc_id] = {**doc, "vector_score": 0.0, "bm25_score": doc["score"]}

        # 计算综合分数
        for _, doc in combined.items():
            doc["score"] = self.vector_weight * doc["vector_score"] + self.bm25_weight * doc["bm25_score"]

        # 排序并返回前K个结果
        sorted_docs = sorted(combined.values(), key=lambda x: x["score"], reverse=True)
        return sorted_docs[:top_k]
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        对外接口：执行混合检索
        """
        return self._hybrid_search(query, top_k)

if __name__ == "__main__":
    retriever = HybridRetriever(
        bm25_path=BM25_PKL,
        embed_path=EMBEDDING_NPY,
        meta_path=META_JSONL,
        vector_weight=0.6,
        bm25_weight=0.4
    )

    query = "How to configure logging in python?"
    results = retriever.search(query, top_k=5)

    print(f"Query: {query}")
    for i, doc in enumerate(results, 1):
        print(f"[{i}] score: {doc['score']}, v: {doc['vector_score']:.4f}, b: {doc['bm25_score']:.4f}")
        print(f"    title: {doc.get('title')}, page: {doc.get('page')}, source: {doc.get('source')}")
        print(f"    content: {doc['content'][:100]}...")