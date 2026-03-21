import re
import numpy as np
import json
import pickle
import hashlib
from pathlib import Path
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from src.config import (
    get_model_path,
    EMBED_MODEL_NAME,
    BM25_DIR,
    VECTOR_DIR
)

BM25_PKL = BM25_DIR / "bm25_index.pkl"
EMBEDDING_NPY = VECTOR_DIR / "embeddings.npy"
META_JSONL = VECTOR_DIR / "chunk_meta.jsonl"

class HybridRetriever:
    def __init__(self,
                 bm25_path: Path,
                 embed_path: Path,
                 meta_path: Path,
                 model_name: str = EMBED_MODEL_NAME,
                 vector_weight: float = 0.5,
                 bm25_weight: float = 0.5,
                 k: int = 60):
        self.bm25_path = bm25_path
        self.embed_path = embed_path
        self.meta_path = meta_path
        self.model_name = model_name
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.k = k
        
        self.model = None
        self.bm25 = None
        self.embeddings = None
        self.chunks = []
        self.meta_rows = []
        self.md5_to_meta = {}

        self._load_all()

    def _load_all(self) -> None:
        # 加载向量模型
        model_path = get_model_path(self.model_name)
        self.model = SentenceTransformer(model_path)

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
                meta = json.loads(line)
                self.meta_rows.append(meta)
                # 建立md5到元数据的映射
                self.md5_to_meta[meta["md5_id"]] = meta

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
            # 通过索引获取对应的元数据
            meta = self.meta_rows[i]
            row = dict(meta)
            row["id"] = meta["chunk_id"]
            row["md5_id"] = meta["md5_id"]
            row["content"] = meta["text"]
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
            row["md5_id"] = hashlib.md5(row["text"].encode("utf-8")).hexdigest()
            results.append(row)
        return results
    
    def _hybrid_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        混合检索
        """
        # 向量检索
        vector_results = self._vector_search(query, top_k * 2) if self.model else []

        # BM25检索
        bm25_results = self._bm25_search(query, top_k * 2) if self.bm25 else []
        
        # 合并结果
        combined = {}

        for rank, doc in enumerate(vector_results, 1):
            doc_id = doc.get("md5_id", doc["content"][:50])
            combined[doc_id] = {
                **doc,
                "vector_rank": rank,
                "bm25_rank": None
            }

        for rank, doc in enumerate(bm25_results, 1):
            doc_id = doc.get("md5_id", doc["content"][:50])
            if doc_id in combined:
                combined[doc_id]["bm25_rank"] = rank
            else:
                combined[doc_id] = {
                    **doc,
                    "vector_rank": None,
                    "bm25_rank": rank
                }

        # 计算综合分数
        for _, doc in combined.items():
            v_score = self.vector_weight * 1.0 / (self.k + doc["vector_rank"]) if doc["vector_rank"] else 0.0
            b_score = self.bm25_weight * 1.0 / (self.k + doc["bm25_rank"]) if doc["bm25_rank"] else 0.0
            doc["rrf_score"] = v_score + b_score

        # 排序并返回前K个结果
        sorted_docs = sorted(combined.values(), key=lambda x: x["rrf_score"], reverse=True)
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
        print(f"[{i}] score: {doc['rrf_score']}, v: {doc['vector_rank']}, b: {doc['bm25_rank']}")
        print(f"    title: {doc.get('title')}, page: {doc.get('page')}, source: {doc.get('source')}")
        print(f"    content: {doc['content'][:100]}...")