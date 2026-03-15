import json
from pathlib import Path
from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHUNKS_JSONL = PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"
VECTOR_DIR = PROJECT_ROOT / "data" / "index" / "vector"

EMBEDDINGS_NPY = VECTOR_DIR / "embeddings.npy"
META_JSON = VECTOR_DIR / "chunk_meta.jsonl"

EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"

def load_chunks(path: Path) -> List[Dict]:
    """
    从 JSONL 文件中加载文本块数据。
    """
    if not path.exists():
        raise FileNotFoundError(f"未找到文件: {path}")
    
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def build_vector_index() -> None:
    """
    构建向量索引并保存。
    """
    VECTOR_DIR.mkdir(parents=True, exist_ok=True)

    chunks = load_chunks(CHUNKS_JSONL)
    if not chunks:
        raise RuntimeError("未加载到任何文本块数据")
    
    texts = [c["text"] for c in chunks]

    model = SentenceTransformer(EMBED_MODEL_NAME)
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    embeddings = embeddings.astype(np.float32)
    
    np.save(EMBEDDINGS_NPY, embeddings)

    with META_JSON.open("w", encoding="utf-8") as f:
        for c in chunks:
            row = {
                "chunk_id": c["chunk_id"],
                "doc_id": c["doc_id"],
                "title": c["title"],
                "source": c["source"],
                "page": c["page"],
                "text": c["text"]
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("向量索引构建完成。")
    print("向量文件路径:", EMBEDDINGS_NPY)
    print("元数据文件路径:", META_JSON)

if __name__ == "__main__":
    build_vector_index()