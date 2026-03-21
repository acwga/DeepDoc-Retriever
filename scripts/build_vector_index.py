import json
from pathlib import Path
from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
import hashlib
from src.config import (
    get_model_path,
    EMBED_MODEL_NAME,
    PROCESSED_DIR,
    VECTOR_DIR
)

CHUNKS_JSONL = PROCESSED_DIR / "chunks.jsonl"
EMBEDDINGS_NPY = VECTOR_DIR / "embeddings.npy"
META_JSONL = VECTOR_DIR / "chunk_meta.jsonl"

def get_md5(text: str) -> str:
    """
    获取文本的 MD5 哈希值。
    """
    return hashlib.md5(text.encode("utf-8")).hexdigest()

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

def load_existing_meta(meta_path: Path):
    """
    加载已有的元数据文件和嵌入数组。
    """
    if not meta_path.exists() or not EMBEDDINGS_NPY.exists():
        return set(), None, []
    
    # 加载已有嵌入
    existing_embeddings = np.load(EMBEDDINGS_NPY).astype(np.float32)

    # 加载已有元数据
    existing_meta_rows = []
    md5_set = set()
    with meta_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            meta = json.loads(line)
            existing_meta_rows.append(meta)
            md5_set.add(meta["md5_id"])

    return md5_set, existing_embeddings, existing_meta_rows

def build_vector_index() -> None:
    """
    构建向量索引并保存。
    """
    VECTOR_DIR.mkdir(parents=True, exist_ok=True)

    # 加载文本块数据
    chunks = load_chunks(CHUNKS_JSONL)
    if not chunks:
        raise RuntimeError("未加载到任何文本块数据")
    
    # 加载现有索引
    existing_md5_set, existing_embeddings, existing_meta_rows = load_existing_meta(META_JSONL)

    # 找出需要处理的新文本块
    new_chunks = []
    for chunk in chunks:
        md5 = get_md5(chunk["text"])
        if md5 not in existing_md5_set:
            new_chunks.append(chunk)
    
    if not new_chunks:
        print("没有新增的文本块，无需更新向量索引。")
        return

    # 生成新文本块的嵌入
    model_path = get_model_path(EMBED_MODEL_NAME)
    model = SentenceTransformer(model_path)
    new_texts = [c["text"] for c in new_chunks]
    new_embeddings = model.encode(
        new_texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    new_embeddings = new_embeddings.astype(np.float32)

    # 合并嵌入
    if existing_embeddings is not None:
        combined_embeddings = np.vstack([existing_embeddings, new_embeddings])
    else:
        combined_embeddings = new_embeddings
    
    np.save(EMBEDDINGS_NPY, combined_embeddings)

    # 构建新元数据行
    new_meta_rows = []
    for chunk in new_chunks:
        md5 = get_md5(chunk["text"])
        row = {
            "md5_id": md5,
            "chunk_id": chunk["chunk_id"],
            "doc_id": chunk["doc_id"],
            "title": chunk["title"],
            "source": chunk["source"],
            "page": chunk["page"],
            "text": chunk["text"]
        }
        new_meta_rows.append(row)

    # 合并元数据
    all_meta_rows = existing_meta_rows + new_meta_rows

    # 保存元数据
    with META_JSONL.open("w", encoding="utf-8") as f:
        for meta in all_meta_rows:
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")

    print("向量索引构建完成。")
    print(f"新增文本块数量: {len(new_chunks)}")
    print(f"总文本块数量: {len(all_meta_rows)}")
    print("向量文件路径:", EMBEDDINGS_NPY)
    print("元数据文件路径:", META_JSONL)

if __name__ == "__main__":
    build_vector_index()