import json
import re
import pickle
from pathlib import Path
from typing import List, Dict, Iterable
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
INDEX_DIR = PROJECT_ROOT / "data" / "index" / "bm25"

DOCS_JSONL = PROCESSED_DIR / "docs.jsonl"
CHUNK_JSONL = PROCESSED_DIR / "chunks.jsonl"
BM25_PKL = INDEX_DIR / "bm25_index.pkl"

def normalize_text(text: str) -> str:
    """
    简单清洗, 去掉多余空白
    """
    text = text.replace("\u0000", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize_for_bm25(text: str) -> List[str]:
    """
    bm25简单分词
    """
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())

def make_doc_id_from_path(path: Path, raw_root: Path) -> str:
    """
    根据文档路径生成文档ID
    """
    rel = path.relative_to(raw_root).as_posix()
    rel = rel.replace("/", "__")
    rel = rel.rsplit(".", 1)[0]
    return rel

def read_txt_docs(raw_dir: Path) -> Iterable[Dict]:
    """
    递归读取txt文档
    """
    for txt_file in raw_dir.rglob("*.txt"):
        text = normalize_text(txt_file.read_text(encoding="utf-8", errors="ignore"))
        if not text:
            continue

        base_id = make_doc_id_from_path(txt_file, raw_dir)
        yield {
            "doc_id": base_id,
            "title": txt_file.name,
            "source": txt_file.as_posix(),
            "page": None,
            "text": text
        }

def read_pdf_docs(raw_dir: Path) -> Iterable[Dict]:
    """
    读取PDF文档, 每页作为一个文档
    """
    for pdf_file in raw_dir.rglob("*.pdf"):
        reader = PdfReader(str(pdf_file))
        base_id = make_doc_id_from_path(pdf_file, raw_dir)

        for page_no, page in enumerate(reader.pages, 1):
            page_text = normalize_text(page.extract_text() or "")
            if not page_text:
                continue

            yield {
                "doc_id": f"{base_id}__p{page_no}",
                "title": pdf_file.name,
                "source": pdf_file.as_posix(),
                "page": page_no,
                "text": page_text
            }

def build_splitter() -> RecursiveCharacterTextSplitter:
    """
    构建文本分块器
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
        keep_separator=False,
        length_function=len
    )

def build_chunks(docs: List[Dict]) -> List[Dict]:
    """
    将文档分块, 每块作为一个独立的文档
    """
    splitter = build_splitter()
    chunks: List[Dict] = []

    for doc in docs:
        pieces = splitter.split_text(doc["text"])
        for i, piece in enumerate(pieces, 1):
            piece = normalize_text(piece)
            if not piece:
                continue

            chunks.append(
                {
                    "chunk_id": f"{doc['doc_id']}__c{i}",
                    "doc_id": doc["doc_id"],
                    "title": doc["title"],
                    "source": doc["source"],
                    "page": doc["page"],
                    "text": piece
                }
            )

    return chunks

def write_jsonl(path: Path, rows: Iterable[Dict]):
    """
    将数据写入jsonl文件
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count

def build_bm25_index(chunks: List[Dict], output_path: Path) -> None:
    """
    构建bm25索引并保存
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tokenized_corpus = [tokenize_for_bm25(chunk["text"]) for chunk in chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    payload = {
        "bm25": bm25,
        "chunk_ids": [c["chunk_id"] for c in chunks],
        "chunks": chunks
    }

    with output_path.open("wb") as f:
        pickle.dump(payload, f)

def build():
    """
    构建语料库
    """
    docs: List[Dict] = []
    docs.extend(read_txt_docs(RAW_DIR))
    docs.extend(read_pdf_docs(RAW_DIR))

    if not docs:
        raise RuntimeError(f"未找到文档: {RAW_DIR}")
    
    chunks = build_chunks(docs)

    docs_n = write_jsonl(DOCS_JSONL, docs)
    chunks_n = write_jsonl(CHUNK_JSONL, chunks)
    build_bm25_index(chunks, BM25_PKL)

    print(f"语料库构建完成")
    print(f"文档数: {docs_n}, 存储路径: {DOCS_JSONL}")
    print(f"分块数: {chunks_n}, 存储路径: {CHUNK_JSONL}")
    print(f"BM25索引已保存: {BM25_PKL}")

if __name__ == "__main__":
    build()