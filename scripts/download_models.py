"""下载模型到本地"""
import os
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder

# 设置项目路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "data" / "models"

def download_models():
    """下载所有需要的模型到本地"""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=" * 50)
    print("开始下载嵌入模型: BAAI/bge-small-en-v1.5")
    print("=" * 50)
    
    # 下载嵌入模型
    embed_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    embed_path = MODEL_DIR / "bge-small-en-v1.5"
    embed_model.save(str(embed_path))
    print(f"✓ 嵌入模型已保存到: {embed_path}")
    
    print("\n" + "=" * 50)
    print("开始下载重排序模型: BAAI/bge-reranker-base")
    print("=" * 50)
    
    # 下载重排序模型
    rerank_model = CrossEncoder("BAAI/bge-reranker-base")
    rerank_path = MODEL_DIR / "bge-reranker-base"
    rerank_model.save_pretrained(str(rerank_path))
    print(f"✓ 重排序模型已保存到: {rerank_path}")
    
    print("\n" + "=" * 50)
    print("所有模型下载完成！")
    print(f"模型缓存目录: {MODEL_DIR}")
    print("=" * 50)

if __name__ == "__main__":
    download_models()