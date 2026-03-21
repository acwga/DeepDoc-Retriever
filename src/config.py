"""项目配置文件"""
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 数据目录
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
INDEX_DIR = DATA_DIR / "index"
EVAL_DIR = DATA_DIR / "eval"

# 模型缓存目录
LOCAL_MODEL_DIR = DATA_DIR / "models"

# 模型名称配置
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
RERANK_MODEL_NAME = "BAAI/bge-reranker-base"

# 是否使用本地模型
USE_LOCAL_MODELS = True

def get_model_path(model_name: str) -> str:
    """
    获取模型路径，优先使用本地模型
    """
    if not USE_LOCAL_MODELS:
        return model_name
    
    # 映射模型名称到本地路径
    local_mappings = {
        "BAAI/bge-small-en-v1.5": LOCAL_MODEL_DIR / "bge-small-en-v1.5",
        "BAAI/bge-reranker-base": LOCAL_MODEL_DIR / "bge-reranker-base"
    }
    
    local_path = local_mappings.get(model_name)
    if local_path and local_path.exists():
        print(f"使用本地模型: {local_path}")
        return str(local_path)
    
    print(f"本地模型不存在，使用网络模型: {model_name}")
    return model_name