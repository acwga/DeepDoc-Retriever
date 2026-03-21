import random
import pandas as pd
import json
from pathlib import Path
from langchain_community.chat_models import ChatTongyi
from src.config import DATA_DIR

CHUNK_JSONL = DATA_DIR / "processed" / "chunks.jsonl"
OUTPUT_DIR = DATA_DIR / "eval"

def generate_test_set(chunk_path: Path = CHUNK_JSONL,
                   num_candidates: int = 50,
                   output_dir: Path = OUTPUT_DIR):
    """
    测试集生成器
    """
    # 构建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载文档块
    print("加载文档块...")
    with chunk_path.open("r", encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f if line.strip()]
    print(f"共加载 {len(chunks)} 个文档块")

    # 随机采样
    sampled = random.sample(chunks, min(num_candidates, len(chunks)))

    # 初始化模型
    llm = ChatTongyi(model="qwen3-max")

    # 生成候选问题
    print(f"生成 {len(sampled)} 个候选问题...")
    candidates = []
    for i, chunk in enumerate(sampled, 1):
        try:
            prompt = f"""根据这段技术文档生成1个自然的技术问题：
文档内容：{chunk['text'][:300]}
注意：只输出问题，不要其他内容。"""
            
            response = llm.invoke(prompt)
            query = response.content.strip()

            candidates.append({
                "id": f"q{i:03d}",
                "query": query,
                "relevant_docs": chunk["chunk_id"]
            })

            if i % 10 == 0:
                print(f"已生成 {i}/{len(sampled)} 个问题...")
        
        except Exception as e:
            print(f"生成问题失败: {e}")

    # 保存为CSV
    df = pd.DataFrame(candidates)
    csv_path = output_dir / "test_set_review.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    
    print(f"测试集已保存到 {csv_path}")
    print("请使用Excel打开, 进行人工审核")

    return csv_path

def build_final_test_set(review_file: Path):
    """
    构建最终测试集
    """
    df = pd.read_csv(review_file, encoding="utf-8-sig")

    test_set = []
    for _, row in df.iterrows():
        test_set.append({
            "id": row["id"],
            "query": row["query"],
            "relevant_docs": row["relevant_docs"]
        })
    
    # 保存最终测试集
    output_path = Path(review_file).parent / "test_set_final.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(test_set, f, indent=2, ensure_ascii=False)
    
    print(f"最终测试集已保存到 {output_path}")
    print(f"共 {len(test_set)} 条测试样例")

    return test_set

if __name__ == "__main__":
    # csv_file = generate_test_set(num_candidates=50)
    build_final_test_set(OUTPUT_DIR / "test_set_review.csv")