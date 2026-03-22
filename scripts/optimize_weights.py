import json
import itertools
from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd
from tqdm import tqdm
from src.qa_service import QASystem
from src.config import EVAL_DIR
from langchain_community.chat_models import ChatOllama

TEST_JSON = EVAL_DIR / "test_set_final.json"
base_qa = QASystem()
base_qa.rewrite_llm = ChatOllama(model="qwen2.5:7b", temperature=0)

def load_test_set(path: Path):
    """加载测试集"""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def evaluate_single_config(test_set: List[Dict], 
                          vector_weight: float, 
                          bm25_weight: float, 
                          k: int,
                          top_k: int = 5,
                          base_qa: QASystem = base_qa,) -> Tuple[float, float, float]:
    """
    评估单个配置的性能
    """
    # 创建使用指定权重的 QA 系统
    base_qa.retriever.vector_weight = vector_weight
    base_qa.retriever.bm25_weight = bm25_weight
    base_qa.retriever.k = k

    total = len(test_set)
    hit_count = 0
    mrr_total = 0.0
    precision_total = 0.0
    
    for item in test_set:
        query = item["query"]
        relevant_docs = {item["relevant_docs"]}
        
        # 获取检索结果
        reranked = base_qa.answer(query, eval_rerank=True)[1]
        retrieved_docs = [doc.get("chunk_id") for doc in reranked[:top_k]]
        
        # 计算命中率
        hit = any(doc_id in relevant_docs for doc_id in retrieved_docs)
        hit_count += int(hit)
        
        # 计算 MRR
        for rank, doc_id in enumerate(retrieved_docs, 1):
            if doc_id in relevant_docs:
                mrr_total += 1.0 / rank
                break
        
        # 计算 Precision@K
        relevant_retrieved = sum(1 for doc_id in retrieved_docs if doc_id in relevant_docs)
        precision_total += relevant_retrieved / top_k
    
    hit_rate = hit_count / total
    mrr = mrr_total / total
    precision_at_k = precision_total / total
    
    return hit_rate, mrr, precision_at_k

def grid_search(test_set: List[Dict],
                vector_weights: List[float] = None,
                k_values: List[int] = None,
                top_k: int = 5,
                file_name: str = None) -> pd.DataFrame:
    """
    网格搜索最优参数
    
    Args:
        test_set: 测试集
        vector_weights: 向量权重列表（BM25权重自动计算为 1-vector_weight）
        k_values: RRF 平滑系数列表
        top_k: 评估时使用的 top K
        file_name: 保存结果的文件名
    """
    if vector_weights is None:
        vector_weights = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    
    if k_values is None:
        k_values = [40, 60, 80, 100, 120, 140]
    
    print("=" * 80)
    print("开始网格搜索")
    print(f"向量权重范围: {vector_weights}")
    print(f"平滑系数范围: {k_values}")
    print(f"总配置数: {len(vector_weights) * len(k_values)}")
    print("=" * 80)
    
    results = []
    
    # 网格搜索
    for vector_weight, k in tqdm(list(itertools.product(vector_weights, k_values)), 
                                  desc="搜索配置"):
        bm25_weight = 1 - vector_weight
        
        # 评估当前配置
        hit_rate, mrr, precision = evaluate_single_config(
            test_set, vector_weight, bm25_weight, k, top_k
        )
        
        results.append({
            "vector_weight": vector_weight,
            "bm25_weight": bm25_weight,
            "k": k,
            "hit_rate": hit_rate,
            "mrr": mrr,
            "precision_at_k": precision,
            "score": mrr  # 使用 MRR 作为主要优化目标
        })
        
        # 打印中间结果
        if len(results) % 5 == 0:
            current_best = max(r['mrr'] for r in results)
            print(f"已评估 {len(results)} 个配置，当前最佳 MRR: {current_best:.4f}")
    
    # 转换为 DataFrame 并排序
    df = pd.DataFrame(results)
    df = df.sort_values("score", ascending=False)
    
    # 保存结果
    if file_name is None:
        file_name = "grid_search_results.csv"
    csv_path = EVAL_DIR / file_name
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    
    # 打印最优结果
    print("\n" + "=" * 80)
    print("最优配置 (按 MRR 排序):")
    print("=" * 80)
    print(df.head(10).to_string())
    
    best = df.iloc[0]
    print(f"\n最优配置:")
    print(f"  vector_weight: {best['vector_weight']:.2f}")
    print(f"  bm25_weight: {best['bm25_weight']:.2f}")
    print(f"  k: {int(best['k'])}")
    print(f"  Hit Rate: {best['hit_rate']:.4f}")
    print(f"  MRR: {best['mrr']:.4f}")
    print(f"  Precision@{top_k}: {best['precision_at_k']:.4f}")
    print(f"\n结果已保存到: {csv_path}")
    
    return df

def fine_tune_search(test_set: List[Dict],
                     best_vector_weight: float,
                     best_k: int,
                     vector_step: float = 0.05,
                     k_step: int = 5,
                     top_k: int = 5,
                     file_name: str = None) -> pd.DataFrame:
    """
    基于最优参数进行精细搜索
    
    Args:
        test_set: 测试集
        best_vector_weight: 粗搜索得到的最优向量权重
        best_k: 粗搜索得到的最优平滑系数
        vector_step: 向量权重的步长
        k_step: 平滑系数的步长
        top_k: 评估时使用的 top K
        file_name: 保存结果的文件名
    Returns:
        精细搜索结果的 DataFrame
    """
    print("\n" + "=" * 80)
    print("开始精细搜索")
    print(f"基于最优参数: vector_weight={best_vector_weight:.3f}, k={best_k}")
    print("=" * 80)
    
    # 生成精细搜索范围
    fine_weights = [
        best_vector_weight - vector_step * 2,
        best_vector_weight - vector_step,
        best_vector_weight,
        best_vector_weight + vector_step,
        best_vector_weight + vector_step * 2
    ]
    
    fine_ks = [
        best_k - k_step * 3,
        best_k - k_step * 2,
        best_k - k_step,
        best_k,
        best_k + k_step,
        best_k + k_step * 2,
        best_k + k_step * 3
    ]
    
    print(f"精细搜索范围:")
    print(f"  vector_weight: {fine_weights}")
    print(f"  k: {fine_ks}")
    print(f"  总配置数: {len(fine_weights) * len(fine_ks)}")
    
    # 执行精细搜索
    fine_results = grid_search(
        test_set,
        vector_weights=fine_weights,
        k_values=fine_ks,
        top_k=top_k,
        file_name=file_name
    )
    
    return fine_results

if __name__ == "__main__":
    # 加载测试集
    print("加载测试集...")
    test_set = load_test_set(TEST_JSON)
    print(f"测试集大小: {len(test_set)}")
    
    # 阶段1: 粗粒度网格搜索
    print("\n阶段1: 粗粒度搜索")
    coarse_results = grid_search(
        test_set,
        vector_weights = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        k_values=[40, 60, 80, 100, 120, 140],
        top_k=5,
        file_name="grid_search_results.csv"
    )

    
    # 阶段2: 在最优区域精细搜索
    print("\n" + "=" * 80)
    print("是否进行精细搜索？(y/n)")
    choice = input().strip().lower()
    
    if choice == 'y':
        best_v = coarse_results.iloc[0]['vector_weight']
        best_k = coarse_results.iloc[0]['k']
        # 精细搜索
        fine_results = fine_tune_search(
            test_set,
            best_vector_weight=best_v,
            best_k=best_k,
            vector_step=0.05,
            k_step=5,
            top_k=5,
            file_name="fine_tuning_results.csv"
        )
    
    print("\n优化完成！")