import streamlit as st
from src.retriever import HybridRetriever, BM25_PKL, EMBEDDING_NPY, META_JSONL
from src.rerank import Reranker

st.set_page_config(
    page_title="TechDoc QA",
    page_icon="📚",
    layout="wide"
)

st.title("📚 技术文档问答系统")
st.caption("基于混合检索（RRF）+ 重排序（CrossEncoder）")

@st.cache_resource
def load_retriever() -> HybridRetriever:
    """
    加载混合检索器
    """
    return HybridRetriever(
        bm25_path=BM25_PKL,
        embed_path=EMBEDDING_NPY,
        meta_path=META_JSONL,
        vector_weight=0.6,
        bm25_weight=0.4,
        k=60
    )

@st.cache_resource
def load_reranker() -> Reranker:
    """
    加载重排序器
    """
    return Reranker()

def search_pipeline(query: str, retrieve_k: int = 20, final_k: int = 5):
    # 创建检索器和重排序器实例
    retriever = load_retriever()
    reranker = load_reranker()

    # 混合检索 + 重排序
    candidates = retriever.search(query, top_k=retrieve_k)
    results = reranker.rerank(query, candidates, top_k=final_k)
    return results

with st.sidebar:
    st.header("参数设置")
    retrieve_k = st.slider("检索候选数", min_value=5, max_value=50, value=20, step=5)
    final_k = st.slider("最终返回数", min_value=1, max_value=10, value=5, step=1)
    show_content_chars = st.slider("片段内容显示长度", min_value=100, max_value=500, value=200, step=20)

query = st.text_input("请输入您的问题：")

if st.button("开始检索", type="primary"):
    if not query.strip():
        st.warning("请输入问题后再检索。")
    else:
        with st.spinner("努力检索中, 请稍侯..."):
            docs = search_pipeline(query, retrieve_k=retrieve_k, final_k=final_k)

        st.success(f"检索完成！共找到 {len(docs)} 条相关文档。")

        for i, doc in enumerate(docs, 1):
            title = doc.get("title", "Unknown")
            page = doc.get("page", "N/A")
            source = doc.get("source", "N/A")
            rerank_score = doc.get("rerank_score", 0.0)
            rrf_score = doc.get("rrf_score", 0.0)
            content = (doc.get("content") or "")[:show_content_chars]

            with st.expander(f"[{i}] {title} | page={page}"):
                st.write(f"rerank_score: {rerank_score:.4f}")
                st.write(f"rrf_score: {rrf_score:.6f}")
                st.write(f"source: {source}")
                st.write(f"content:")
                st.write(content)