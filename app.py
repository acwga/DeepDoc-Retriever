import streamlit as st
from src.qa_service import QASystem
import time

# 页面配置
st.set_page_config(
    page_title="TechDoc QA",
    page_icon="📚",
    layout="wide"
)

# 标题和说明
st.title("📚 技术文档问答系统")
st.caption("基于混合检索（RRF）+ 重排序（CrossEncoder）")

@st.cache_resource
def load_qa_system() -> QASystem:
    """
    加载问答系统实例
    """
    return QASystem()

# 初始化历史对话
if "history" not in st.session_state:
    st.session_state["history"] = []

# 侧边栏参数设置
with st.sidebar:
    st.header("参数设置")
    retrieve_k = st.slider("检索候选数", min_value=5, max_value=50, value=20, step=5)
    final_k = st.slider("最终返回数", min_value=1, max_value=10, value=5, step=1)
    show_content_chars = st.slider("片段内容显示长度", min_value=100, max_value=500, value=200, step=20)

query = st.text_input("请输入您的问题：")

if st.button("确定", type="primary"):
    if not query.strip():
        st.warning("输入的内容不能为空。")
    else:
        qa = load_qa_system()
        # 更新参数
        qa.retrieve_k = retrieve_k
        qa.final_k = final_k
        qa.context_max_chars = max(show_content_chars * final_k, 1000)

        # 将用户问题添加到历史对话
        st.session_state["history"].append({"role": "user", "content": query})

        # 生成答案
        with st.spinner("模型思考中, 请稍侯..."):
            start_time = time.time()
            answer_stream, reranked = qa.answer(query, history=st.session_state["history"])
            answer_placeholder = st.empty()
            answer_text = ""
            first_token = True
            for msg in answer_stream:
                if first_token:                              # 第一个 token 出现时
                    elapsed_time = time.time() - start_time
                    st.write(f"⏱️ 思考时间：{elapsed_time:.2f} 秒")
                    first_token = False
                answer_text += msg.content
                answer_placeholder.write(answer_text, unsafe_allow_html=True)
            # 将模型回答添加到历史对话
            st.session_state["history"].append({"role": "assistant", "content": answer_text})
            st.success("回答生成完毕！")

        # 显示历史对话
        st.subheader("历史对话")
        for msg in st.session_state["history"]:
            if msg["role"] == "user":
                st.markdown(f"**用户：** {msg['content']}")
            else:
                st.markdown(f"**模型：** {msg['content']}")

        # 显示相关文档
        st.subheader("参考文档")
        for i, doc in enumerate(reranked, 1):
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