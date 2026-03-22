# DeepDoc-Retriever - 技术文档智能问答系统

## 📌 项目简介
一个面向技术文档的智能问答系统，采用**混合检索（RRF）+ 重排序**技术，能够精准定位用户问题对应的文档片段，并生成准确答案。

## 🚀 核心技术
- **混合检索**：结合向量检索（BAAI/bge-small-en）和BM25，使用RRF算法融合结果
- **重排序**：CrossEncoder（BAAI/bge-reranker-base）优化结果排序
- **查询改写**：LLM优化用户问题，提高检索准确率
- **RAG架构**：基于检索增强生成，保证答案的准确性和可追溯性

## 📊 性能表现
在50个技术文档问题的测试集上：
- **命中率@10**：84%
- **MRR**：0.7180
- **Precision@10**：0.1680（单文档标注的理论最大值为0.2）

## 🛠️ 技术栈
- **语言模型**：通义千问 qwen3-max
- **向量模型**：SentenceTransformers（bge-small-en）
- **重排序**：CrossEncoder（bge-reranker-base）
- **查询重写**：qwen2.5:7b
- **检索框架**：混合检索（向量+BM25）+ RRF融合
- **前端界面**：Streamlit

## 📁 项目结构
```
DeepDoc-Retriever/
├── src/
│   ├── qa_service.py       # 问答系统主逻辑
│   ├── retriever.py        # 混合检索实现
│   ├── rerank.py           # 重排序实现
|   ├── prompts.py          # 存放提示词
|   └── config.py           # 配置文件
|
├── scripts/                    # 构建和工具脚本
│   ├── build_corpus.py         # 构建语料库和BM25索引
│   ├── build_vector_index.py   # 构建向量索引
│   ├── build_test_set.py       # 构建测试集
|   ├── download_models.py      # 下载模型到本地
|   ├── optimize_weights.py     # 寻找最优超参
│   └── eval_retrieval.py       # 检索评估
|
├── data/
│   ├── raw/                # 原始PDF/TXT文档
│   ├── processed/          # 处理后的文档块
│   ├── index/              # BM25和向量索引
|   └── eval/               # 评估测试集
├──  requirements           #配置文件
└──  app.py                 # Streamlit界面
```

## 💡 核心亮点
1. **混合检索策略**：同时利用语义相似度和关键词匹配，兼顾准确性和召回率
2. **RRF融合算法**：有效融合不同检索结果，提升排序质量
3. **查询优化**：LLM改写问题，提高与英文技术文档的匹配度
4. **可解释性**：答案附带引用文档，便于验证
5. **性能优化**：使用MD5作为文档块唯一标识，降低内存占用，提升检索速度
6. **最优参数搜索**: 通过网格搜索和精细搜索, 寻找最优超参

## 🎯 应用场景
- 技术文档智能客服
- 开发者文档问答助手
- 企业内部知识库检索
