import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.documents import Document

import os
from dotenv import load_dotenv
load_dotenv()

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")

embeddings = DashScopeEmbeddings(
    dashscope_api_key=DASHSCOPE_API_KEY,
    model="text-embedding-v4"
)

# 计算嵌入维度（通过测试文本获取）
embedding_dim = len(embeddings.embed_query("hello world"))
# 创建 FAISS 索引：IndexFlatL2 是基础索引（精确检索，无量化，适合测试）
index = faiss.IndexFlatL2(embedding_dim)

# InMemoryDocstore：内存型文档存储（存储文档内容、元数据，无持久化）
# index_to_docstore_id：索引位置到文档 ID 的映射（FAISS 索引仅存向量，需关联文档）
vector_store = FAISS(
    embedding_function=embeddings,  # 嵌入模型（用于文本转向量）
    index=index,                    # FAISS 索引（用于向量检索）
    docstore=InMemoryDocstore(),    # 文档存储（内存型，重启丢失）
    index_to_docstore_id={},        # 索引-文档ID映射（初始为空）
)

# 构造示例文档（可替换为你的业务文档）
sample_docs = [
    Document(
        page_content="FAISS 是 Facebook 开发的高效向量检索库，主打高性能相似性搜索",
        metadata={"source": "tech_doc", "category": "vector_db", "author": "Facebook AI"}
    ),
    Document(
        page_content="LangChain 是构建 LLM 应用的框架，支持集成多种向量存储",
        metadata={"source": "framework_doc", "category": "llm_framework"}
    ),
    Document(
        page_content="向量数据库用于存储和检索高维度向量，核心场景是语义检索、推荐系统",
        metadata={"source": "industry_doc", "category": "vector_db", "tag": "application"}
    )
]

# 添加文档到向量库（自动完成：文本→向量→存入FAISS索引 + 文档存入Docstore + 映射关系更新）
vector_store.add_documents(documents=sample_docs)
print(f"向量库初始化完成，共存储 {vector_store.index.ntotal} 条向量")

# 检索与 "向量数据库的应用场景" 最相似的文档（返回 Top2）
query = "向量数据库的应用场景"
retrieval_results = vector_store.similarity_search(
    query=query,
    k=2,  # 返回最相似的2条
    filter={"category": "vector_db"}  # 可选：按元数据过滤（只检索分类为vector_db的文档）
)

# 打印检索结果
print("\n===== 相似检索结果 =====")
for i, doc in enumerate(retrieval_results, 1):
    print(f"\n【结果 {i}】")
    print(f"文档内容：{doc.page_content}")
    print(f"元数据：{doc.metadata}")

# 保存向量库到本地目录（LangChain 封装了 FAISS 索引 + Docstore 的持久化）
vector_store.save_local("faiss_db")
print("\n向量库已保存到本地：faiss_db 目录")

# 从本地加载向量库（重启程序后恢复）
loaded_vector_store = FAISS.load_local(
    folder_path="faiss_db",
    embeddings=embeddings,
    allow_dangerous_deserialization=True  # 测试场景开启，生产需注意安全
)

# 验证加载结果
loaded_results = loaded_vector_store.similarity_search(query="LangChain 集成向量存储", k=1)
print("\n===== 加载后检索结果 =====")
print(f"文档内容：{loaded_results[0].page_content}")