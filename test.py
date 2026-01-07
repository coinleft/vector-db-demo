from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.chat_models import ChatTongyi
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
import os

from dotenv import load_dotenv
load_dotenv()

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
embeddings = DashScopeEmbeddings(
    model="text-embedding-v4",
    dashscope_api_key=DASHSCOPE_API_KEY
)

import shutil
import os

# 检查持久化目录是否存在，如果存在则删除（重新开始）
if os.path.exists("./chroma_langchain_db"):
    print("检测到已有数据，正在清空...")
    shutil.rmtree("./chroma_langchain_db")
    print("数据已清空\n")

# 创建 Chroma 向量存储（在清空后创建，避免连接问题）
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",
)

# 示例文档
documents = [
    Document(
        page_content="Python 是一种高级编程语言，广泛用于数据科学和机器学习。",
        metadata={"source": "tutorial", "topic": "programming"}
    ),
    Document(
        page_content="机器学习是人工智能的一个分支，通过算法让计算机从数据中学习。",
        metadata={"source": "tutorial", "topic": "ai"}
    ),
    Document(
        page_content="向量数据库可以高效地存储和检索高维向量数据。",
        metadata={"source": "tutorial", "topic": "database"}
    ),
    Document(
        page_content="LangChain 是一个用于构建 LLM 应用的框架。",
        metadata={"source": "tutorial", "topic": "framework"}
    ),
    Document(
        page_content="Chroma 是一个开源的向量数据库，专为 AI 应用设计。",
        metadata={"source": "tutorial", "topic": "database"}
    ),
]

print("正在添加文档到向量存储...")
vector_store.add_documents(documents)
print(f"成功添加 {len(documents)} 个文档\n")
print("=" * 50)

# # 进行相似度搜索
# query = "什么是向量数据库？"
# print(f"查询：{query}")

# similar_docs = vector_store.similarity_search(query, k=3)

# print(f"\n找到 {len(similar_docs)} 个相似文档：\n")
# for i, doc in enumerate(similar_docs, 1):
#     print(f"文档 {i}:")
#     print(f"  内容: {doc.page_content}")
#     print(f"  元数据: {doc.metadata}")
#     print()

# # 带过滤条件的搜索示例
# print("=" * 50)
# print("带过滤条件的搜索（source='tutorial', topic='database'）:")
# filtered_docs = vector_store.similarity_search(
#     "数据库相关的内容",
#     k=2,
#     filter={"$and": [{"source": "tutorial"}, {"topic": "database"}]}
# )
# for i, doc in enumerate(filtered_docs, 1):
#     print(f"\n文档 {i}: {doc.page_content}")
#     print(f"元数据: {doc.metadata}")


chatLLM = ChatTongyi(
    model="qwen-plus",
    dashscope_api_key=DASHSCOPE_API_KEY,
    temperature=0.7
)

# 定义格式化文档的函数
def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

# 提示模板
system_template = """

    基于以下上下文信息回答问题。如果你不知道答案，就说不知道，不要编造答案。

    上下文信息：
    {context}

"""

human_template = "{question}"

prompt = ChatPromptTemplate.from_messages([
    ("system", system_template),
    ("human", human_template)
])

# 定义问答函数（直接使用 vector_store，更简单直观）
def ask_question(question, k=3):
    # 直接从向量存储检索相关文档
    retrieved_docs = vector_store.similarity_search(question, k=k)
    
    # 格式化文档为上下文
    context = format_docs(retrieved_docs)
    
    # 构建提示并调用 LLM
    messages = prompt.format_messages(context=context, question=question)
    answer = chatLLM.invoke(messages)
    
    return answer.content, retrieved_docs

# 进行问答
questions = [
    "什么是向量数据库？",
    "Python 主要用于什么领域？",
    "LangChain 是什么？"
]

for question in questions:
    print(f"\n问题：{question}")
    print("-" * 50)
    
    # 获取回答和参考文档
    answer, retrieved_docs = ask_question(question, k=3)
    
    print(f"回答：{answer}")
    print(f"\n参考文档：")
    for i, doc in enumerate(retrieved_docs, 1):
        print(f"  {i}. {doc.page_content[:50]}...")
    print()


# from openai import OpenAI

# client = OpenAI(
#     api_key = os.getenv("DASHSCOPE_API_KEY"),  # 如果您没有配置环境变量，请在此处用您的API Key进行替换
#     base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"  # 百炼服务的base_url
# )

# completion = client.embeddings.create(
#     model="text-embedding-v4",
#     input='衣服的质量杠杠的，很漂亮，不枉我等了这么久啊，喜欢，以后还来这里买',
#     dimensions=64, # 指定向量维度（仅 text-embedding-v3及 text-embedding-v4支持该参数）
#     encoding_format="float"
# )

# print(completion.model_dump_json())