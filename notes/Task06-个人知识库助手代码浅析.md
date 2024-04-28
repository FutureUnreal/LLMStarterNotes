# 一、技术架构分析
[项目地址](https://github.com/logan-zou/Chat_with_Datawhale_langchain)

项目的技术架构分为五个主要层次：
- **LLM层**：包括对流行的大型语言模型API的调用封装，实现了模型的灵活切换。
- **数据层**：管理原始数据和经过预处理的嵌入数据。
- **数据库层**：构建基于知识库数据的向量数据库，使用如 Chroma 等技术存储向量化数据。
- **应用层**：通过 Langchain 提供的检索问答链基类，封装了核心的检索问答功能。
- **服务层**：实现了基于 Gradio 和 FastAPI 的用户界面和API服务，以支持系统的访问和交互。
![](../resources/imgs/C6-1-structure.jpg)

# 二、关键功能与实现浅析

## 知识库构建和数据处理
- **数据获取**：通过自动化脚本从 GitHub 仓库批量获取项目的文件，这些文件作为知识库的原始数据源。
- **数据预处理**：使用文本分割器和向量化工具处理原始数据，如使用 m3e 文本嵌入模型将数据转换为向量格式，便于后续的快速检索。

```python
# 示例代码：数据预处理和向量化
from langchain.embedding import HuggingFaceEmbeddings

# 初始化嵌入模型
embedder = HuggingFaceEmbeddings(model_name="moka-ai/m3e-base")

# 文本向量化函数
def vectorize_text(text):
    return embedder.embed(text)
```
## 问答系统实现
索引和检索：向量数据库在存储了文本向量后，使用相似度搜索技术快速检索出与查询最相关的文档片段。
生成答案：结合检索到的文档片段和用户的查询，构建适当的提示（prompt），并通过调用配置好的大型语言模型生成精确的回答。
```python
from langchain.chains import RetrievalQA

# 初始化问答系统
qa_system = RetrievalQA(llm_model="gpt-4", vector_db="path_to_vector_db")

# 执行查询并生成答案
def get_answer(query):
    result = qa_system.answer(query)
    return result
```

## UI实现
启动服务：服务通过在后端配置如 uvicorn 和 Gradio，支持以 Web 应用形式对外提供接口。
模型调用与管理：封装不同的大型语言模型（如 OpenAI 的 GPT、文心 ERNIE-Bot 等），为应用层提供统一的调用接口。
```python
import gradio as gr

def chat_with_model(question):
    return get_answer(question)

iface = gr.Interface(fn=chat_with_model, inputs="text", outputs="text")
iface.launch()
```
# 三、应用和示例
应用部署：通过简洁的 Gradio 界面和 FastAPI 构建的服务，用户可以方便地访问系统，进行实时的问答交互。
使用示例：
```
输入问题：“南瓜书是什么？”
系统通过检索和生成，输出详细的书籍介绍。
```