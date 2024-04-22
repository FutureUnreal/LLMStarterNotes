# 一、词向量及向量知识库简介

![vector](../resources/imgs/C3-vector.png)

# 二、使用 Embedding API

## OpenAI API
- OpenAI提供了三种性能不同的embedding model，包括`text-embedding-3-large`、`text-embedding-3-small`、和`text-embedding-ada-002`。其中，`text-embedding-3-large`性能最优但成本最高，适用于对性能有高要求且预算充足的场景
- [官方文档](https://platform.openai.com/docs/guides/embeddings)

```python
import os
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

# 如果你需要通过代理端口访问，你需要如下配置
# os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
# os.environ["HTTP_PROXY"] = 'http://127.0.0.1:7890'

def openai_embedding(text: str, model: str=None):
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_BASE_URL")
    )

    # embedding model：'text-embedding-3-small', 'text-embedding-3-large', 'text-embedding-ada-002'
    if model == None:
        model="text-embedding-ada-002"

    response = client.embeddings.create(
        input=text,
        model=model
    )
    print(response)
    return response

response = openai_embedding(text='要生成 embedding 的输入文本，字符串形式。')

print(f'返回的embedding类型为：{response.object}')
print(f'embedding长度为：{len(response.data[0].embedding)}')
print(f'embedding（前10）为：{response.data[0].embedding[:10]}')
print(f'本次embedding model为：{response.model}')
print(f'本次token使用情况为：{response.usage}')
```
API返回的数据为`json`格式，除`object`向量类型外还有存放数据的`data`、embedding model 型号`model`以及本次 token 使用情况`usage`等数据，具体如下所示：

```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "index": 0,
      "embedding": [
        -0.006929283495992422,
        -0.005336422007530928,
        ... (omitted for spacing)
        -4.547132266452536e-05,
        -0.024047505110502243
      ],
    }
  ],
  "model": "text-embedding-3-small",
  "usage": {
    "prompt_tokens": 5,
    "total_tokens": 5
  }
}
```

## 文心千帆API
- 文心千帆API提供了Embedding-V1模型，基于百度文心大模型技术。在使用前需要通过API Key和Secret Key获取Access token，然后通过token调用接口生成embedding
- [官方文档](https://cloud.baidu.com/doc/WENXINWORKSHOP/s/alj562vvu)

```python
import requests
import json

def wenxin_embedding(text: str):
    api_key = os.environ['QIANFAN_AK']
    secret_key = os.environ['QIANFAN_SK']

    # 使用API Key、Secret Key向https://aip.baidubce.com/oauth/2.0/token 获取Access token
    url = "https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id={0}&client_secret={1}".format(api_key, secret_key)
    payload = json.dumps("")
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    
    # 通过获取的Access token 来embedding text
    url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/embeddings/embedding-v1?access_token=" + str(response.json().get("access_token"))
    input = []
    input.append(text)
    payload = json.dumps({
        "input": input
    })
    headers = {
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    return json.loads(response.text)
# text应为List(string)
text = "要生成 embedding 的输入文本，字符串形式。"
response = wenxin_embedding(text=text)

print('本次embedding id为：{}'.format(response['id']))
print('本次embedding产生时间戳为：{}'.format(response['created']))
print('返回的embedding类型为:{}'.format(response['object']))
print('embedding长度为：{}'.format(len(response['data'][0]['embedding'])))
print('embedding（前10）为：{}'.format(response['data'][0]['embedding'][:10]))
```

API以JSON格式返回包含嵌入向量列表、创建时间戳和token使用情况的数据
```json

HTTP/1.1 200 OK
Date: Thu, 23 Mar 2023 03:12:03 GMT
Content-Type: application/json;charset=utf-8
Statement: AI-generated

{
  "id": "as-gjs275mj6s",
  "object": "embedding_list",
  "created": 1687155816,
  "data": [
    {
      "object": "embedding",
      "embedding": [
        0.018314670771360397,
        0.00942440889775753,
        ...（共384个float64）
        -0.36294862627983093
      ],
      "index": 0
    },
    {
      "object": "embedding",
      "embedding": [
        0.12250778824090958,
        0.07934671640396118,
        ...（共384个float64）
        0
      ],
      "index": 1
    }
  ],
  "usage": {
    "prompt_tokens": 12,
    "total_tokens": 12
  }
}
```
## 讯飞星火API
- 测试阶段，需向客服申请测试资格
- [官方文档](https://www.xfyun.cn/doc/spark/Embedding_api.html)

自动下载官方示例并修改
```python
import os
import re
import requests
import zipfile
from io import BytesIO
from dotenv import load_dotenv, find_dotenv

# 下载压缩包
url = "https://openres.xfyun.cn/xfyundoc/2024-03-26/78dc60db-b67d-4fb7-97a9-710fa5e226b0/1711443170449/Embedding.zip"
response = requests.get(url)
zip_file = zipfile.ZipFile(BytesIO(response.content))

# 解压压缩包到当前目录
zip_file.extractall()

# 重命名解压后的文件
original_file_name = "Embedding.py"
new_file_name = "Spark_Embedding.py"
os.rename(original_file_name, new_file_name)

# 读取文件内容
with open(new_file_name, "r") as file:
    content = file.read()

# 使用正则表达式确保能匹配到对应的行，并进行替换
content = re.sub(r"APPID ='.*?'", "APPID = os.environ.get('SPARK_APPID')", content)
content = re.sub(r"APISecret = '.*?'", "APISecret = os.environ.get('SPARK_API_SECRET')", content)
content = re.sub(r"APIKEY = '.*?'", "APIKEY = os.environ.get('SPARK_API_KEY')", content)

# 确保在文件顶部添加所需的import语句
if "import os" not in content:
    content = "import os\nfrom dotenv import load_dotenv, find_dotenv\n" + content

# 确保在main部分添加环境变量加载的代码
if "if __name__ == '__main__':" in content and "_ = load_dotenv(find_dotenv())" not in content:
    content = content.replace("if __name__ == '__main__':", "if __name__ == '__main__':\n    _ = load_dotenv(find_dotenv())")

# 将修改后的内容写回文件
with open(new_file_name, "w") as file:
    file.write(content)

print("文件处理完成!")
```
返回示例，`feature`输出数据
```json
{
    "header": {
        "code": 0,
        "message": "success",
        "sid": "ase000704fa@dx16ade44e4d87a1c802"
    },
    "payload": {
        "feature": {
            "encoding": "utf8",
            "compress": "raw",
            "format": "plain",
            "text": ""
        }
    }
}
```

## 智谱API
- 智谱API通过封装好的SDK提供服务，支持直接调用`embeddings.create`方法生成文本的向量表示
- [官方文档](https://open.bigmodel.cn/dev/api#vector)
```python
from zhipuai import ZhipuAI
def zhipu_embedding(text: str):

    api_key = os.environ['ZHIPUAI_API_KEY']
    client = ZhipuAI(api_key=api_key)
    response = client.embeddings.create(
        model="embedding-2",
        input=text,
    )
    return response

text = '要生成 embedding 的输入文本，字符串形式。'
response = zhipu_embedding(text=text)

print(f'response类型为：{type(response)}')
print(f'embedding类型为：{response.object}')
print(f'生成embedding的model为：{response.model}')
print(f'生成的embedding长度为：{len(response.data[0].embedding)}')
print(f'embedding（前10）为: {response.data[0].embedding[:10]}')
```

# 三、数据处理

数据处理是构建本地知识库的一个关键步骤。以下是详细的过程和代码，用于处理多种类型的文档，并将其内容转化为词向量，进而构建向量数据库。

## 1. 源文档选取
以[《机器学习公式详解》(PDF) ](https://github.com/datawhalechina/pumpkin-book/releases)为例

## 2. 数据读取
### PDF 文档读取
使用`LangChain`的`PyMuPDFLoader`来读取PDF文件，这是速度最快的PDF解析器之一。
```python
from langchain.document_loaders.pdf import PyMuPDFLoader

# 实例化PyMuPDFLoader，指定pdf文档路径
loader = PyMuPDFLoader("../../data_base/knowledge_db/pumkin_book/pumpkin_book.pdf")

# 调用load函数进行文档加载
pdf_pages = loader.load()
```

- 加载后的文档存储在pdf_pages变量中，每个页面作为一个Document对象存储。
- 可以打印出pdf_pages的长度，了解PDF包含的页面数量。
### MD 文档读取
Markdown文档的读取方法与PDF类似，使用了UnstructuredMarkdownLoader。

```python
from langchain.document_loaders.markdown import UnstructuredMarkdownLoader

loader = UnstructuredMarkdownLoader("../../data_base/xxx.md")
md_pages = loader.load()
```

- md_pages变量同样包含了Markdown文档的页面，每页作为一个Document对象。
### 数据清洗
目标是使知识库数据尽量有序、优质、精简。通过正则表达式和替换方法去除不必要的换行符和空格。

```python

import re

# 使用正则表达式匹配并删除不需要的换行符
pattern = re.compile(r'[^\u4e00-\u9fff](\n)[^\u4e00-\u9fff]', re.DOTALL)
pdf_page.page_content = re.sub(pattern, lambda match: match.group(0).replace('\n', ''), pdf_page.page_content)

# 进一步去除不必要的字符和空格
pdf_page.page_content = pdf_page.page_content.replace('•', '').replace(' ', '')
```

### 文档分割
为避免单个文档的长度超过模型支持的上下文限制，需要将文档分割为若干chunk。

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 设定单个文本块的大小和重合长度
CHUNK_SIZE = 500
OVERLAP_SIZE = 50

text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=OVERLAP_SIZE)

# 对内容进行分割
split_chunks = text_splitter.split_text(pdf_page.page_content[0:1000])
```
- 分割后的每个chunk将转化为词向量并存储到向量数据库中。
- 在检索时，chunk作为检索的基本单位，可以自由设定检索到的chunk数量。
>注：文档分割是数据处理中的核心步骤，决定了检索系统的效率。根据不同的业务和数据类型，可能需要设定个性化的文档分割方式。

# 四、搭建并使用向量数据库

## 1. 前序配置

在搭建向量数据库之前，首先需要进行一些前序配置，包括读取环境变量以及获取文件路径等。

```python
import os
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

# 如果需要通过代理访问，可以进行如下配置
# os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
# os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'

# 获取所有文件路径
file_paths = []
folder_path = '../../data_base/knowledge_db'
for root, dirs, files in os.walk(folder_path):
    for file in files:
        file_path = os.path.join(root, file)
        file_paths.append(file_path)
```

## 2. 构建Chroma向量库
使用Langchain中的Chroma作为向量存储库，其特点是轻量级且数据存储在内存中，便于快速开始使用。可以直接使用OpenAI和百度千帆的Embedding，也支持自定义Embedding API。

```python
from langchain.vectorstores.chroma import Chroma
from zhipuai_embedding import ZhipuAIEmbeddings  # 假设这是自定义的智谱Embedding

# 定义Embeddings
embedding = ZhipuAIEmbeddings()

# 定义持久化路径
persist_directory = '../../data_base/vector_db/chroma'

# 删除旧的数据库文件
!rm -rf '../../data_base/vector_db/chroma'

# 从文档构建向量数据库
vectordb = Chroma.from_documents(
    documents=split_docs[:20],
    embedding=embedding,
    persist_directory=persist_directory
)

# 持久化向量数据库
vectordb.persist()
```

## 3. 向量检索
向量检索分为相似度检索和最大边际相关性(MMR)检索。

### 相似度检索
相似度检索基于余弦距离，用于找到与查询最相似的文档。

```python
question = "什么是大语言模型"
sim_docs = vectordb.similarity_search(question, k=3)

for i, sim_doc in enumerate(sim_docs):
    print(f"检索到的第{i}个内容:\n{sim_doc.page_content[:200]}")
```

### MMR检索
MMR检索在保持相关性的同时，增加内容的多样性，避免结果过于单一。

```python
mmr_docs = vectordb.max_marginal_relevance_search(question, k=3)

for i, sim_doc in enumerate(mmr_docs):
    print(f"MMR检索到的第{i}个内容:\n{sim_doc.page_content[:200]}")
```

---


> # 问题&思考❓
> - 词向量如何处理多义词的情况，即一个词在不同语境下有不同含义时，它的向量表示如何变化？
>   - 一些模型如ELMo、BERT通过为每个词的每次出现提供独立的向量表示，更好地捕获了词在不同上下文中的含义
> - 在数据处理阶段，如何评估处理质量、选择最佳实践，并可能地实现自动化以提升效率和准确性？
> - 示例中提到了自定义`Embedding`的集成，这种自定义集成是否存在性能损失？如何优化这一过程以保证检索效率和准确性？