# 使用LLM API开发应用

## 一、基本概念

### Prompt 的理解
- **概念**：Prompt 最初设计为NLP下游任务的输入模板，现已成为给大模型所有输入的统称。
- **例子**：在ChatGPT中提出的问题是一个Prompt，ChatGPT的回复是Completion。
- **理解**：Prompt是向大模型提出的指令或问题，Completion是大模型的答案或反馈。

### Temperature 参数的作用
- **概念**：Temperature用于控制LLM生成结果的随机性与创造性。
- **取值**：范围为0~1，接近0产生保守文本，接近1时产生创意多样的文本。

### System Prompt 的定义和应用
- **概念**：System Prompt旨在提升用户体验，对模型的回复有持久影响。
- **区分**：System Prompt在会话中比User Prompt具有更高的重要性。
- **例子**：通过System Prompt设定模型人设，然后通过User Prompt提问。
    ```json
    {
        "system prompt": "你是一个幽默风趣的个人知识库助手，可以根据给定的知识库内容回答用户的提问，注意，你的回答风格应是幽默风趣的",
        "user prompt": "我今天有什么事务？"
    }
    ```
- **理解**：System Prompt在会话中设定一个“基调”或“角色”，引导模型响应。

## 二、使用LLM API

### 调用示例
#### 1. 读取 API Key
```python
import os
from dotenv import load_dotenv, find_dotenv

# 读取本地/项目的环境变量。
# find_dotenv() 寻找并定位 .env 文件的路径
# load_dotenv() 读取该 .env 文件，并将其中的环境变量加载到当前的运行环境中  
_ = load_dotenv(find_dotenv())
```
#### 2. 不同 API 调用方法

- **ChatGPT API**：通过设置model、messages、temperature等参数调用
  ```python
  from openai import OpenAI

  # 打印API密钥，用于调试
  # print("使用的API密钥：", api_key)

  client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=os.environ.get("OPENAI_BASE_URL")
  )


  def gen_gpt_messages(prompt):
    '''
    构造 GPT 模型请求参数 messages
    
    请求参数：
        prompt: 对应的用户提示词
    '''
    messages = [{"role": "user", "content": prompt}]
    return messages


  def get_completion(prompt, model="gpt-3.5-turbo", temperature = 0):
    '''
    获取 GPT 模型调用结果

    请求参数：
        prompt: 对应的提示词
        model: 调用的模型，默认为 gpt-3.5-turbo，也可以按需选择 gpt-4 等其他模型
        temperature: 模型输出的温度系数，控制输出的随机程度，取值范围是 0~2。温度系数越低，输出内容越一致。
    '''
    response = client.chat.completions.create(
        model=model,
        messages=gen_gpt_messages(prompt),
        temperature=temperature,
    )
    if len(response.choices) > 0:
        print(response.choices[0].message.content)
        return response.choices[0].message.content
    return "generate answer error"

  get_completion("你好")
  ```
- **文心一言API**：模型人设通过system字段传入
  ```python
  import qianfan  # 导入qianfan库，用于调用文心API

  def gen_wenxin_messages(prompt):
      '''
      构造文心模型请求参数 messages
      
      请求参数：
          prompt: 用户输入的提示词
      返回值：
          返回格式化的messages列表，适用于文心API请求
      '''

      messages = [{"role": "user", "content": prompt}]
      return messages  # 返回构造好的消息列表

  def get_completion(prompt, model="ERNIE-Bot", temperature=0.01):
      '''
      调用文心模型，获取生成的文本
      
      请求参数：
          prompt: 用户输入的提示词
          model: 使用的文心模型，默认为ERNIE-Bot
          temperature: 生成文本的温度系数，影响文本的多样性，默认值为0.01
      返回值：
          返回模型生成的文本结果
      '''
      
      chat_comp = qianfan.ChatCompletion()
      # 使用gen_wenxin_messages函数处理用户输入的提示词
      message = gen_wenxin_messages(prompt)
      
      resp = chat_comp.do(messages=message, 
                          model=model,
                          temperature=temperature,
                          system="你是一名个人助理-小鲸鱼")
      
      return resp["result"]  # 从响应中提取并返回结果部分

  get_completion("你好，介绍一下你自己")
  ```

- **讯飞星火API**：提供SDK方式和WebSocket方式调用
  1.通过 WebSocket 调用 (自动化下载调用示例并修改)
  ```python
  import requests
  from io import BytesIO
  from zipfile import ZipFile

  # 下载zip文件
  zip_url = 'https://xfyun-doc.xfyun.cn/lc-sp-sparkAPI-1709535448185.zip'
  response = requests.get(zip_url)
  zip_file = ZipFile(BytesIO(response.content))

  # 解压zip文件
  zip_file.extractall()

  # 读取并修改sparkAPI.py文件
  with open('sparkAPI.py', 'r', encoding='utf-8') as file:
      lines = file.readlines()

  with open('sparkAPI.py', 'w', encoding='utf-8') as file:
      for line in lines:
          if 'import openpyxl' in line:
              # 注释掉import openpyxl行
              file.write('# ' + line)
          elif 'def on_close(ws):' in line:
              # 修改on_close函数定义
              file.write('def on_close(ws, close_status_code, close_msg):\n')
          else:
              file.write(line)

  print('Done!')
  ```
  2.通过 SDK 方式调用
  ```python
  from sparkai.llm.llm import ChatSparkLLM, ChunkPrintHandler
  from sparkai.core.messages import ChatMessage
  from sympy import false

  def gen_spark_params(model):
      '''
      构造星火模型请求参数
      '''

      spark_url_tpl = "wss://spark-api.xf-yun.com/{}/chat"
      model_params_dict = {
          # v1.5 版本
          "v1.5": {
              "domain": "general", # 用于配置大模型版本
              "spark_url": spark_url_tpl.format("v1.1") # 云端环境的服务地址
          },
          # v2.0 版本
          "v2.0": {
              "domain": "generalv2", # 用于配置大模型版本
              "spark_url": spark_url_tpl.format("v2.1") # 云端环境的服务地址
          },
          # v3.0 版本
          "v3.0": {
              "domain": "generalv3", # 用于配置大模型版本
              "spark_url": spark_url_tpl.format("v3.1") # 云端环境的服务地址
          },
          # v3.5 版本
          "v3.5": {
              "domain": "generalv3.5", # 用于配置大模型版本
              "spark_url": spark_url_tpl.format("v3.5") # 云端环境的服务地址
          }
      }
      return model_params_dict[model]

  def gen_spark_messages(prompt):
      '''
      构造星火模型请求参数 messages

      请求参数：
          prompt: 对应的用户提示词
      '''

      messages = [ChatMessage(role="user", content=prompt)]
      return messages


  def get_completion(prompt, model="v3.5", temperature = 0.1):
      '''
      获取星火模型调用结果

      请求参数：
          prompt: 对应的提示词
          model: 调用的模型，默认为 v3.5，也可以按需选择 v3.0 等其他模型
          temperature: 模型输出的温度系数，控制输出的随机程度，取值范围是 0~1.0，且不能设置为 0。温度系数越低，输出内容越一致。
      '''

      spark_llm = ChatSparkLLM(
          spark_api_url=gen_spark_params(model)["spark_url"],
          spark_app_id=os.environ["SPARK_APPID"],
          spark_api_key=os.environ["SPARK_API_KEY"],
          spark_api_secret=os.environ["SPARK_API_SECRET"],
          spark_llm_domain=gen_spark_params(model)["domain"],
          temperature=temperature,
          streaming=False,
      )
      messages = gen_spark_messages(prompt)
      handler = ChunkPrintHandler()
      # 当 streaming设置为 False的时候, callbacks 并不起作用
      resp = spark_llm.generate([messages], callbacks=[handler])
      return resp

  get_completion("你好").generations[0][0].text
  ```

- **智谱GLM API**：提供SDK和原生HTTP调用方式
  ```python
  from zhipuai import ZhipuAI

  client = ZhipuAI(
      api_key=os.environ["ZHIPUAI_API_KEY"]
  )

  def gen_glm_params(prompt):
      '''
      构造 GLM 模型请求参数 messages

      请求参数：
          prompt: 对应的用户提示词
      '''
      messages = [{"role": "user", "content": prompt}]
      return messages


  def get_completion(prompt, model="glm-4", temperature=0.95):
      '''
      获取 GLM 模型调用结果

      请求参数：
          prompt: 对应的提示词
          model: 调用的模型，默认为 glm-4，也可以按需选择 glm-3-turbo 等其他模型
          temperature: 模型输出的温度系数，控制输出的随机程度，取值范围是 0~1.0，且不能设置为 0。温度系数越低，输出内容越一致。
      '''

      messages = gen_glm_params(prompt)
      response = client.chat.completions.create(
          model=model,
          messages=messages,
          temperature=temperature
      )
      if len(response.choices) > 0:
          return response.choices[0].message.content
      return "generate answer error"
  
  get_completion("你好")
  ```

## 三、Prompt Engineering

### Prompt Engineering 的意义
- **定义**: Prompt（提示）是用户与大模型交互输入的名称，即用户给大模型的输入。
- **重要性**: 对于大语言模型（LLM），一个好的Prompt设计极大地决定了模型的性能极限。通过有效的Prompt使用，可以充分发挥LLM的能力。
- **设计原则**: 设计高效Prompt的关键在于编写清晰、具体的指令，并给予模型充足的思考时间。

### Prompt 设计的原则及使用技巧
#### 1. 编写清晰、具体的指令
- **清晰明确的需求**: Prompt需要清楚表达需求，提供充足的上下文，以便模型准确理解意图。
- **使用分隔符**: 利用不同的标点符号作为分隔符，明确区分指令、上下文、输入等，避免混淆。
- **结构化输出**: 当需要结构化输出（如JSON格式）时，明确指定格式。
- **条件检查**: 告诉模型先检查假设条件，不满足则指出并停止执行。
- **少量示例**: 通过提供参考样例（少样本提示），帮助模型了解期望的输出样式。

#### 2. 给模型时间去思考
- **推理时间**: 给予语言模型充足的时间进行深入思考，通过Prompt引导模型进行逐步推理。
- **指定完成任务所需的步骤**: 明确给出完成任务的步骤，引导模型逐步解决问题。
- **指导模型自主思考**: 在下结论前，先指导模型找出自己的解法，然后再进行对比评估。

### 语言模型可能产生的问题
- **幻觉（Hallucination）**: 语言模型可能构造出似是而非的细节，称为“幻觉”，这是其一大缺陷。
- **应对策略**: 认识到幻觉问题的存在，并采取优化Prompt、引入外部知识等措施来缓解，以开发出更可靠的应用。

## 问题&思考❓
- Prompt的设计对于激发LLM的潜力非常关键，那么有没有什么通用的原则或指南来指导如何设计有效的Prompt？
  - [OpenAi 官方指南](https://platform.openai.com/docs/guides/prompt-engineering)
- 在实际应用中，如何根据项目的需要来决定Temperature的值？是否有一些经验规则或者是要通过大量的尝试和错误来确定？
  - [大语言模型(LLM)中的温度(Temperature)和Top_P怎么调](https://zhuanlan.zhihu.com/p/666315413)
- 在不同的应用场景中，应该如何选择最合适的大模型
  - 成本、性能、易用性、生态、支持、安全、隐私
