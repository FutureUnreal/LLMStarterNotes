# 一、LLM接入LangChain
[LangChain官方文档](https://python.langchain.com/docs/get_started/introduction)

## 实际操作
1. **调用 LLM**:
   - 通过 LangChain，可以方便地调用 LLM 并集成到个人应用中。

2. **Prompt 模板**:
   - 在开发大模型应用时，通常不会直接将用户输入传递给LLM。而是通过提示模板添加更多上下文，这些模板转化用户输入为完全格式化的提示，辅助模型更好地理解和响应。

3. **输出解析器**:
   - OutputParsers 负责将语言模型的输出转换为可用格式，比如 JSON 或字符串，这对于整合模型输出到应用中非常关键。

> ### 思考与疑问
> - **自定义 LLM 的性能如何优化？**
>   - 在自定义 LLM 接入时，是否存在性能损失？如果有，如何通过优化减少这种损失？
  
# 二、构建检索问答链
![](../resources/imgs/C4-RAG.png)

# 三、部署知识库助手
[streamlit官方文档](https://docs.streamlit.io/)
修改UI，部署成功！
![](../resources/imgs/C4-sreamlit.png)