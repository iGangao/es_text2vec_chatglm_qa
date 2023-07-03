# es_text2vec_chatglm_qa
## 基于chatglm实现的知识问答系统
[流程]
### 本地知识库经过es检索系统召回+text2vec向量相似度计算再召回，作为prompt输入给chatglm后生成回答


Embedding使用text2vec-large-chinese
https://huggingface.co/GanymedeNil/text2vec-large-chinese/tree/main
LLM使用chatglm-6b
https://github.com/THUDM/ChatGLM-6B
https://huggingface.co/THUDM/chatglm-6b

<h5>
  参考
  <br/>
  https://github.com/imClumsyPanda/langchain-ChatGLM
  <br/>
  https://github.com/shibing624/text2vec
</h5>
