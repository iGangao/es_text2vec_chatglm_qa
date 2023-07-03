# es_text2vec_chatglm_qa

## 基于chatglm实现的知识问答系统

### 本地知识库经过es检索系统召回+text2vec向量相似度计算再召回，作为prompt输入给chatglm后生成回答

#### 模型需下载到本地
- Embedding使用[text2vec-large-chinese](https://huggingface.co/GanymedeNil/text2vec-large-chinese/tree/main)
- LLM使用[chatglm-6b](https://huggingface.co/THUDM/chatglm-6b)

#### 如何运行
##### 1. 安装requirements.txt
`pip install -r requirements.txt`
##### 2. 运行ui.py
`python ui.py`
#### 效果展示
![image](https://github.com/iGangao/es_text2vec_chatglm_qa/assets/73676846/8b5198aa-f590-4e71-8f09-453b519969f8)
- 参数ES Top-k是根据检索召回的top-k
- 参数VEC top-k是对检索召回的数据进行向量化后与query计算相似度后进行第二次召回的top-k
<h5>
  参考
  <br/>
  https://github.com/imClumsyPanda/langchain-ChatGLM
  <br/>
  https://github.com/shibing624/text2vec
</h5>
