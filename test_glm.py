
from chatglm.loader import LoaderCheckPoint
from chatglm.base import BaseAnswer
import chatglm.shared as shared
shared.loaderCheckPoint = LoaderCheckPoint()
llm_model_ins = shared.loaderLLM() # 实例化ChatGLM对象 in chatglm_llm.py
llm_model_ins.set_history_len(3) 
llm: BaseAnswer = llm_model_ins # 继承 ChatGLM
generator = llm_model_ins.generatorAnswer("你好")
for answer_result in generator:
    print(answer_result.llm_output)