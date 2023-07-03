import sys
from typing import Any
from chatglm.loader import LoaderCheckPoint

loaderCheckPoint: LoaderCheckPoint = None
# 此处请写绝对路径
llm_model_dict = {
    "chatglm-6b": {
        "name": "chatglm-6b",
        "pretrained_model_name": "THUDM/chatglm-6b",
        "local_model_path": "/data/agl/models/chatglm-6b",
        "provides": "ChatGLM"
    },
}
LLM_MODEL = "chatglm-6b"
def loaderLLM() -> Any:
    pre_model_name = loaderCheckPoint.model_name
    llm_model_info = llm_model_dict[pre_model_name]

    loaderCheckPoint.model_name = llm_model_info['pretrained_model_name']

    loaderCheckPoint.model_path = llm_model_info["local_model_path"]

    loaderCheckPoint.reload_model()

    provides_class = getattr(sys.modules['chatglm'], llm_model_info['provides'])
    modelInsLLM = provides_class(checkPoint=loaderCheckPoint)
    return modelInsLLM
