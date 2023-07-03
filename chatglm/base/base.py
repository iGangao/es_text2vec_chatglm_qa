from abc import ABC, abstractmethod
from typing import Optional, List
from chatglm.loader import LoaderCheckPoint


class AnswerResult:
    """
    消息实体
    """
    history: List[List[str]] = []
    llm_output: Optional[dict] = None


class BaseAnswer(ABC):      # Abstract Base Classes（抽象基类）
    """上层业务包装器.用于结果生成统一api调用"""

    @property           # @property装饰器用于将一个方法转换为类的属性
    @abstractmethod     # @abstractmethod装饰器用于定义抽象方法。抽象方法是在基类中声明但没有实现的方法，子类必须实现这些方法。
    def _check_point(self) -> LoaderCheckPoint:
        """Return _check_point of llm."""

    @property
    @abstractmethod
    def _history_len(self) -> int:
        """Return _history_len of llm."""

    @abstractmethod
    def set_history_len(self, history_len: int) -> None:
        """Return _history_len of llm."""

    def generatorAnswer(self, prompt: str,
                        history: List[List[str]] = [],
                        streaming: bool = False):
        pass
