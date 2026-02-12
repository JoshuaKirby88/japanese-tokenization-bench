from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, Literal, TypedDict, TypeVar

from src.task.model import Task

DatasetName = Literal["jcommonsenseqa", "jnli", "jsquad"]
DATASET_NAMES: list[DatasetName] = ["jcommonsenseqa", "jnli", "jsquad"]

T = TypeVar("T")


@dataclass
class DatasetConfig(Generic[T]):
    path: str
    name: str
    transform: Callable[[T], Task]


class JCommonsenseQA(TypedDict):
    q_id: int
    question: str
    choice0: str
    choice1: str
    choice2: str
    choice3: str
    choice4: str
    label: int


class JNLI(TypedDict):
    id: str
    sentence1: str
    sentence2: str
    label: int


class JSQuADAnswer(TypedDict):
    text: list[str]
    answer_start: list[int]


class JSQuADT(TypedDict):
    id: str
    title: str
    context: str
    question: str
    answers: JSQuADAnswer
