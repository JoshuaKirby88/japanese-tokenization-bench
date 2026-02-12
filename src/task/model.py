from dataclasses import dataclass
from typing import Callable, Literal

from src.tokenizer import TokenizationStrategy

TaskType = Literal["multiple_choice", "nli", "extraction", "correction"]
TASK_TYPES: list[TaskType] = ["multiple_choice", "nli", "extraction", "correction"]

NIL_LABELS = ["Entailment", "Neutral", "Contradiction"]


@dataclass
class Task:
    id: str
    type: TaskType
    context: str | None
    question: str
    options: list[str]
    ground_truths: list[str] | list[int]


@dataclass
class TaskConfig:
    get_system_prompt: Callable[[Task, TokenizationStrategy], str]
    get_user_prompt: Callable[[Task, TokenizationStrategy], str]
    evaluate: Callable[[Task, TokenizationStrategy, str], bool]


@dataclass
class TaskResult:
    task_id: str
    task_type: TaskType
    tokenization_strategy: TokenizationStrategy
    user_prompt: str
    response: str
    dollars: float
    evaluation: bool
