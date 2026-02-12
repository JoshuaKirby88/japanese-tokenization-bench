from dataclasses import dataclass

from src.dataset.model import DatasetName
from src.task.model import TaskResult


@dataclass
class RunResult:
    dataset: DatasetName
    model: str
    n: int
    dollars: float
    results: list[TaskResult]
