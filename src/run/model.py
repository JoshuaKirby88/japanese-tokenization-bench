from dataclasses import dataclass

from src.dataset.model import DatasetName
from src.task.model import TaskResult
from src.tokenizer import TokenizationStrategy


@dataclass
class StrategySummary:
    avg_score: float
    total_dollars: float
    delta: float | None = None


ResultSummary = dict[TokenizationStrategy, StrategySummary]


@dataclass
class DatasetResult:
    dollars: float
    summary: ResultSummary
    strategy_results: list[dict[TokenizationStrategy, TaskResult]]


@dataclass
class ModelResut:
    dollars: float
    summary: ResultSummary
    dataset_results: dict[DatasetName, DatasetResult]


@dataclass
class BatchResult:
    models: list[str]
    datasets: list[DatasetName]
    strategies: list[TokenizationStrategy]
    dollars: float
    summary: ResultSummary
    model_results: dict[str, ModelResut]
