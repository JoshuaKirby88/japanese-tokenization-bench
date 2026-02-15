from dataclasses import dataclass
from typing import Literal, override

from src.dataset.model import DatasetName
from src.task.model import TaskResult
from src.tokenizer import TokenizationStrategy

Reasoning = Literal[None, "none", "low", "medium", "high"]
REASONINGS: list[Reasoning] = [None, "none", "low", "medium", "high"]


@dataclass()
class ModelConfig:
    model: str
    reasoning: Reasoning = None

    @override
    def __str__(self) -> str:
        parts: list[str] = [self.model]
        if self.reasoning:
            parts.append(self.reasoning)
        return ":".join(parts)


@dataclass
class StrategySummary:
    avg_score: float
    total_dollars: float
    delta: float | None = None


ResultSummary = dict[TokenizationStrategy, StrategySummary]


@dataclass
class LengthMultiplierResult:
    dollars: float
    summary: ResultSummary
    strategy_results: list[dict[TokenizationStrategy, TaskResult]]


@dataclass
class DatasetResult:
    dollars: float
    summary: ResultSummary
    length_multiplier_results: dict[int, LengthMultiplierResult]


@dataclass
class ModelResult:
    dollars: float
    summary: ResultSummary
    dataset_results: dict[DatasetName, DatasetResult]


@dataclass
class BatchResult:
    model_config: list[ModelConfig]
    datasets: list[DatasetName]
    strategies: list[TokenizationStrategy]
    dollars: float
    n: int
    length_multipliers: list[int]
    seed: int
    summary: ResultSummary
    model_results: dict[str, ModelResult]
