import datetime
import json
import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import src.patch_sdk as _
from src.dataset.index import DatasetLoader
from src.dataset.model import DATASET_NAMES, DatasetName
from src.run.model import BatchResult, DatasetResult, ModelConfig, ModelResult, ResultSummary, StrategySummary
from src.task.index import TaskRunner
from src.task.model import TaskResult
from src.tokenizer import TOKENIZATION_STRATEGIES, TokenizationStrategy

RESULT_DIR = Path("data/results")


class Runner:
    task_runner = TaskRunner()

    def run(
        self, model_config: ModelConfig, dataset_name: DatasetName, strategies: list[TokenizationStrategy], n: int, length_multiplier: int, seed: int
    ):
        print(f"Running {dataset_name} with {model_config} for n={n}, seed={seed}...")
        dataset_loader = DatasetLoader(length_multiplier=length_multiplier)
        all_tasks = list(dataset_loader.load_tasks(dataset_name))
        random.Random(seed).shuffle(all_tasks)
        tasks = all_tasks[:n]

        with ThreadPoolExecutor(max_workers=5) as executor:
            strategy_to_result_list: list[dict[TokenizationStrategy, TaskResult]] = list(
                executor.map(
                    lambda t: self.task_runner.run(
                        model_config=model_config, strategies=strategies, task=t, distractor_candidates=all_tasks, length_multiplier=length_multiplier
                    ),
                    tasks,
                )
            )

        return DatasetResult(
            dollars=sum(r.dollars for s_to_r in strategy_to_result_list for r in s_to_r.values()),
            summary=self.calculate_summary(strategies, strategy_to_result_list),
            strategy_results=strategy_to_result_list,
        )

    def calculate_summary(self, strategies: list[TokenizationStrategy], strategy_to_result_list: list[dict[TokenizationStrategy, TaskResult]]):
        baseline_scores = [s_to_r["baseline"].evaluation for s_to_r in strategy_to_result_list]
        baseline_avg = sum(baseline_scores) / len(baseline_scores)

        summary: ResultSummary = {}

        for strategy in strategies:
            strategy_results = [r[strategy] for r in strategy_to_result_list]
            scores = [r.evaluation for r in strategy_results]
            dollars_list = [r.dollars for r in strategy_results]
            avg = sum(scores) / len(scores)

            summary[strategy] = StrategySummary(
                avg_score=avg, total_dollars=sum(dollars_list), delta=avg - baseline_avg if strategy != "baseline" else None
            )

        return summary

    def run_batch(
        self,
        model_configs: list[ModelConfig],
        dataset_names: list[DatasetName],
        strategies: list[TokenizationStrategy],
        n: int,
        length_multiplier: int,
        seed: int,
    ):
        model_results: dict[str, ModelResult] = {}

        for model_config in model_configs:
            dataset_results: dict[DatasetName, DatasetResult] = {}
            for dataset_name in dataset_names:
                try:
                    dataset_results[dataset_name] = self.run(
                        model_config=model_config,
                        dataset_name=dataset_name,
                        strategies=strategies,
                        n=n,
                        length_multiplier=length_multiplier,
                        seed=seed,
                    )
                except Exception as e:
                    print(f"Error running {dataset_name} with {model_config}: {e}")

            if dataset_results:
                model_results[str(model_config)] = ModelResult(
                    dollars=sum(r.dollars for r in dataset_results.values()),
                    summary=self.aggregate_summaries(strategies=strategies, summaries=[r.summary for r in dataset_results.values()]),
                    dataset_results=dataset_results,
                )

        if not model_results:
            return

        batch_result = BatchResult(
            model_config=model_configs,
            datasets=dataset_names,
            strategies=strategies,
            dollars=sum(m.dollars for m in model_results.values()),
            summary=self.aggregate_summaries(strategies=strategies, summaries=[m.summary for m in model_results.values()]),
            model_results=model_results,
            n=n,
            length_multiplier=length_multiplier,
            seed=seed,
        )

        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        result_path = RESULT_DIR / f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(asdict(batch_result), f, indent=4, ensure_ascii=False)
        print(f"Results saved to {result_path}")

    def aggregate_summaries(self, strategies: list[TokenizationStrategy], summaries: list[ResultSummary]):
        baseline_scores = [s["baseline"].avg_score for s in summaries]
        baseline_avg = sum(baseline_scores) / len(baseline_scores)

        root_summary: ResultSummary = {}
        for strategy in strategies:
            scores = [s[strategy].avg_score for s in summaries]
            dollars = [s[strategy].total_dollars for s in summaries]
            avg = sum(scores) / len(scores)

            root_summary[strategy] = StrategySummary(
                avg_score=avg, total_dollars=sum(dollars), delta=avg - baseline_avg if strategy != "baseline" else None
            )
        return root_summary


if __name__ == "__main__":
    runner = Runner()
    runner.run_batch(
        strategies=TOKENIZATION_STRATEGIES,
        model_configs=[ModelConfig(model="google/gemini-3-flash-preview:floor", reasoning="none")],
        dataset_names=DATASET_NAMES,
        n=5,
        length_multiplier=1,
        seed=0,
    )
