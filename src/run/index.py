import datetime
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

from src.dataset.index import DatasetLoader
from src.dataset.model import DatasetName
from src.run.model import RunResult
from src.task.index import TaskRunner
from src.task.model import Task

RESULT_DIR = Path("data/results")


class Runner:
    dataset_loader = DatasetLoader()
    task_runner = TaskRunner()

    def run(self, dataset_name: DatasetName, model: str, n: int):
        tasks: list[Task] = []
        for i, task in enumerate(self.dataset_loader.load_tasks(dataset_name)):
            if i == n:
                break
            tasks.append(task)

        with ThreadPoolExecutor(max_workers=5) as executor:
            all_results = list(
                executor.map(lambda t: self.task_runner.run(model=model, task=t), tasks)
            )
        task_results = [r for rs in all_results for r in rs]

        run_result = RunResult(
            dataset=dataset_name,
            model=model,
            n=n,
            dollars=sum(r.dollars for r in task_results),
            results=task_results,
        )

        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        with open(
            RESULT_DIR / f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(asdict(run_result), f, indent=4, ensure_ascii=False)

        print(f"\nResults for {dataset_name} ({model}):")
        strategies = sorted(list({r.tokenization_strategy for r in task_results}))
        for strategy in strategies:
            scores = [
                r.evaluation
                for r in task_results
                if r.tokenization_strategy == strategy
            ]
            avg_score = sum(scores) / len(scores) if scores else 0
            print(f"  {strategy:10}: {avg_score:.4f}")


if __name__ == "__main__":
    runner = Runner()
    runner.run(dataset_name="JCommonsenseQA", model="mistralai/ministral-3b-2512", n=1)
