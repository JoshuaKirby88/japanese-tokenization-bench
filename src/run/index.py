import datetime
import json
from dataclasses import asdict
from pathlib import Path

from src.dataset.index import DatasetLoader
from src.dataset.model import DatasetName
from src.run.model import RunResult
from src.task.index import TaskRunner
from src.task.model import TaskResult

RESULT_DIR = Path("data/results")


class Runner:
    dataset_loader = DatasetLoader()
    task_runner = TaskRunner()

    def run(self, dataset_name: DatasetName, model: str, n: int):
        task_results: list[TaskResult] = []

        for i, task in enumerate(self.dataset_loader.load_tasks(dataset_name)):
            if i == n:
                break

            results = self.task_runner.run(model=model, task=task)
            task_results.extend(results)

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


if __name__ == "__main__":
    runner = Runner()
    runner.run(dataset_name="jcommonsenseqa", model="mistralai/ministral-3b-2512", n=1)
