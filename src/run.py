from src.dataset.index import DatasetLoader
from src.task.index import TaskRunner


class Runner:
    dataset_loader = DatasetLoader()
    task_runner = TaskRunner()

    def run(self, n: int):
        for i, task in enumerate(self.dataset_loader.load_tasks("jcommonsenseqa")):
            if i == n:
                break

            results = self.task_runner.run(task)
            print(results)


if __name__ == "__main__":
    runner = Runner()
    runner.run(1)
