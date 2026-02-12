import os
from typing import Any, cast

from datasets.combine import concatenate_datasets
from datasets.dataset_dict import DatasetDict
from datasets.load import load_dataset

from src.dataset.jwtd import prepare_jwtd
from src.dataset.model import (
    JNLI,
    DatasetConfig,
    DatasetName,
    JCommonsenseQA,
    JSQuADT,
    WikipediaTypo,
)
from src.task.model import Task

os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "1"


class DatasetLoader:
    configs: dict[DatasetName, DatasetConfig[Any]] = {
        "JCommonsenseQA": DatasetConfig[JCommonsenseQA](
            path="shunk031/JGLUE",
            name="JCommonsenseQA",
            transform=lambda r: Task(
                id=str(r["q_id"]),
                type="multiple_choice",
                context=None,
                question=r["question"],
                options=[r[f"choice{i}"] for i in range(5)],
                ground_truths=[r["label"]],
            ),
        ),
        "JNLI": DatasetConfig[JNLI](
            path="shunk031/JGLUE",
            name="JNLI",
            transform=lambda r: Task(
                id=r["id"],
                type="nli",
                context=r["sentence1"],
                question=r["sentence2"],
                options=[],
                ground_truths=[r["label"]],
            ),
        ),
        "JSQuAD": DatasetConfig[JSQuADT](
            path="shunk031/JGLUE",
            name="JSQuAD",
            transform=lambda r: Task(
                id=r["id"],
                type="extraction",
                context=r["context"],
                question=r["question"],
                options=[],
                ground_truths=r["answers"]["text"],
            ),
        ),
        "JWTD": DatasetConfig[WikipediaTypo](
            path="json",
            name="data/jwtd/test.jsonl",
            prepare=prepare_jwtd,
            transform=lambda r: Task(
                id=f"{r['page']}_{r['pre_rev']}_{r['post_rev']}",
                type="correction",
                context=None,
                question=r["pre_text"],
                options=[],
                ground_truths=[f"{d['pre']} -> {d['post']}" for d in r["diffs"]],
            ),
        ),
    }

    def load_raw(self, dataset_name: DatasetName):
        config = self.configs[dataset_name]
        if config.prepare:
            config.prepare()

        if config.path == "json":
            dataset = cast(DatasetDict, load_dataset("json", data_files=config.name))
            return dataset["train"]

        dataset = cast(
            DatasetDict, load_dataset(config.path, config.name, trust_remote_code=True)
        )
        return concatenate_datasets([dataset["train"], dataset["validation"]])

    def load_tasks(self, dataset: DatasetName):
        config = self.configs[dataset]
        for row in self.load_raw(dataset):
            yield config.transform(row)


if __name__ == "__main__":
    loader = DatasetLoader()

    print("JCommonsenseQA:")
    print(loader.load_raw("JCommonsenseQA")[0])

    print("\nJNLI:")
    print(loader.load_raw("JNLI")[0])

    print("\nJSQuAD:")
    print(loader.load_raw("JSQuAD")[0])

    print("\nWikipedia Typo:")
    print(loader.load_raw("JWTD")[0])
