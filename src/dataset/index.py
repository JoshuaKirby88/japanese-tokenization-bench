import json
import os
from typing import Any, cast

from datasets.combine import concatenate_datasets
from datasets.dataset_dict import DatasetDict
from datasets.load import load_dataset
from datasets.utils.logging import set_verbosity_error
from dotenv import load_dotenv

from src.dataset.char_count import get_char_count_output_file, prepare_char_count
from src.dataset.jwtd import prepare_jwtd
from src.dataset.model import JNLI, CharCount, DatasetConfig, DatasetName, JCommonsenseQA, JSQuADT, WikipediaTypo
from src.task.model import Task

load_dotenv()
set_verbosity_error()

os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "1"


class DatasetLoader:
    def __init__(self, length_multiplier: int, seed: int):
        if length_multiplier < 1:
            raise ValueError("length_multiplier must be an integer >= 1.")
        self.configs: dict[DatasetName, DatasetConfig[Any]] = {
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
                prepare=None,
            ),
            "JNLI": DatasetConfig[JNLI](
                path="shunk031/JGLUE",
                name="JNLI",
                transform=lambda r: Task(
                    id=r["sentence_pair_id"], type="nli", context=r["sentence1"], question=r["sentence2"], options=[], ground_truths=[r["label"]]
                ),
                prepare=None,
            ),
            "JSQuAD": DatasetConfig[JSQuADT](
                path="shunk031/JGLUE",
                name="JSQuAD",
                transform=lambda r: Task(
                    id=r["id"], type="extraction", context=r["context"], question=r["question"], options=[], ground_truths=r["answers"]["text"]
                ),
                prepare=None,
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
            "CharCount": DatasetConfig[CharCount](
                path="json",
                name=str(get_char_count_output_file(length_multiplier)),
                prepare=lambda: prepare_char_count(length_multiplier, seed),
                transform=lambda r: Task(
                    id=r["id"], type="char_counting", context=r["text"], question=r["character"], options=[], ground_truths=[r["count"]]
                ),
            ),
        }

    def load_raw(self, dataset_name: DatasetName):
        config = self.configs[dataset_name]
        if config.prepare:
            config.prepare()

        if config.path == "json":
            with open(config.name, "r", encoding="utf-8") as f:
                return [json.loads(line) for line in f]

        dataset = cast(DatasetDict, load_dataset(config.path, config.name, trust_remote_code=True))
        return concatenate_datasets([dataset["train"], dataset["validation"]])

    def load_tasks(self, dataset: DatasetName):
        config = self.configs[dataset]
        for row in self.load_raw(dataset):
            yield config.transform(row)


if __name__ == "__main__":
    loader = DatasetLoader(length_multiplier=1, seed=0)

    print("JCommonsenseQA:")
    print(loader.load_raw("JCommonsenseQA")[0])

    print("\nJNLI:")
    print(loader.load_raw("JNLI")[0])

    print("\nJSQuAD:")
    print(loader.load_raw("JSQuAD")[0])

    print("\nWikipedia Typo:")
    print(loader.load_raw("JWTD")[0])

    print("\nCharCount:")
    print(loader.load_raw("CharCount")[0])
