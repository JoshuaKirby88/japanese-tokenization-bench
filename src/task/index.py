from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from ai_sdk import generate_text, openai
from ai_sdk.generate_text import GenerateTextResult
from dotenv import load_dotenv

from src.task.model import (
    NIL_LABELS,
    Task,
    TaskConfig,
    TaskResult,
    TaskType,
)
from src.tokenizer import TokenizationStrategy, Tokenizer

load_dotenv()

tokenizer = Tokenizer()


class TaskRunner:
    configs: dict[TaskType, TaskConfig] = {
        "multiple_choice": TaskConfig(
            get_system_prompt=lambda task, strategy: (
                "Answer with a single choice label only. Do not use markdown or extra formatting."
            ),
            get_user_prompt=lambda task, strategy: (
                f"Question: {tokenizer.tokenize(task.question, strategy)}\n"
                + "\n"
                + "Choices:\n"
                + "\n".join(
                    tokenizer.tokenize(option, strategy) for option in task.options
                )
            ),
            evaluate=lambda task, strategy, response: (
                1.0
                if any(
                    tokenizer.normalize(response, strategy)
                    == tokenizer.normalize(option, strategy)
                    and task.options.index(option) in task.ground_truths
                    for option in task.options
                )
                else 0.0
            ),
        ),
        "nli": TaskConfig(
            get_system_prompt=lambda task, strategy: (
                "Answer with a single choice label only. Do not use markdown or extra formatting."
            ),
            get_user_prompt=lambda task, strategy: (
                f"Premise: {tokenizer.tokenize(task.context or '', strategy)}\n"
                + f"Hypothesis: {tokenizer.tokenize(task.question, strategy)}\n"
                + "\n"
                + "Choices:\n"
                + "\n".join(label for label in NIL_LABELS)
            ),
            evaluate=lambda task, strategy, response: (
                1.0
                if any(
                    tokenizer.normalize(response, strategy)
                    == tokenizer.normalize(label, strategy)
                    and NIL_LABELS.index(label) in task.ground_truths
                    for label in NIL_LABELS
                )
                else 0.0
            ),
        ),
        "extraction": TaskConfig(
            get_system_prompt=lambda task, strategy: (
                'Extract the answer from the "Context", and return only the answer. Do not use markdown or extra formatting.'
            ),
            get_user_prompt=lambda task, strategy: (
                f"Context: {tokenizer.tokenize(task.context or '', strategy)}\n"
                + f"Question: {tokenizer.tokenize(task.question, strategy)}"
            ),
            evaluate=lambda task, strategy, response: max(
                (
                    TaskRunner.compute_f1(
                        tokenizer.normalize(response, strategy),
                        tokenizer.normalize(str(gt), strategy),
                    )
                    for gt in task.ground_truths
                ),
                default=0.0,
            ),
        ),
        "correction": TaskConfig(
            get_system_prompt=lambda task, strategy: (
                'Identify and correct typos in "Text".\n'
                + 'Return corrections in the format: "Typo -> Correction".\n'
                + "If multiple exist, list them one per line.\n"
                + "Return only the corrections. Do not use markdown or extra formatting."
            ),
            get_user_prompt=lambda task, strategy: (
                f"Text: {tokenizer.tokenize(task.question, strategy)}"
            ),
            evaluate=lambda task, strategy, response: (
                sum(
                    1.0
                    if tokenizer.normalize(str(gt), strategy)
                    in tokenizer.normalize(response, strategy)
                    else 0.0
                    for gt in task.ground_truths
                )
                / len(task.ground_truths)
            ),
        ),
        "char_counting": TaskConfig(
            get_system_prompt=lambda task, strategy: (
                'Count the number of "Character" in "Text". Answer with a single number only. Do not use markdown or extra formatting.'
            ),
            get_user_prompt=lambda task, strategy: (
                f"Text: {tokenizer.tokenize(task.context or '', strategy)}\n"
                + f"Character: {task.question}"
            ),
            evaluate=lambda task, strategy, response: (
                1.0
                if any(str(gt) == response.strip() for gt in task.ground_truths)
                else 0.0
            ),
        ),
    }

    @staticmethod
    def compute_f1(prediction: str, ground_truth: str):
        prediction_tokens = list(prediction)
        ground_truth_tokens = list(ground_truth)
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0.0
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        return (2 * precision * recall) / (precision + recall)

    def get_cost_from_response(self, res: GenerateTextResult):
        dollars = 0.0
        if (
            res.raw_response
            and hasattr(res.raw_response, "usage")
            and res.raw_response.usage
        ):
            retrieved_cost = getattr(res.raw_response.usage, "cost", None)
            if retrieved_cost is not None:
                try:
                    dollars = float(retrieved_cost)
                except (TypeError, ValueError):
                    pass
        return dollars

    def run_strategy(self, model: str, strategy: TokenizationStrategy, task: Task):
        config = self.configs[task.type]
        user_prompt = config.get_user_prompt(task, strategy)
        res = generate_text(
            model=openai(model),
            system=config.get_system_prompt(task, strategy),
            prompt=user_prompt,
        )

        return TaskResult(
            task_id=task.id,
            task_type=task.type,
            tokenization_strategy=strategy,
            user_prompt=user_prompt,
            response=res.text,
            dollars=self.get_cost_from_response(res),
            evaluation=config.evaluate(task, strategy, res.text),
            ground_truths=task.ground_truths,
        )

    def run(self, model: str, strategies: list[TokenizationStrategy], task: Task):
        with ThreadPoolExecutor() as executor:
            task_results = list(
                executor.map(
                    lambda strategy: self.run_strategy(
                        model=model, strategy=strategy, task=task
                    ),
                    strategies,
                )
            )
        strategy_to_result: dict[TokenizationStrategy, TaskResult] = {
            r.tokenization_strategy: r for r in task_results
        }
        return strategy_to_result
