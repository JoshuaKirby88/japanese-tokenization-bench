import random
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from ai_sdk import generate_text, openai
from ai_sdk.generate_text import GenerateTextResult
from dotenv import load_dotenv

from src.run.model import ModelConfig
from src.task.model import NIL_LABELS, Task, TaskConfig, TaskResult, TaskType
from src.tokenizer import TokenizationStrategy, Tokenizer

load_dotenv()

tokenizer = Tokenizer()


class TaskRunner:
    configs: dict[TaskType, TaskConfig] = {
        "multiple_choice": TaskConfig(
            get_instruction_prompt=lambda task, strategy: (
                "Answer with exactly one of the provided choices from [Target Question], and nothing else. "
                "Ignore [Auxiliary Questions]. Do not use markdown or extra formatting."
            ),
            get_task_prompt=lambda task, strategy, distractors, length_multiplier: (
                "[Target Question]\n"
                + f"{tokenizer.tokenize(task.question, strategy)}\n\n"
                + (
                    "[Auxiliary Questions]\n"
                    + f"{tokenizer.tokenize('\n'.join(d.question for d in distractors), strategy)}\n\n"
                    if distractors
                    else ""
                )
                + "Choices:\n"
                + "\n".join(tokenizer.tokenize(option, strategy) for option in task.options)
            ),
            get_ground_truths=lambda task, distractors, length_multiplier: task.ground_truths,
            evaluate=lambda task, strategy, response: (
                1.0
                if any(
                    tokenizer.normalize(response, strategy) == tokenizer.normalize(option, strategy)
                    and task.options.index(option) in task.ground_truths
                    for option in task.options
                )
                else 0.0
            ),
        ),
        "nli": TaskConfig(
            get_instruction_prompt=lambda task, strategy: (
                "Answer with exactly one of the provided choices based only on [Target Premise] and [Target Hypothesis]. "
                "Ignore [Auxiliary Premises]. Do not use markdown or extra formatting."
            ),
            get_task_prompt=lambda task, strategy, distractors, length_multiplier: (
                "[Target Premise]\n"
                + f"{tokenizer.tokenize(task.context or '', strategy)}\n\n"
                + (
                    "[Auxiliary Premises]\n"
                    + f"{tokenizer.tokenize('\n'.join(d.context or '' for d in distractors), strategy)}\n\n"
                    if distractors
                    else ""
                )
                + "[Target Hypothesis]\n"
                + f"{tokenizer.tokenize(task.question, strategy)}\n\n"
                + "Choices:\n"
                + "\n".join(label for label in NIL_LABELS)
            ),
            get_ground_truths=lambda task, distractors, length_multiplier: task.ground_truths,
            evaluate=lambda task, strategy, response: (
                1.0
                if any(
                    tokenizer.normalize(response, strategy) == tokenizer.normalize(label, strategy) and NIL_LABELS.index(label) in task.ground_truths
                    for label in NIL_LABELS
                )
                else 0.0
            ),
        ),
        "extraction": TaskConfig(
            get_instruction_prompt=lambda task, strategy: (
                'Extract the answer from [Target Context] for [Target Question]. '
                'Ignore [Auxiliary Context]. Return only the answer. Do not use markdown or extra formatting.'
            ),
            get_task_prompt=lambda task, strategy, distractors, length_multiplier: (
                "[Target Context]\n"
                + f"{tokenizer.tokenize(task.context or '', strategy)}\n\n"
                + (
                    "[Auxiliary Context]\n"
                    + f"{tokenizer.tokenize('\n'.join(d.context or '' for d in distractors), strategy)}\n\n"
                    if distractors
                    else ""
                )
                + "[Target Question]\n"
                + f"{tokenizer.tokenize(task.question, strategy)}"
            ),
            get_ground_truths=lambda task, distractors, length_multiplier: task.ground_truths,
            evaluate=lambda task, strategy, response: max(
                (TaskRunner.compute_f1(tokenizer.normalize(response, strategy), tokenizer.normalize(str(gt), strategy)) for gt in task.ground_truths),
                default=0.0,
            ),
        ),
        "correction": TaskConfig(
            get_instruction_prompt=lambda task, strategy: "\n".join(
                [
                    "This is a typo-correction task for Japanese text.",
                    "Process both [Primary Text] and [Additional Text]. Output only typo corrections with minimal edits.",
                    "Spaces are analysis artifacts. Do not treat spacing restoration/removal as a correction.",
                    "Before answering, verify each pair is minimal and local.",
                    "",
                    "Rules:",
                    '1) Each line must be exactly: "Typo -> Correction".',
                    "2) The left side must be a contiguous substring from Text.",
                    "3) The right side must be only the replacement for that typo.",
                    "4) Each pair must be minimal (typo span only, no extra context).",
                    "5) No paraphrasing, grammar rewriting, or summarization.",
                    "6) No content-word substitution (noun/stem rewrites).",
                    "7) No explanations, notes, extra symbols, or extra prose.",
                ]
            ),
            get_task_prompt=lambda task, strategy, distractors, length_multiplier: (
                "[Primary Text]\n"
                + f"{tokenizer.tokenize(task.question, strategy)}\n\n"
                + (
                    "[Additional Text]\n"
                    + f"{tokenizer.tokenize('\n'.join(d.question for d in distractors), strategy)}"
                    if distractors
                    else ""
                )
            ),
            get_ground_truths=lambda task, distractors, length_multiplier: (
                [str(gt) for gt in task.ground_truths] + [str(gt) for d in distractors for gt in d.ground_truths]
            ),
            evaluate=lambda task, strategy, response: TaskRunner.correction_score(task, strategy, response),
        ),
        "char_counting": TaskConfig(
            get_instruction_prompt=lambda task, strategy: (
                'Count the number of "Character" in "Text". Answer with a single number only. Do not use markdown or extra formatting.'
            ),
            get_task_prompt=lambda task, strategy, distractors, length_multiplier: (
                f"Text: {tokenizer.tokenize(task.context or '', strategy)}\n" + f"Character: {task.question}"
            ),
            get_ground_truths=lambda task, distractors, length_multiplier: task.ground_truths,
            evaluate=lambda task, strategy, response: (
                max(
                    (max(0.0, 1.0 - abs(int(response.strip()) - int(gt)) / int(gt)) if int(gt) > 0 else (1.0 if int(response.strip()) == 0 else 0.0))
                    for gt in task.ground_truths
                )
                if response.strip().isdigit()
                else 0.0
            ),
        ),
    }

    @staticmethod
    def correction_score(task: Task, strategy: TokenizationStrategy, response: str):
        def normalize_part(text: str):
            return tokenizer.normalize(text.strip(), strategy).lower()

        def parse_pairs(text: str):
            pairs: set[tuple[str, str]] = set()
            for raw_line in text.splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                line = re.sub(r"^\d+[.)]\s*", "", line)
                match = re.compile(r"^(.+?)\s*->\s*(.+)$").match(line)
                if not match:
                    continue
                typo = normalize_part(match.group(1))
                correction = normalize_part(match.group(2))
                if typo and correction:
                    pairs.add((typo, correction))
            return pairs

        ground_truth_pairs = parse_pairs("\n".join(str(gt) for gt in task.ground_truths))
        predicted_pairs = parse_pairs(response)

        true_positives = len(predicted_pairs & ground_truth_pairs)
        precision = true_positives / len(predicted_pairs) if predicted_pairs else (1.0 if not ground_truth_pairs else 0.0)
        recall = true_positives / len(ground_truth_pairs) if ground_truth_pairs else 1.0
        return (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0.0

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
        if res.raw_response and hasattr(res.raw_response, "usage") and res.raw_response.usage:
            retrieved_cost = getattr(res.raw_response.usage, "cost", None)
            if retrieved_cost is not None:
                try:
                    dollars = float(retrieved_cost)
                except (TypeError, ValueError):
                    pass
        return dollars

    @staticmethod
    def select_distractors(task: Task, distractor_candidates: list[Task], length_multiplier: int) -> list[Task]:
        pool = [d for d in distractor_candidates if d.id != task.id and d.type == task.type]
        if not pool:
            return []
        sample_size = min(length_multiplier, len(pool))
        return random.sample(pool, sample_size)

    def run_strategy(self, model_config: ModelConfig, strategy: TokenizationStrategy, task: Task, distractors: list[Task], length_multiplier: int):
        config = self.configs[task.type]
        task_prompt = config.get_task_prompt(task, strategy, distractors, length_multiplier)
        effective_ground_truths = config.get_ground_truths(task, distractors, length_multiplier)
        evaluation_task = Task(
            id=task.id,
            type=task.type,
            context=task.context,
            question=task.question,
            options=task.options,
            ground_truths=effective_ground_truths,
        )
        user_prompt = "\n\n".join([config.get_instruction_prompt(task, strategy), task_prompt])

        res = generate_text(model=openai(model_config.model), reasoning=model_config.reasoning, prompt=user_prompt)

        return TaskResult(
            task_id=task.id,
            task_type=task.type,
            tokenization_strategy=strategy,
            task_prompt=task_prompt,
            response=res.text,
            dollars=self.get_cost_from_response(res),
            evaluation=config.evaluate(evaluation_task, strategy, res.text),
            ground_truths=effective_ground_truths,
            reasoning=res.reasoning,
        )

    def run(
        self,
        model_config: ModelConfig,
        strategies: list[TokenizationStrategy],
        task: Task,
        distractor_candidates: list[Task],
        length_multiplier: int,
    ):
        distractors = self.select_distractors(task=task, distractor_candidates=distractor_candidates, length_multiplier=length_multiplier)
        with ThreadPoolExecutor() as executor:
            task_results = list(
                executor.map(
                    lambda strategy: self.run_strategy(
                        model_config=model_config,
                        strategy=strategy,
                        task=task,
                        distractors=distractors,
                        length_multiplier=length_multiplier,
                    ),
                    strategies,
                )
            )
        strategy_to_result: dict[TokenizationStrategy, TaskResult] = {r.tokenization_strategy: r for r in task_results}
        return strategy_to_result
