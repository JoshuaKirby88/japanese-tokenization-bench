import textwrap

from ai_sdk import generate_text, openai
from ai_sdk.generate_text import GenerateTextResult
from dotenv import load_dotenv

from src.task.model import NIL_LABELS, Task, TaskConfig, TaskResult, TaskType
from src.tokenizer import TOKENIZATION_STRATEGIES, Tokenizer

load_dotenv()

tokenizer = Tokenizer()


class TaskRunner:
    configs: dict[TaskType, TaskConfig] = {
        "multiple_choice": TaskConfig(
            get_system_prompt=lambda task, strategy: textwrap.dedent("""
                Answer with a single choice label only.
            """),
            get_user_prompt=lambda task, strategy: textwrap.dedent(f"""
                Question: {tokenizer.tokenize(task.question, strategy)}

                Choices:
                {"\n".join(task.options)}
            """),
            evaluate=lambda task, strategy, response: any(
                tokenizer.normalize(response, strategy)
                == tokenizer.normalize(option, strategy)
                and task.options.index(option) in task.ground_truths
                for option in task.options
            ),
        ),
        "nli": TaskConfig(
            get_system_prompt=lambda task, strategy: textwrap.dedent("""
                Answer with a single choice label only.
            """),
            get_user_prompt=lambda task, strategy: textwrap.dedent(f"""
                Premise: {tokenizer.tokenize(task.context or "", strategy)}
                Hypothesis: {tokenizer.tokenize(task.question, strategy)}

                Choices:
                {"\n".join(NIL_LABELS)}
            """),
            evaluate=lambda task, strategy, response: any(
                tokenizer.normalize(response, strategy)
                == tokenizer.normalize(label, strategy)
                and NIL_LABELS.index(label) in task.ground_truths
                for label in NIL_LABELS
            ),
        ),
        "extraction": TaskConfig(
            get_system_prompt=lambda task, strategy: textwrap.dedent("""
                Extract the answer from the "Context", and return only the answer.
            """),
            get_user_prompt=lambda task, strategy: textwrap.dedent(f"""
                Context: {tokenizer.tokenize(task.context or "", strategy)}
                Question: {tokenizer.tokenize(task.question, strategy)}
            """),
            evaluate=lambda task, strategy, response: any(
                tokenizer.normalize(str(gt), strategy)
                == tokenizer.normalize(response, strategy)
                for gt in task.ground_truths
            ),
        ),
        "correction": TaskConfig(
            get_system_prompt=lambda task, strategy: textwrap.dedent("""
                Identify and correct typos in the "Context".
                Return corrections in the format: "Typo -> Correction".
                If multiple exist, list them one per line.
                Return only the corrections.
            """),
            get_user_prompt=lambda task, strategy: textwrap.dedent(f"""
                Context: {tokenizer.tokenize(task.context or "", strategy)}
            """),
            evaluate=lambda task, strategy, response: all(
                tokenizer.normalize(str(gt), strategy)
                in tokenizer.normalize(response, strategy)
                for gt in task.ground_truths
            ),
        ),
    }

    def get_cost_from_response(self, res: GenerateTextResult) -> float:
        dollars = 0.0
        if (
            res.raw_response
            and hasattr(res.raw_response, "usage")
            and res.raw_response.usage
        ):
            retrieved_cost = getattr(res.raw_response.usage, "cost", None)
            if retrieved_cost is not None:
                dollars = retrieved_cost
        return dollars

    def run(self, model: str, task: Task):
        results: list[TaskResult] = []

        for strategy in TOKENIZATION_STRATEGIES:
            config = self.configs[task.type]
            user_prompt = config.get_user_prompt(task, strategy)
            res = generate_text(
                model=openai(model),
                system=config.get_system_prompt(task, strategy),
                prompt=user_prompt,
            )

            result = TaskResult(
                task_id=task.id,
                task_type=task.type,
                tokenization_strategy=strategy,
                user_prompt=user_prompt,
                response=res.text,
                dollars=self.get_cost_from_response(res),
                evaluation=config.evaluate(task, strategy, res.text),
            )
            results.append(result)

        return results
