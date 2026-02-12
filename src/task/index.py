import textwrap

from ai_sdk import generate_text, openai
from ai_sdk.generate_text import GenerateTextResult
from dotenv import load_dotenv

from src.task.model import NIL_LABELS, Task, TaskConfig, TaskResult, TaskType
from src.tokenizer import TOKENIZATION_STRATEGIES, Tokenizer

_ = load_dotenv()

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
            evaluate=lambda task, strategy, response: (
                response in task.options
                and task.options.index(response) in task.ground_truths
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
            evaluate=lambda task, strategy, response: (
                response in NIL_LABELS
                and NIL_LABELS.index(response) in task.ground_truths
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
            evaluate=lambda task, strategy, response: response in task.ground_truths,
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
