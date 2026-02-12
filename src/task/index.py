import textwrap

from ai_sdk import generate_text, openai
from dotenv import load_dotenv

from src.task.model import TASK_TYPES, Task, TaskConfig, TaskResult, TaskType
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
                response.lower() in task.options
                and task.options.index(response.lower()) in task.ground_truths
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
                {"\n".join(["Entailment", "Neutral", "Contradiction"])}
            """),
            evaluate=lambda task, strategy, response: (
                response.lower() in ["entailment", "neutral", "contradiction"]
                and ["entailment", "neutral", "contradiction"].index(response.lower())
                in task.ground_truths
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
            evaluate=lambda task, strategy, response: (
                response.lower() in task.ground_truths
            ),
        ),
    }

    def run(self, task: Task):
        results: list[TaskResult] = []

        for strategy in TOKENIZATION_STRATEGIES:
            config = self.configs[task.type]
            res = generate_text(
                model=openai("mistralai/ministral-3b-2512"),
                system=config.get_system_prompt(task, strategy),
                prompt=config.get_user_prompt(task, strategy),
            )
            result = TaskResult(
                task_id=task.id,
                task_type=task.type,
                tokenization_strategy=strategy,
                response=res.text,
                evaluation=config.evaluate(task, strategy, res.text),
            )
            results.append(result)

        return results
