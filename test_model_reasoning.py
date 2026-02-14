from ai_sdk import generate_text, openai
from dotenv import load_dotenv

import src.patch_sdk as _
from src.run.model import REASONINGS, ModelConfig

load_dotenv()


def run_reasoning_test(model_config: ModelConfig) -> None:
    print(f"Reasoning: {model_config.reasoning}")

    result = generate_text(
        model=openai(model_config.model),
        reasoning=model_config.reasoning,
        prompt="What is 2 plus 2?",
    )

    reasoning_content = getattr(result, "reasoning", "")
    print(f"Reasoning Length: {len(reasoning_content or '')}")

    print("-" * 40)


def main(model: str) -> None:
    for reasoning in REASONINGS:
        try:
            model_config = ModelConfig(model=model, reasoning=reasoning)
            run_reasoning_test(model_config)
        except Exception as e:
            print(f"Error testing reasoning {reasoning}: {e}")


if __name__ == "__main__":
    main("mistralai/mistral-small-3.2-24b-instruct:floor")
