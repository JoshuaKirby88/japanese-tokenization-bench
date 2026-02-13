from typing import Any

from ai_sdk.providers.openai import OpenAIModel

_is_patched = False


def patch_openai_provider():
    global _is_patched
    if _is_patched:
        return

    original_generate_text = OpenAIModel.generate_text

    def patched_generate_text(
        self: OpenAIModel, *, prompt: str | None = None, system: str | None = None, messages: list[dict[str, Any]] | None = None, **kwargs: Any
    ):
        reasoning = kwargs.pop("reasoning", None)
        if reasoning:
            if "/" in self._model:
                extra_body = kwargs.setdefault("extra_body", {})
                extra_body["reasoning_effort"] = reasoning
                extra_body["include_reasoning"] = reasoning != "none"
            else:
                kwargs["reasoning_effort"] = reasoning

        result = original_generate_text(self, prompt=prompt, system=system, messages=messages, **kwargs)

        raw_response = result.get("raw_response")
        if raw_response and hasattr(raw_response, "choices") and raw_response.choices:
            message = raw_response.choices[0].message
            reasoning_val = getattr(message, "reasoning_content", None) or getattr(message, "reasoning", None) or getattr(message, "thought", None)
            if reasoning_val:
                result["reasoning"] = reasoning_val

        return result

    OpenAIModel.generate_text = patched_generate_text
    _is_patched = True


patch_openai_provider()
