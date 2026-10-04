import re
from typing import Any


USER_INPUT_RE = re.compile(
    r"<user_input>\s*(.*?)\s*</user_input>",
    flags=re.IGNORECASE | re.DOTALL,
)

ROUTING_TAG_RE = re.compile(
    r"(?<!\w)[!@]ppro\b",
    flags=re.IGNORECASE,
)


class PromptInterpretationError(ValueError):
    pass


def content_to_text_blocks(content: Any) -> list[str]:
    if isinstance(content, str):
        return [content]

    if isinstance(content, list):
        blocks: list[str] = []

        for part in content:
            if not isinstance(part, dict):
                continue

            if part.get("type") != "text":
                continue

            text = part.get("text")
            if isinstance(text, str):
                blocks.append(text)

        return blocks

    return []


def clean_request(text: str) -> str:
    text = ROUTING_TAG_RE.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_current_user_request(payload: dict[str, Any]) -> str:
    messages = payload.get("messages")

    if not isinstance(messages, list):
        raise PromptInterpretationError("Campo 'messages' ausente ou inválido.")

    for message in reversed(messages):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue

        user_inputs: list[str] = []

        for block in content_to_text_blocks(message.get("content")):
            user_inputs.extend(USER_INPUT_RE.findall(block))

        for text in reversed(user_inputs):
            cleaned = clean_request(text)

            if cleaned:
                return cleaned

    raise PromptInterpretationError(
        "Nenhum bloco <user_input> útil foi encontrado na requisição do Trae."
    )


def build_perplexity_prompt(payload: dict[str, Any]) -> str:
    return extract_current_user_request(payload)
