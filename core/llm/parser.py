import json
import re
from typing import Type, TypeVar, Tuple, Optional
from pydantic import BaseModel, ValidationError

from .base import BaseLLMProvider
from .schemas import ChatRequest, Message

T = TypeVar("T", bound=BaseModel)


class StructuredOutputParser:
    """
    Utility for enforcing Pydantic schema outputs from standard LLM completions.
    """

    @staticmethod
    def _extract_json_string(text: str) -> str:
        """Strips markdown code blocks and extracts raw JSON content."""
        text = text.strip()
        # Regex to capture JSON block inside ```json ... ``` or ``` ... ```
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        # Fallback to direct string if no backticks found
        return text

    @classmethod
    async def parse_response(
        cls,
        provider: BaseLLMProvider,
        request: ChatRequest,
        response_model: Type[T],
        max_retries: int = 2,
    ) -> Tuple[T, Optional[str]]:
        """
        Sends the request to the LLM and forces response into the requested Pydantic model schema.
        Returns a tuple of (parsed_pydantic_instance, raw_text_response).
        """
        schema_json = json.dumps(response_model.model_json_schema(), indent=2)
        system_instruction = (
            f"You must respond ONLY with a valid JSON object matching this schema:\n"
            f"{schema_json}\n"
            f"Do not include any preambles, markdown formatting, or conversational text."
        )

        # Inject system instruction into request messages
        messages = [Message(role="system", content=system_instruction)] + request.messages
        current_request = request.model_copy(update={"messages": messages, "stream": False})

        last_error = None
        for attempt in range(max_retries + 1):
            response = await provider.generate(current_request)
            raw_content = response.message.content
            cleaned_content = cls._extract_json_string(raw_content)

            try:
                parsed_json = json.loads(cleaned_content)
                parsed_model = response_model.model_validate(parsed_json)
                return parsed_model, raw_content
            except (json.JSONDecodeError, ValidationError) as err:
                last_error = f"Attempt {attempt + 1} failed to parse JSON: {str(err)}"
                # Feed error feedback into message chain for retry
                current_request.messages.append(Message(role="assistant", content=raw_content))
                current_request.messages.append(
                    Message(
                        role="user", 
                        content=f"Your previous response failed validation with error: {str(err)}. Correct the JSON and reply ONLY with the valid object."
                    )
                )

        raise ValueError(f"Failed to generate structured output after {max_retries + 1} attempts. Last error: {last_error}")