import os
from typing import Any, Optional, Sequence
import datetime
import time

from paradigm_client.remote_model import RemoteModel
from paradigm_client.communicator import SagemakerCommunicator
from paradigm_client.request import (
    CreateParameters,
    CreateRequest,
    ChatRequest,
    SelectRequest,
)
from paradigm_client.response import CreateResponse, CreateResponseChat, SelectResponse


from llama_index.bridge.pydantic import Field, PrivateAttr
from llama_index.llms.custom import CustomLLM
from llama_index.callbacks import CallbackManager
from llama_index.llms.base import (
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
    llm_completion_callback,
    llm_chat_callback,
    ChatMessage,
    ChatResponse,
    MessageRole,
)


class ParadigmLLM(CustomLLM):
    n_tokens: Optional[int]
    model_name: Optional[str] = Field(description="Model to use")
    generate_kwargs: dict = Field(
        default_factory=dict, description="Kwargs for generation."
    )
    _model: Any = PrivateAttr()

    def __init__(
        self,
        model_name: Optional[str] = "alfred-40b-1123",
        api_key: Optional[str] = None,
        n_tokens: Optional[int] = 1024,
        callback_manager: Optional[CallbackManager] = None,
        **generate_kwargs: Any,
    ) -> None:
        try:
            from paradigm_client.remote_model import RemoteModel
        except ImportError:
            raise ValueError(
                "You need to install paradigm_client first."
                "Please install it with `pip install paradigm-client`"
            )
        paradigm_api_key = (
            os.environ.get("PARADIGM_API_KEY") if not api_key else api_key
        )
        assert paradigm_api_key, "PARADIGM_API_KEY is not set"

        self._model = RemoteModel(
            model_name=model_name,
            api_key=paradigm_api_key,
        )
        generate_kwargs = generate_kwargs or {}
        super().__init__(
            model_name=model_name,
            n_tokens=n_tokens,
            generate_kwargs=generate_kwargs,
            callback_manager=callback_manager,
        )

    @classmethod
    def class_name(cls) -> str:
        return "paradigm"

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=8192,
            num_output=self.n_tokens,
            model_name=self.model_name,
            is_chat_model=True,
        )

    def _current_prefix(self):
        date = datetime.datetime.fromtimestamp(time.time())
        prefix = "You are Alfred, a helpful assistant trained by LightOn. Knowledge cutoff: November 2022. Current date: {date}"
        return prefix.format(date=date.strftime("%B %d, %Y"))

    def _wrap_prompt_to_chatml(self, prompt: str):
        return (
            "<start_system>"
            + self._current_prefix()
            + "<end_message>"
            + "<start_user>"
            + prompt
            + "<end_message><start_assistant>"
        )

    def _messages_to_prompt(self, messages: Sequence[ChatMessage]) -> str:
        prompt = ""
        for message in messages:
            if message.role.value != "system":
                prompt += f"{message.role.value.capitalize()}: "
            prompt += message.content + "\n"
        return prompt + "Assistant:"

    def _get_params(self, **kwargs: Any) -> CreateParameters:
        if "n_tokens" not in kwargs:
            if "max_tokens" in kwargs:
                kwargs["n_tokens"] = kwargs.pop("max_tokens")
            else:
                kwargs["n_tokens"] = self.n_tokens
        if "temperature" not in kwargs and "temperature" in self.generate_kwargs:
            kwargs["temperature"] = self.generate_kwargs["temperature"]
        if self.metadata.is_chat_model:
            kwargs["stop_sequences"] = kwargs.get("stop_sequences", [])
            kwargs["stop_sequences"].append("<end_message>")
        kwargs["prettify"] = False
        return CreateParameters(**kwargs)

    def _complete(
        self, prompts: str | list[str], **kwargs: Any
    ) -> list[CreateResponse]:
        if isinstance(prompts, str):
            prompts = [prompts]
        params = self._get_params(**kwargs)
        requests = [
            CreateRequest(text=prompt, params=params, use_session=True)
            for prompt in prompts
        ]
        responses = self._model.create_from_objects(requests)
        try:
            text = responses[0].completions[0].output_text
        except TypeError as e:
            print(responses)
            raise e
        return responses

    def _chat(
        self,
        batch_messages: Sequence[ChatMessage] | list[Sequence[ChatMessage]],
        **kwargs: Any,
    ) -> list[CreateResponseChat]:
        if isinstance(batch_messages[0], ChatMessage):
            batch_messages = [batch_messages]
        messages_to_send = []
        for messages in batch_messages:
            messages_to_send.append(
                [{"role": m.role.value, "content": m.content} for m in messages]
            )
        params = self._get_params(**kwargs)
        requests = [
            ChatRequest(messages=messages, params=params, use_session=True)
            for messages in messages_to_send
        ]
        responses = self._model.chat_from_objects(requests)
        try:
            text = responses[0].completions[0].output_text
        except TypeError as e:
            print(responses)
            raise e
        return responses

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        if self.metadata.is_chat_model:
            prompt = self._wrap_prompt_to_chatml(prompt)

        responses = self._complete(prompt, **kwargs)
        return CompletionResponse(
            text=responses[0].completions[0].output_text,
            raw=responses[0],
            additional_kwargs=kwargs,
        )

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=self._current_prefix())
        ] + messages
        if self.metadata.is_chat_model:
            responses = self._chat(messages, **kwargs)
            response_text = responses[0].completions[0].output_text
        else:
            prompt = self._messages_to_prompt(messages)
            responses = self._complete(prompt, **kwargs)
            response_text = responses[0].completions[0].output_text.split("\nUser:")[0]

        return ChatResponse(
            message=ChatMessage(role=MessageRole.ASSISTANT, content=response_text),
            raw=responses[0],
            additional_kwargs=kwargs,
        )

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        raise NotImplementedError("Streaming mode not implemented in this wrapper.")
