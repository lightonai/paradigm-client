import os
from typing import Any, Optional

from llama_index.bridge.pydantic import Field, PrivateAttr
from llama_index.callbacks import CallbackManager
from llama_index.llms.base import (
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
    llm_completion_callback,
)
from llama_index.llms.custom import CustomLLM


class ParadigmLLM(CustomLLM):
    n_tokens: Optional[int]
    model_name: str = Field(description="Model to use")
    generate_kwargs: dict = Field(default_factory=dict, description="Kwargs for generation.")
    _model: Any = PrivateAttr()

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = "alfred-40b-0723",
        n_tokens: Optional[int] = 20,
        callback_manager: Optional[CallbackManager] = None,
        **generate_kwargs: Any,
    ) -> None:
        try:
            from paradigm_client.remote_model import RemoteModel
        except ImportError:
            raise ValueError(
                "You need to install paradigm_client first." "Please install it with `pip install paradigm-client`"
            )
        paradigm_api_key = os.environ.get("PARADIGM_API_KEY") if not api_key else api_key
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
        from paradigm_client.request import MAX_SEQ_LEN

        return LLMMetadata(
            context_window=MAX_SEQ_LEN,
            num_output=self.n_tokens,
            model_name=self.model_name,
        )

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        from paradigm_client.request import CreateParameters, CreateRequest

        if "n_tokens" not in kwargs:
            kwargs["n_tokens"] = self.n_tokens
        if "temperature" not in kwargs and "temperature" in self.generate_kwargs:
            kwargs["temperature"] = self.generate_kwargs["temperature"]

        params = CreateParameters(**kwargs)

        request = CreateRequest(text=prompt, params=params, use_session=True)
        response = self._model.create_from_objects(request)
        try:
            response_text = response[0].completions[0].output_text
        except TypeError:
            print("An error as occured:", response)
            raise
        return CompletionResponse(
            text=response_text,
            raw=response[0],
            additional_kwargs={**kwargs},
        )

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        raise NotImplementedError("Streaming mode not implemented in this wrapper.")
