from functools import partial
import os
import re
from re import Pattern
from typing import Mapping, Any, Optional, Union

try:
    from langchain.llms.base import LLM
    from langchain.callbacks.manager import CallbackManagerForLLMRun
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "Please install langchain: pip install langchain"
    )
from pydantic import Field

from paradigm_client.communicator import SagemakerCommunicator
from paradigm_client.remote_model import RemoteModel
from paradigm_client.request import CreateParameters


DEFAULT_BASE_ADDRESS = "https://paradigm.lighton.ai"


class ParadigmLLM(LLM):
    def __init__(self, client, **kwargs):
        super().__init__(
            client=client,
            **kwargs
        )
        self.stop_regex = self._transform_stop_words_to_regex(
            stop_words=self.stop_words, stop_regex=self.stop_regex
        )
        
    client: RemoteModel
    stop_words: Optional[list[str]] = None
    n_tokens: int = 20  # number of tokens to generate
    temperature: float = 0.7  # temperature to apply to the logits
    top_p: float = 0.9  # p parameter for nucleus sampling
    n_completions: int = 1  # number of generated samples per input
    generate_stop: bool = True
    seed: Optional[int] = None  # set the seed for the sampling phase
    show_special_tokens: bool = False
    biases: dict[int, float] = Field(default_factory=dict)
    stop_regex: Optional[Pattern] = None
    prettify: bool = True
    return_log_probs: bool = False
    echo: bool = False

    def _transform_stop_words_to_regex(
        self, stop_words: Optional[list[str]], stop_regex: Optional[Pattern]
    ) -> Optional[Pattern]:
        if not stop_regex:  # we use stop_regex in remote models
            if stop_words:
                return r"(?i)(" + "|".join(re.escape(word) for word in stop_words) + ")"
        return stop_regex

    def _default_params(self) -> dict[str, Any]:
        return {
            "n_tokens": self.n_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "n_completions": self.n_completions,
            "generate_stop": self.generate_stop,
            "seed": self.seed,
            "show_special_tokens": self.show_special_tokens,
            "biases": self.biases,
            "stop_regex": self.stop_regex,
            "prettify": self.prettify,
            "return_log_probs": self.return_log_probs,
            "echo": self.echo,
        }

    @property
    def _llm_type(self) -> str:
        return "paradigm"

    def _call(
        self,
        prompt: str,
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        if not stop:
            stop = self.stop_words
        stop_regex = self._transform_stop_words_to_regex(
            stop_words=stop, stop_regex=self.stop_regex
        )
        params = CreateParameters(
            **self._default_params(),
        )
        params.stop_regex = stop_regex
        text_callback = None
        if run_manager:
            text_callback = partial(run_manager.on_llm_new_token, verbose=self.verbose)
        text = ""
        try:
            for completion in self.client.create(
                prompt=prompt,
                params=params,
            ).completions:
                if text_callback:
                    text_callback(completion)
                text += completion.output_text
            return text
        except Exception as e:
            raise e

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return self._default_params()

    def get_num_tokens(self, text: str) -> int:
        return self.client.tokenize(text).n_tokens
