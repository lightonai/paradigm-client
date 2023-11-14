import os
import warnings
from dataclasses import dataclass
from enum import Enum
from re import Pattern
from typing import Optional, Union

from pydantic import BaseModel, Field, validator

MAX_SEQ_LEN = int(os.environ.get("MAX_SEQ_LEN", 2048))


class Endpoint(str, Enum):
    create = "create"
    stream_create = "stream_create"
    chat = "chat"
    stream_chat = "stream_chat"
    analyse = "analyse"
    select = "select"
    score = "score"
    tokenize = "tokenize"


class Status(str, Enum):
    waiting = "waiting"
    in_progress = "in_progress"
    done = "done"


class CreateParameters(BaseModel):
    n_tokens: int = 20  # number of tokens to generate
    temperature: float = 0.7  # temperature to apply to the logits
    top_p: float = 0.9  # p parameter for nucleus sampling
    n_completions: int = 1  # number of generated samples per input
    generate_stop: bool = True
    seed: Optional[int] = None  # set the seed for the sampling phase
    show_special_tokens: bool = False
    biases: dict[int, float] = Field(default_factory=dict)
    stop_regex: Optional[Pattern] = None
    stop_sequences: Optional[list[str]] = None
    prettify: bool = True
    return_log_probs: bool = False
    echo: bool = False

    class Config:
        extra: str = "forbid"
        allow_population_by_field_name: bool = True

    @validator("n_tokens")
    def check_n_tokens(cls, n_tokens):
        if n_tokens < 1:
            raise ValueError(f"n_tokens should satisfy n_tokens > 0. Found {n_tokens}")
        if n_tokens >= MAX_SEQ_LEN:
            raise ValueError(f"n_tokens should satisfy n_tokens < {MAX_SEQ_LEN}. Found {n_tokens}")
        return n_tokens

    @validator("temperature")
    def check_temperature(cls, temperature):
        if not 0.0 <= temperature <= 2.0:
            raise ValueError(f"temperature should satisfy 0.0 <= temperature <= 2.0. Found {temperature}")
        return temperature

    @validator("top_p")
    def check_p(cls, top_p):
        if not 0.0 < top_p <= 1.0:
            raise ValueError(f"top_p nucleus parameter should satisfy 0.0 < top_p < 1.0. Found {top_p}")
        return min(top_p, 0.999)  # we do not want p = 1.0

    @validator("n_completions")
    def check_n_completions(cls, n_completions):
        if n_completions < 1:
            raise ValueError(f"n_completions parameter should be positive. Found {n_completions}")
        return n_completions

    @validator("seed")
    def check_seed(cls, seed):
        if seed is None:
            return seed
        if seed < 0:
            raise ValueError(f"seed parameter should satisfy seed >= 0. Found {seed}")
        return seed

    @validator("biases", always=True)
    def check_biases(cls, biases):
        for token_id, bias_value in biases.items():
            # out of range token_ids are ignored in task.transform_output()
            if not -100.0 <= bias_value <= 100.0:
                raise ValueError(f"bias_value from should satisfy -100.0 <= bias <= 100.0. Found {bias_value}")
        return biases

    def dict(self, *args, **kwargs):
        d = super().dict(*args, **kwargs)
        # remove stop_sequences if not used to have compatibility with previous versions of the API
        if d["stop_sequences"] is None:
            d.pop("stop_sequences")
        if d["stop_regex"] is not None:
            warnings.warn("`stop_regex` has been deprecated in favor of `stop_sequences`. It is ignored.")
        d.pop("stop_regex")
        return d


def is_empty_text(txt: str) -> bool:
    return not txt


def check_text(text: Union[str, list[str]]) -> Union[str, list[str]]:
    def validate(t: str):
        if is_empty_text(t):
            raise ValueError(f"Receive empty text. Abort")

    if isinstance(text, str):
        validate(text)
    else:
        for t in text:
            validate(t)
    return text


def check_messages(
    messages: Union[list[dict[str, str]],list[list[dict[str, str]]]]
) -> Union[list[dict[str, str]],list[list[dict[str, str]]]]:
    def validate(t: list[dict[str, str]]):
        if len(t) == 0:
            raise ValueError("Received empty messages. Abort")

        valid_roles = ["system", "user", "assistant"]

        for message in t:
            if not isinstance(message, dict):
                raise ValueError("Each message should be a dictionary.")

            if "role" not in message:
                raise ValueError("Role key is missing in a message.")

            if message["role"] not in valid_roles:
                raise ValueError(
                    f"Invalid role '{message['role']}' in a message. Role should be one of {', '.join(valid_roles)}."
                )

            if "content" not in message:
                raise ValueError("Content key is missing in a message.")

            if not isinstance(message["role"], str):
                raise ValueError("Role should be a string.")

            if not isinstance(message["content"], str):
                raise ValueError("Content should be a string.")

    if len(messages) > 0 and isinstance(messages[0], list):
        for t in messages:
            validate(t)
    else:
        validate(messages)
    return messages


@dataclass
class Progress:
    status: Status
    completed_task: int


class CreateRequest(BaseModel):
    text: str
    params: Optional[CreateParameters] = None
    use_session: bool = True
    _validate_text = validator("text", allow_reuse=True)(check_text)

    @validator("params", always=True, pre=True)
    def check_params(cls, params, values) -> CreateParameters:
        if params is None:
            return CreateParameters()
        return params

    class Config:
        extra: str = "forbid"


class ChatRequest(BaseModel):
    messages: list[dict[str, str]]
    params: Optional[CreateParameters] = None
    use_session: bool = True
    _validate_messages = validator("messages", allow_reuse=True)(check_messages)

    @validator("params", always=True, pre=True)
    def check_params(cls, params, values) -> CreateParameters:
        if params is None:
            return CreateParameters()
        return params


    class Config:
        extra: str = "forbid"
        

class AnalyseRequest(BaseModel):
    text: str
    _validate_text = validator("text", allow_reuse=True)(check_text)

    class Config:
        extra: str = "forbid"


class SelectRequest(BaseModel):
    reference: str
    candidates: list[str]
    conjunction: Optional[str] = None
    evaluate_reference: bool = False
    return_is_greedy_generation: bool = False
    return_log_probs: bool = False
    _validate_reference = validator("reference", allow_reuse=True)(check_text)

    class Config:
        extra: str = "forbid"

    @validator("candidates")
    def check_candidates(cls, candidates, values):
        return check_text(candidates)


class ScoreRequest(BaseModel):
    text: str

    class Config:
        extra: str = "forbid"


class TokenizeRequest(BaseModel):
    text: str
    _validate_text = validator("text", allow_reuse=True)(check_text)

    class Config:
        extra: str = "forbid"
