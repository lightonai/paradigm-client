from enum import Enum
import pickle
from typing import Optional

from pydantic import BaseModel


class FinishReason(str, Enum):
    length = "length"
    generated_stop = "generated_stop"
    stop_regex = "stop_regex"
    stop_word = "stop_word"  # TODO: remove when server up-to-date


class LogProbs(BaseModel):
    log_prob: float
    normalized_log_prob: float
    token_log_probs: Optional[list[dict[str, float]]]


class CreateCandidatesOutput(BaseModel):
    output_text: str
    log_probs: Optional[LogProbs]
    finish_reason: FinishReason
    completion_id: Optional[str] = None


class CreateResponse(BaseModel):
    response_id: Optional[str] = None
    input_text: str
    completions: list[CreateCandidatesOutput]


class AnalyseResponse(BaseModel):
    text: str
    log_probs: LogProbs


class Rankings(BaseModel):
    text: str
    log_probs: LogProbs
    is_greedy_generation: Optional[list[bool]] = None


class SelectResponse(BaseModel):
    reference: str
    rankings: Optional[list[Rankings]]
    best: str


class TokenizeResponse(BaseModel):
    text: str
    n_tokens: int
    tokens: list[dict[str, int]]


class ErrorResponse(BaseModel):
    request_id: Optional[str] = None
    error_msg: str
    status_code: int

    def get_formatted_output(self, unused):
        return self

    def pickle(self):
        return pickle.dumps((self.dict(), None), protocol=pickle.HIGHEST_PROTOCOL)


class FeedbackResponse(BaseModel):
    status_code: int
