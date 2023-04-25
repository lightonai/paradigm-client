from enum import Enum
import pickle
from typing import Optional

from pydantic import BaseModel


class FinishReason(str, Enum):
    length = "length"
    generated_stop = "generated_stop"
    stop_regex = "stop_regex"
    stop_word = "stop_word"  # TODO: remove when server up-to-date


class TokenMetadata(BaseModel):
    input_tokens: int
    output_tokens: Optional[int] = None


class LogProbs(BaseModel):
    log_prob: float
    normalized_log_prob: float
    token_log_probs: list[dict[str, float]] | None


class CreateCandidatesOutput(BaseModel):
    output_text: str
    log_probs: LogProbs | None
    finish_reason: FinishReason
    token_metadata: TokenMetadata


class CreateResponse(BaseModel):
    input_text: str
    completions: list[CreateCandidatesOutput]


class AnalyseResponse(BaseModel):
    text: str
    log_probs: LogProbs
    token_metadata: TokenMetadata


class Rankings(BaseModel):
    text: str
    log_probs: LogProbs
    is_greedy_generation: list[bool] | None = None


class SelectResponse(BaseModel):
    reference: str
    rankings: list[Rankings] | None
    best: str
    token_metadata: TokenMetadata


class TokenizeResponse(BaseModel):
    text: str
    n_tokens: int
    tokens: list[dict[str, int]]
    token_metadata: TokenMetadata


class ErrorResponse(BaseModel):
    request_id: Optional[str] = None
    error_msg: str
    status_code: int

    def get_formatted_output(self, unused):
        return self

    def pickle(self):
        return pickle.dumps((self.dict(), None), protocol=pickle.HIGHEST_PROTOCOL)
