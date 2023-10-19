import os
import re
import time
from typing import Any, Dict, Generator, Literal, Optional, Union

import aiohttp
import requests
from pydantic import BaseModel, validate_arguments

from .communicator import Communicator, SagemakerCommunicator
from .request import (AnalyseRequest, ChatRequest, CreateParameters,
                      CreateRequest, Endpoint, ScoreRequest, SelectRequest,
                      TokenizeRequest)
from .response import (AnalyseResponse, CreateResponse, CreateResponseChat,
                       CreateResponseCompletion, ErrorResponse,
                       FeedbackResponse, ScoreResponse, SelectResponse,
                       TokenizeResponse)

DEFAULT_BASE_ADDRESS = "https://paradigm.lighton.ai"


def print_logs(msg, end: Optional[str] = None, verbose: bool = False):
    if verbose:
        print(msg, end=end, flush=True)


class RemoteModel:
    def __init__(
        self,
        base_address: Optional[str] = None,
        headers: Optional[dict[str, str]] = None,
        timeout_s: int = 180,
        verbose: bool = False,
        comm=None,
        raise_for_status: bool = False,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None
    ) -> None:
        self._api_key = api_key if api_key is not None else os.environ.get("PARADIGM_API_KEY", None)
        # Do not check for an API key if the communicator is for Sagemaker
        if not isinstance(comm, SagemakerCommunicator):
            assert self._api_key is not None, "You must provide an API key through the PARADIGM_API_KEY environment variable or the api_key parameter"

        self.base_address = DEFAULT_BASE_ADDRESS if base_address is None else base_address
        # Remove '/' at the end of the given base address
        self.base_address = re.sub(r"\/+$", "", self.base_address)

        self.base_headers = {"Content-Type": "application/json", "Accept": "application/json"}
        updated_headers = {**self.base_headers, **{"X-API-KEY": str(self._api_key), "X-Model": str(model_name)}}
        self.comm = comm or Communicator(
            self.base_address,
            headers or updated_headers,
            timeout_s,
            raise_for_status=raise_for_status,
        )
        self.verbose = verbose
        self._wait_for_model_server()

    def _post(
        self, data: Any, endpoint: Endpoint, num_tasks: int, show_progress: bool = True
    ) -> Union[list[SelectResponse], list[AnalyseResponse], list[CreateResponse], list[ScoreResponse], list[TokenizeResponse], ErrorResponse]:

        try:
            response = self.comm(data, endpoint, stream=False, **{"num_tasks": num_tasks, "show_progress": show_progress})

            def convert_output(response):
                if endpoint == Endpoint.select:
                    return SelectResponse(**response)
                elif endpoint == Endpoint.analyse:
                    return AnalyseResponse(**response)
                elif endpoint == Endpoint.create:
                    return CreateResponseCompletion(**response)
                elif endpoint == Endpoint.chat:
                    return CreateResponseChat(**response)
                elif endpoint == Endpoint.tokenize:
                    return TokenizeResponse(**response)
                elif endpoint == Endpoint.score:
                    return ScoreResponse(**response)
            if "responses" not in response:
                if "detail" in response:
                    return ErrorResponse(
                        request_id="", error_msg=response.get("detail"), status_code=response.get("status_code")
                    )
                return ErrorResponse(**response)

            outputs = [convert_output(r) for r in response["responses"]]

            return outputs
        except aiohttp.ContentTypeError as e:
            return ErrorResponse(
                        request_id="", error_msg="An unexpected error occurred. Contact Paradigm Support at support@lighton.ai", status_code=500
                    )

    def _post_stream(self, data: Any, endpoint: Endpoint) -> Generator[str, None, None]:
        return self.comm(data, endpoint=endpoint, stream=True)

    def _post_objects(
        self, objects: Union[BaseModel, list[BaseModel]], endpoint: Endpoint, show_progress: bool = False
    ) -> Union[list[SelectResponse], list[AnalyseResponse], list[CreateResponse], list[TokenizeResponse], ErrorResponse]:
        def compute_num_tasks(obj) -> int:
            if endpoint == Endpoint.create:
                return obj.params.n_completions
            elif endpoint == Endpoint.chat:
                return obj.params.n_completions
            elif endpoint == Endpoint.select:
                return len(obj.candidates)
            else:
                return 1

        if isinstance(objects, list):
            num_tasks = sum([compute_num_tasks(obj) for obj in objects])
            data = [obj.dict() for obj in objects]
        else:
            num_tasks = compute_num_tasks(objects)
            data = objects.dict()
        return self._post(data, endpoint, num_tasks=num_tasks, show_progress=show_progress)

    def _get_params(self, params: Optional[CreateParameters] = None, **kwargs) -> dict[str, Any]:
        if params is None:
            params = CreateParameters()
        if kwargs:
            params = params.copy(update=kwargs)
        return params.dict()

    def _format_single_request_output(
        self,
        response: Union[list[CreateResponse],
        list[AnalyseResponse],
        list[SelectResponse],
        list[ScoreResponse],
        list[TokenizeResponse],
        ErrorResponse],
    ) -> Union[CreateResponse, AnalyseResponse, SelectResponse, ScoreResponse, TokenizeResponse, ErrorResponse]:
        return response if isinstance(response, ErrorResponse) else response[0]

    @validate_arguments
    def create(
        self, prompt: str, params: Optional[CreateParameters] = None, use_session: bool = True, show_progress: bool = False, **kwargs: Any
    ) -> Union[CreateResponseCompletion, ErrorResponse]:
        params = self._get_params(params, **kwargs)
        response = self._post(
            {"text": prompt, "params": params, "use_session": use_session},
            Endpoint.create,
            num_tasks=params.get("n_completions", 1),
            show_progress=show_progress,
        )
        return self._format_single_request_output(response)

    @validate_arguments
    def chat(
        self, messages: list[dict[str, str]], params: Optional[CreateParameters] = None, use_session: bool = True, show_progress: bool = False, **kwargs: Any
    ) -> Union[CreateResponseChat, ErrorResponse]:
        params = self._get_params(params, **kwargs)
        response = self._post(
            {"messages": messages, "params": params, "use_session": use_session},
            Endpoint.chat,
            num_tasks=params.get("n_completions", 1),
            show_progress=show_progress,
        )
        return self._format_single_request_output(response)

    @validate_arguments
    def stream_create(
        self, prompt: str, params: Optional[CreateParameters] = None, **kwargs: Any
    ) -> Generator[str, None, None]:
        params = self._get_params(params, **kwargs)
        return self._post_stream({"text": prompt, "params": params}, endpoint=Endpoint.stream_create)

    @validate_arguments
    def stream_chat(
        self, messages: list[dict[str, str]], params: Optional[CreateParameters] = None, **kwargs: Any
    ) -> Generator[str, None, None]:
        params = self._get_params(params, **kwargs)
        return self._post_stream({"messages": messages, "params": params}, endpoint=Endpoint.stream_chat)

    @validate_arguments
    def analyse(self, text: str, show_progress: bool = False) -> Union[AnalyseResponse , ErrorResponse]:
        response = self._post({"text": text}, Endpoint.analyse, num_tasks=1, show_progress=show_progress)
        return self._format_single_request_output(response)

    @validate_arguments
    def select(
        self,
        reference: str,
        candidates: list[str],
        conjunction: Optional[str] = None,
        evaluate_reference: bool = False,
        return_is_greedy_generation: bool = False,
        return_log_probs: bool = False,
        show_progress: bool = False,
    ) -> Union[SelectResponse, ErrorResponse]:
        response = self._post(
            {
                "reference": reference,
                "candidates": candidates,
                "conjunction": conjunction,
                "evaluate_reference": evaluate_reference,
                "return_is_greedy_generation": return_is_greedy_generation,
                "return_log_probs": return_log_probs,
            },
            Endpoint.select,
            num_tasks=len(candidates),
            show_progress=show_progress,
        )
        return self._format_single_request_output(response)

    @validate_arguments
    def score(self, text:str, show_progress: bool = False) -> Union[ScoreResponse, ErrorResponse]:
        response = self._post({"text": text}, Endpoint.score, num_tasks=1, show_progress=show_progress)
        return self._format_single_request_output(response)

    @validate_arguments
    def tokenize(self, text: str, show_progress: bool = False) -> Union[TokenizeResponse, ErrorResponse]:
        response = self._post({"text": text}, Endpoint.tokenize, num_tasks=1, show_progress=show_progress)
        return self._format_single_request_output(response)

    @validate_arguments
    def create_from_objects(
        self, create_obj: Union[CreateRequest, list[CreateRequest]], show_progress: bool = False
    ) -> Union[list[CreateResponseCompletion], ErrorResponse]:
        return self._post_objects(create_obj, Endpoint.create, show_progress=show_progress)

    @validate_arguments
    def chat_from_objects(
        self, create_obj: Union[ChatRequest, list[ChatRequest]], show_progress: bool = False
    ) -> Union[list[CreateResponseChat], ErrorResponse]:
        return self._post_objects(create_obj, Endpoint.chat, show_progress=show_progress)

    @validate_arguments
    def analyse_from_objects(
        self, analyse_obj: Union[AnalyseRequest, list[AnalyseRequest]], show_progress: bool = False
    ) -> Union[list[AnalyseResponse], ErrorResponse]:
        return self._post_objects(analyse_obj, Endpoint.analyse, show_progress=show_progress)

    @validate_arguments
    def select_from_objects(
        self, select_obj: Union[SelectRequest, list[SelectRequest]], show_progress: bool = False
    ) -> Union[list[SelectResponse], ErrorResponse]:
        return self._post_objects(select_obj, Endpoint.select, show_progress=show_progress)

    @validate_arguments
    def score_from_objects(
        self, score_obj: Union[ScoreRequest, list[ScoreRequest]], show_progress: bool = False
    ) -> Union[list[ScoreResponse], ErrorResponse]:
        return self._post_objects(score_obj, Endpoint.score, show_progress=show_progress)
    

    @validate_arguments
    def tokenize_from_objects(
        self, tokenize_obj: Union[TokenizeRequest, list[TokenizeRequest]], show_progress: bool = False
    ) -> Union[list[TokenizeResponse], ErrorResponse]:
        return self._post_objects(tokenize_obj, Endpoint.tokenize, show_progress=show_progress)

    def log_feedback(
            self,
            rating_id: Union[int, str],
            completion_id: str,
            data: Dict[Literal["flag", "value", "tag", "comment"], Union[float, str, bool]]
    ):
        """
        Log a feedback into Paradigm.
        :param rating_id: ID of the feedback type to use; must be created in Paradigm beforehand
        :param completion_id: ID of the completion to link the feedback to. Can be obtained from a llm response.
        :param data: Actual feedback data. Must be a dictionary
        :return: FeedbackResponse object with the HTTP status code
        """
        if isinstance(self.comm, SagemakerCommunicator):
            raise NotImplementedError("Feedback logging is not implemented for Sagemaker Endpoints.")
        response = requests.post(
            f"{self.base_address}/api/v1/rate/{rating_id}/{completion_id}",
            headers={**self.base_headers, **{'Authorization': f'Api-Key {self._api_key}'}},
            json=data)
        return FeedbackResponse(status_code=response.status_code)

    def _wait_for_model_server(self):
        print_logs(f"Waiting for the ModelServer to be ready", end="", verbose=self.verbose)
        counter = 0
        while not self.comm.is_available():
            print_logs(f".", end="", verbose=self.verbose)
            time.sleep(10.0)
            counter += 1
            if counter > 4:
                break
        if self.comm.is_available():
            print_logs(" ModelServer is ready!", verbose=self.verbose)
        else:
            raise ConnectionError(
                "We're sorry, but the ModelServer is currently unavailable. Please try again later. If you continue to experience issues, please contact our support team for further assistance. Thank you."
            )

    def is_api_key_none(self):
        """
        Checks if no API key has been set
        :return: boolean to False if the private api_key attribute has a different value than the default "None" string
        """
        return self._api_key is None
