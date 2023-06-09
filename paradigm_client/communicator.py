from abc import abstractmethod, ABC
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
from typing import Any, Generator, Optional, Union

import aiohttp
import requests
from tqdm import tqdm

from .request import Endpoint, Progress, Status
from .response import CreateResponse, AnalyseResponse, SelectResponse, TokenizeResponse, ErrorResponse

IS_BOTO3_AVAILABLE = True
try:
    import boto3
except ModuleNotFoundError as e:
    IS_BOTO3_AVAILABLE = False


def _safe_run_tasks(tasks):
    async def _run_tasks(tasks):
        return await asyncio.gather(*tasks)

    event_loop = asyncio._get_running_loop()
    if event_loop and event_loop.is_running():
        with ThreadPoolExecutor(1) as pool:
            return pool.submit(lambda: asyncio.run(_run_tasks(tasks))).result()
    return asyncio.run(_run_tasks(tasks))


class AbstractCommunicator(ABC):
    @abstractmethod
    def __call__(self, data: Any, endpoint: Endpoint, stream: bool, **kwargs):
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        pass

    @abstractmethod
    def get_session_id(self) -> str:
        pass

    @abstractmethod
    def is_available(self) -> bool:
        pass


class Communicator(AbstractCommunicator):
    def __init__(
        self, base_address: str, headers: dict[str, str], timeout_s: Union[int, float], raise_for_status: bool = False
    ) -> None:
        self.base_address = base_address
        self.headers = headers
        self.timeout_s = aiohttp.ClientTimeout(total=timeout_s)
        self.model_name: str | None = None
        self.raise_for_status = raise_for_status

    async def _post(self, data: Any, endpoint: Endpoint, session_id: Optional[str] = None):
        request_id = {"request_id": session_id} if session_id else {}
        async with aiohttp.ClientSession(
            base_url=self.base_address,
            headers=self.headers | request_id,
            timeout=self.timeout_s,
            raise_for_status=self.raise_for_status,
        ) as session:
            async with session.post(f"/llm/{endpoint.value}", json={"data": data}) as resp:
                response = await resp.json()
                response["status_code"] = resp.status
        return response

    async def _get_progress_task(self, session_id: str, num_tasks: int, endpoint: Endpoint):
        num_completed_tasks = 0
        with tqdm(total=num_tasks) as pbar:
            async with aiohttp.ClientSession(base_url=self.base_address) as session:
                while True:
                    async with session.get(f"/progress/{session_id}") as resp:
                        response = await resp.json()
                        progress = Progress(**response)
                        if progress.status == Status.done.value:
                            pbar.set_description(f"{endpoint.value.upper()} Status: {Status.done}")
                            pbar.update(num_tasks - num_completed_tasks)
                            break
                        delta = progress.completed_task - num_completed_tasks
                        num_completed_tasks = progress.completed_task
                        pbar.set_description(f"{endpoint.value.upper()} Status: {progress.status}")
                        if delta > 0:
                            pbar.update(delta)
                        await asyncio.sleep(2.0)

    def stream_response(self, data):
        stream = requests.post(  # TODO: implement stream with aiohttp
            f"{self.base_address}/llm/{Endpoint.stream_create.value}",
            json={"data": data},
            headers=self.headers,
            timeout=self.timeout_s.total,
            stream=True,
        )
        yield from stream.iter_content(chunk_size=1024, decode_unicode=True)

    def __call__(
        self, data: Any, endpoint: Endpoint, stream: bool, **kwargs
    ) -> Union[list[SelectResponse], list[AnalyseResponse], list[CreateResponse], list[TokenizeResponse], ErrorResponse, Generator[str, None, None]]:

        if self.model_name is None:
            self.model_name = self.get_model_name()

        show_progress = kwargs.get("show_progress", False)
        num_tasks = kwargs.get("num_tasks", 1)

        if not stream:
            session_id = None
            tasks = []
            if show_progress:
                session_id = self.get_session_id()
                tasks.append(self._get_progress_task(session_id, num_tasks, endpoint))

            tasks.append(self._post(data, endpoint, session_id))
            return _safe_run_tasks(tasks)[-1]
        else:
            return self.stream_response(data)

    def get_model_name(self) -> str:
        return requests.get(f"{self.base_address}/model").text

    def is_available(self) -> bool:
        try:
            r = requests.get(f"{self.base_address}/availability", headers={"X-API-KEY": self.headers.get("X-API-KEY", None)})
            return r.status_code == 200
        except Exception as e:
            print(f"Detected exception {e}")
            return False

    def get_session_id(self) -> str:
        return requests.get(f"{self.base_address}/session").text


class SagemakerCommunicator(AbstractCommunicator):
    def __init__(self, endpoint_name: str) -> None:
        self.endpoint_name = endpoint_name
        assert IS_BOTO3_AVAILABLE, "boto3 is required to use the Sagemaker client"
        self._runtime_sm_client = boto3.client("sagemaker-runtime")
        self._sm_client = boto3.client("sagemaker")

        self.model_name: str | None = None

    def _invoke_endpoint(self, endpoint: str, session_id: Optional[str] = None, data: Optional[dict] = None):
        request_id = {"request_id": session_id} if session_id else {}
        data = {"data": data} if data else {}

        response = self._runtime_sm_client.invoke_endpoint(
            EndpointName=self.endpoint_name,
            ContentType="application/json",
            Body=json.dumps({"endpoint": endpoint} | request_id | data),
        )

        return response["Body"].read().decode("utf8")

    async def _post(self, data: Any, endpoint: Endpoint, session_id: Optional[str] = None):
        return json.loads(self._invoke_endpoint(f"llm/{endpoint.value}", session_id, data))

    async def _get_progress_task(self, session_id: str, num_tasks: int, endpoint: Endpoint):
        num_completed_tasks = 0
        with tqdm(total=num_tasks) as pbar:
            while True:
                resp = self._invoke_endpoint(f"/progress/{session_id}")
                response = json.loads(resp)
                progress = Progress(**response)
                if progress.status == Status.done.value:
                    pbar.set_description(f"{endpoint.value.upper()} Status: {Status.done}")
                    pbar.update(num_tasks - num_completed_tasks)
                    break
                delta = progress.completed_task - num_completed_tasks
                num_completed_tasks = progress.completed_task
                pbar.set_description(f"{endpoint.value.upper()} Status: {progress.status}")
                if delta > 0:
                    pbar.update(delta)
                await asyncio.sleep(2.0)

    def stream_response(self, data):
        session_id = _safe_run_tasks([self._post(data, Endpoint.stream_create)])[0].pop('request_id')
        done = False
        while not done:
            stream_tokens = _safe_run_tasks([self._post({}, Endpoint.stream_tokens, session_id=session_id)])[0]
            yield ''.join(stream_tokens.get('tokens'))
            done = stream_tokens.get('done')

    def __call__(
        self, data: Any, endpoint: Endpoint, stream: bool, **kwargs
    ) -> Union[list[SelectResponse], list[AnalyseResponse], list[CreateResponse], list[TokenizeResponse], ErrorResponse, Generator[str, None, None]]:
        if self.model_name is None:
            self.model_name = self.get_model_name()

        show_progress = kwargs.get("show_progress", False)
        num_tasks = kwargs.get("num_tasks", 1)

        if not stream:
            session_id = None
            tasks = []
            if show_progress:
                session_id = self.get_session_id()
                tasks.append(self._get_progress_task(session_id, num_tasks, endpoint))

            tasks.append(self._post(data, endpoint, session_id))
            return _safe_run_tasks(tasks)[-1]
        else:
            return self.stream_response(data)

    def get_model_name(self) -> str:
        return self._invoke_endpoint("/model")

    def is_available(self) -> bool:
        waiter = self._sm_client.get_waiter("endpoint_in_service")
        waiter.wait(EndpointName=self.endpoint_name)
        try:
            self._invoke_endpoint("/availability")
        except self._runtime_sm_client.exceptions.ModelError:
            return False
        return True

    def get_session_id(self) -> str:
        return self._invoke_endpoint("/session")
