import os
from unittest import mock

import pytest

from paradigm_client.communicator import SagemakerCommunicator
from paradigm_client.remote_model import DEFAULT_BASE_ADDRESS, RemoteModel
from paradigm_client.request import CreateRequest
from paradigm_client.response import (CreateResponse, SelectResponse,
                                      TokenizeResponse)

DEV_HOST = os.environ.get("DEV_HOST", None)


@pytest.fixture
def remote_model():
    return RemoteModel(DEV_HOST, model_name="alfred-40b-0723")


def test_should_use_default_address_when_no_url_given():
    model = RemoteModel(model_name="alfred-40b-0723")
    assert model.base_address == DEFAULT_BASE_ADDRESS


def test_should_use_default_address_when_given_explicit_none():
    model = RemoteModel(None, model_name="alfred-40b-0723")
    assert model.base_address == DEFAULT_BASE_ADDRESS


def test_should_remove_the_slashes_at_the_end_of_base_address():
    model = RemoteModel(DEFAULT_BASE_ADDRESS + "/////", model_name="alfred-40b-0723")
    assert model.base_address == DEFAULT_BASE_ADDRESS


def test_should_raise_error_when_no_api_key_available():
    # Locally removing the PARADIGM_API_KEY variable from the environment
    names_to_remove = {"PARADIGM_API_KEY"}
    modified_environ = {k: v for k, v in os.environ.items() if k not in names_to_remove}
    with mock.patch.dict(os.environ, modified_environ, clear=True):
        with pytest.raises(AssertionError, match="You must provide an API key through the PARADIGM_API_KEY environment variable or the api_key parameter"):
            model = RemoteModel(model_name="alfred-40b-0723")


@pytest.mark.skipif(
    "SAGEMAKER_ENDPOINT" not in os.environ or
    "AWS_ACCESS_KEY_ID" not in os.environ or
    "AWS_SECRET_ACCESS_KEY" not in os.environ or
    "AWS_DEFAULT_REGION" not in os.environ,
    reason="Missing environmental variable for this test; Check that 'AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'AWS_DEFAULT_REGION' and 'SAGEMAKER_ENDPOINT' variables are defined."
)
def test_should_not_check_api_key_when_sagemaker_communicator_used():
    # Locally removing the PARADIGM_API_KEY variable from the environment
    names_to_remove = {"PARADIGM_API_KEY"}
    modified_environ = {k: v for k, v in os.environ.items() if k not in names_to_remove}
    with mock.patch.dict(os.environ, modified_environ, clear=True):
        comm = SagemakerCommunicator(endpoint_name=os.environ.get("SAGEMAKER_ENDPOINT"))
        model = RemoteModel(comm=comm)
        assert model.is_api_key_none()


def test_create(remote_model: RemoteModel):
    response = remote_model.create("Test completion text:")
    assert isinstance(response, CreateResponse)


def test_select(remote_model: RemoteModel):
    response = remote_model.select("Cake is ", ["yummy", "flower"])
    assert isinstance(response, SelectResponse)


def test_tokenize(remote_model: RemoteModel):
    response = remote_model.tokenize("Test completion text:")
    assert isinstance(response, TokenizeResponse)


# Related issue: https://github.com/lightonai/paradigm-client/issues/7
def test_create_from_objects(remote_model: RemoteModel):
    requests = CreateRequest(text="Hello I am")
    response = remote_model.create_from_objects(requests)
    assert isinstance(response[0], CreateResponse)


def test_can_log_a_boolean_feedback(
        remote_model: RemoteModel
):
    create_response = remote_model.create("Test completion text:")
    completion_id = create_response.completions[0].completion_id
    response = remote_model.log_feedback(rating_id=3, completion_id=completion_id, data={"flag": True})
    assert response.status_code < 400


def test_can_log_a_text_feedback(
        remote_model: RemoteModel
):
    create_response = remote_model.create("Test completion text:")
    completion_id = create_response.completions[0].completion_id
    response = remote_model.log_feedback(rating_id=4, completion_id=completion_id, data={"comment": "Test comment"})
    assert response.status_code < 400

