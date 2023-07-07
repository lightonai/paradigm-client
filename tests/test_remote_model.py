from paradigm_client.response import CreateResponse, SelectResponse, TokenizeResponse
from paradigm_client.remote_model import RemoteModel, DEFAULT_BASE_ADDRESS
import pytest
import os

DEV_HOST = os.environ.get("DEV_HOST", None)


@pytest.fixture
def remote_model():
    return RemoteModel(DEV_HOST, model_name="llm-mini")


def test_should_use_default_address_when_no_url_given():
    model = RemoteModel(model_name="llm-mini")
    assert model.base_address == DEFAULT_BASE_ADDRESS


def test_should_use_default_address_when_given_explicit_none():
    model = RemoteModel(None, model_name="llm-mini")
    assert model.base_address == DEFAULT_BASE_ADDRESS


def test_should_remove_the_slashes_at_the_end_of_base_address():
    model = RemoteModel(DEFAULT_BASE_ADDRESS + "/////", model_name="llm-mini")
    assert model.base_address == DEFAULT_BASE_ADDRESS


def test_create(remote_model: RemoteModel):
    response = remote_model.create("Test completion text:")
    assert isinstance(response, CreateResponse)


def test_select(remote_model: RemoteModel):
    response = remote_model.select("Cake is ", ["yummy", "flower"])
    assert isinstance(response, SelectResponse)


def test_tokenize(remote_model: RemoteModel):
    response = remote_model.tokenize("Test completion text:")
    assert isinstance(response, TokenizeResponse)


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

