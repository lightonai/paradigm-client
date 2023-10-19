import json
import time
from datetime import datetime

from paradigm_client.communicator import SagemakerCommunicator
from paradigm_client.remote_model import RemoteModel

sagemaker_endpoint_name = "<endpoint-name>"
comm = SagemakerCommunicator(endpoint_name=sagemaker_endpoint_name)
model = RemoteModel(comm=comm)

def current_date():
    date = datetime.fromtimestamp(time.time())
    return date.strftime("%B %d, %Y")

messages = [
    {
        "role": "system",
        "content": f"You are Alfred, a helpful assistant trained by LightOn. Knowledge cutoff: November 2022. Current date: {current_date()}",
    },
    {
        "role": "user",
        "content": "What is the weather today?",
    },
]
response = model.chat(messages=messages, stop_sequences=["\n\n"], n_tokens=100, temperature=0.)
print(response.completions[0].output_text)
