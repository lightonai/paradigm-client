import os
from paradigm_client.remote_model import RemoteModel

assert os.getenv("PARADIGM_API_KEY") is not None

model = RemoteModel(model_name="llm-mini")

response = model.stream_create("What is the result of 2+2?")

for line in response:
    print(line)
