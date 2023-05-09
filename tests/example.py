from paradigm_client.remote_model import RemoteModel
import os

host = os.environ.get('HOST')

model = RemoteModel(host, model_name="llm-mini")

print(model.create("Hello, I am").completions[0].output_text)
