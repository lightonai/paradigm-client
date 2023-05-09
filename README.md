# paradigm-client
Python client for LightOn Paradigm LLMs

## Installation

The following command will pull and install the latest commit from this repository, along with its Python dependencies:
```
pip install -U git+https://github.com/lightonai/paradigm-client.git
```

Once the package is installed, make sure to define environment variables PARADIGM_API_KEY and HOST to your API key, e.g. by adding the following lines to your .bashrc

```
export PARADIGM_API_KEY="<your api key>"
```

## Quick Start

```python
from paradigm_client.remote_model import RemoteModel
import os

host=<HOST_IP>

model = RemoteModel(host, model_name="llm-mini")

print(model.create("Hello, I am").completions[0].output_text)
```
