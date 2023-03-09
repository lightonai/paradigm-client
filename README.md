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
export HOST="<your host adress>"
```

## Quick Start

```python
from paradigm_client.remote_model import RemoteModel
import os

host = os.environ.get("HOST", None)
assert host is not None, "{HOST} env var is not properly set. Run `export HOST=<value>` in your shell or add it to your `.bashrc`"
api_key = os.environ.get("PARADIGM_API_KEY", None)
assert api_key is not None, "{PARADIGM_API_KEY} env var is not properly set. Run `export PARADIGM_API_KEY=<value>` in your shell or add it to your `.bashrc`"

model = RemoteModel(
    f"{host}", 
    headers={
        "Content-Type": "application/json", 
        "Accept": "application/json",
        "X-API-KEY": api_key,
        "X-Model": "llm-mini",
    },
    timeout_s=120
)

print(model.create("Hello, I am happy because", echo=True).completions[0].output_text)
```
