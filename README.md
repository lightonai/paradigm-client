# paradigm-client

Python client for LightOn Paradigm Large language Model.

## Installation

You can install this package from PyPi with:
```
pip install paradigm-client
```

Or from source:
```
pip install -U git+https://github.com/lightonai/paradigm-client.git
```

Once the package is installed, make sure to define environment variables PARADIGM_API_KEY and HOST to your API key, e.g. by adding the following lines to your .bashrc

```
export PARADIGM_API_KEY="<your api key>"
export HOST="<your host IP>"
```

## Quick Start

Using paradigm-client is pretty simple, here are a code example to show you how you can use it.

```
from paradigm_client.remote_model import RemoteModel
import os

host = os.environ.get('HOST')

model = RemoteModel(host, model_name="llm-mini")

print(model.create("Hello, I am").completions[0].output_text)
```

## Access to LightOn Paradigm

Try our Paradigm LLM at https://www.lighton.ai/fr/paradigm.
See some Paradigm user cases at https://www.lighton.ai/fr/blog/ai-use-case-5.
