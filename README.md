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

Once the package is installed, make sure to define environment variables PARADIGM_API_KEY to your API key, e.g. by adding the following line to your .bashrc

```
export PARADIGM_API_KEY="<your api key>"
```

## Quick Start

Using paradigm-client is pretty simple, here are a code example to show you how you can use it.

```
from paradigm_client.remote_model import RemoteModel

model = RemoteModel(model_name="llm-mini")

print(model.create("Hello, I am").completions[0].output_text)
```

## Logging a feedback into Paradigm

After using the Create endpoint of Paradigm, you can log a feedback about it.

Feedback data is expected to be in a dictionary format with a key being one of "flag", "value", "tag" and "comment".
 - `flag`: used for **boolean** feedbacks
 - `value`: used for **float** feedbacks
 - `tag`: used for **short text** feedbacks
 - `comment`: used for **free text** feedbacks

Here are the different steps to log a feedback:
1. If the feedback type you want to use does not exist on Paradigm yet, go to the admin interface of Paradigm and create it. 
2. Get the `rating_id` of the feedback type you want to use.
3. Instantiate a `RemoteModel` object.
4. Get the `completion_id` from the response of a `RemoteModel.create()` call.
5. Call the `log_feedback()` method of your `RemoteModel` object with the `rating_id`, the `completion_id` related to your feedback and your feedback data.

> **Important note**: The API Key used to generate the `completion_id` and to send the feedback must authorize the logging of its requests from the admin interface of Paradigm.

## Access to LightOn Paradigm

Learn more about Paradigm: https://www.lighton.ai/fr/paradigm.
For a list of use cases: https://www.lighton.ai/fr/blog/ai-use-case-5.

## Known Issues

If you find that a `RemoteModel` instantiation and subsequent completions are unusually slow, it may be that your network does not support IPv6. Try disabling IPv6 and see if that helps.
