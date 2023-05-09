# paradigm-client

Python client for LightOn Paradigm LLMs

## Local installation

The following command will pull and install the latest commit from this repository, along with its Python dependencies:
```
pip install -U git+https://github.com/lightonai/paradigm-client.git
```

Once the package is installed, make sure to define environment variables PARADIGM_API_KEY and HOST to your API key, e.g. by adding the following lines to your .bashrc

```
export PARADIGM_API_KEY="<your api key>"
export HOST="<your host IP>"
```

## Remote installation

You can just run `pip install paradigm-client` to get the latest version of the package from the https://pypi.org/ repository.

## Quick Start

See `tests/example.py` code example to know how you can use the library.

## Deployment

To deploy a new version of the `paradigm-client` package, you should push your local commits on the `main` branch with a new tag that started with `v` (e.g. `v1.0`),
then the GitHub workflow allows to automatically install dependencies, run tests, build the package and publish it to PyPI (see `.github/workflows/build-test-deploy.yaml` file for more details).


