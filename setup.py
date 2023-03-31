import pathlib

from setuptools import setup, find_packages


here = pathlib.Path(__file__).parent.resolve()

name = "paradigm_client"
description = "Python client for LightOn Paradigm"
# Get the long description from the README file
long_description = (here / "README.md").read_text(encoding="utf-8")
author = "LightOn AI"
author_email = "support@lighton.ai"
classifiers = [
    # Trove classifiers
    # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Development Status :: 4 - Beta",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
]


setup(
    name=name,
    description=description,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=author,
    author_email=author_email,
    url="https://github.com/lightonai/paradigm-client",
    use_scm_version=True,
    setup_requires=["setuptools_scm"],
    install_requires=["aiohttp==3.8.4", "pydantic==1.10.2", "requests==2.28.2", "tqdm==4.64.1"],
    packages=find_packages(exclude=["examples", "tests"]),
    keywords=["NLP", "API", "AI", "LLM"],
    classifiers=classifiers,
)
