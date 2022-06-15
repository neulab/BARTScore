#!/usr/bin/env python

from distutils.core import setup

setup(
    name="bart_score",
    version="0.1.0",
    description="BARTScore: Evaluating Generated Text as Text Generation",
    author="Weizhe Yuan",
    url="https://github.com/neulab/BARTScore",
    install_requires=[
        "torch>=1.6.0",
        "transformers>=4.6.1",
    ],
)
