#!/usr/bin/env python

from distutils.core import setup

setup(
    name="BARTScore",
    version="0.1.0",
    description="BARTScore: Evaluating Generated Text as Text Generation",
    author="Weizhe Yuan",
    url="https://github.com/neulab/BARTScore",
    packages=[
        "torch>=1.0",
        "transformers>=4.6.1",
    ],
)
