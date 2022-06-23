import setuptools

setuptools.setup(
    name="bart_score",
    version="0.1.0",
    description="BARTScore: Evaluating Generated Text as Text Generation",
    author="John Giorgi",
    url="https://github.com/allenai/BARTScore",
    python_requires=">=3.6",
    packages=setuptools.find_packages(),
    install_requires=[
        "torch>=1.6.0",
        "transformers>=4.6.1",
        "pytorch_pretrained_bert>=0.6.2",
        "jsonlines>=3.0.0",
        "mosestokenizer>=1.2.1",
        "pyrouge>=0.1.3",
        "bert-score>=0.3.11",
    ],
)
