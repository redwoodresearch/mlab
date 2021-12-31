#!/usr/bin/env python

from setuptools import setup

# from distutils.core import setup

setup(
    name="ml_bootcamp",
    version="0.1",
    description="Machine Learning for Alignment Bootcamp",
    author="Redwood Research",
    author_email="tao@rdwrs.com",
    # install_requires=[
    #     "torch",
    #     "torchtyping",
    #     "einops",
    #     "torchvision",
    #     "torchtext",
    #     "numpy",
    #     "transformers",
    #     "sentencepiece",
    #     "unidecode",
    #     "jupyter_kernel_gateway",
    #     "ipykernel",
    #     "pytest",
    #     "matplotlib",
    #     "plotly",
    #     "sklearn",
    #     "black",
    #     "comet_ml",
    #     "git+https://github.com/google/gin-config.git@e518c4ec7755a3e5da973e894ab23cc80c6665ed#egg=gin-config",
    # ],
    packages=["days", "mlab_tests"],
)
