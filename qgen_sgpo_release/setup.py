from setuptools import setup, find_packages

setup(
    name="qgen-sgpo",
    version="0.1.0",
    description="SGPO and query-generation task extension for external ROLL",
    packages=find_packages(include=["qgen_sgpo", "qgen_sgpo.*"]),
    python_requires=">=3.10",
)
