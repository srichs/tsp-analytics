"""Package metadata for tsp-analytics."""

from setuptools import find_packages, setup


setup(
    packages=find_packages(where="src", exclude=("tests",)),
    package_dir={"": "src"},
)
