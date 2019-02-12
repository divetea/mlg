#! /usr/bin/python3
"""This is the installation script for my package."""

import setuptools


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mlg-pkg-divetea",
    version="0.0.1",
    author="Test Niklas",
    author_email="niklas@test.com",
    description="A small MLG decoding example package to test packaging.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/divetea/mlg_package",
    package_dir={'': 'src'},
    packages=setuptools.find_packages(
        where='src',
        exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        str("License :: OSI Approved :: GNU General Public License v3 or later"
            " (GPLv3+)"),
        "Operating System :: OS Independent",
    ],
)
