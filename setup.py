#!/usr/bin/env python

from setuptools import setup

def parse_requirements(filename):
    with open(filename, 'r') as f:
        return f.read().splitlines()


setup(
    name="Flood Tool",
    version="1.0.1",
    description="Flood Risk Analysis Tool",
    author="ACDS project Team Thames",
    packages=["flood_tool"],
    install_requires=parse_requirements('requirements.txt'),
)
