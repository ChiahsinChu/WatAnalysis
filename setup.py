"""
Setup script for WatDynamics
"""

from setuptools import setup, find_packages

setup(
    name="watdyn",
    version="0.0",
    # include all packages in src
    packages=find_packages(),
    include_package_data=True)
