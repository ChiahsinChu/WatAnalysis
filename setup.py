"""
Setup script for `zjxpack`
"""

from setuptools import setup, find_packages

setup(
    name = "zjxpack",
    version = "0.1",
    # include all packages in src
    packages = find_packages(),  
    # exclude by
    #find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"])
    # all packages required for this package is in src
    #package_dir = {'':'zjxpack'},   
    include_package_data = True,
    # exclude all update.sh
    exclude_package_data = { '': ['update.sh'] }
)

