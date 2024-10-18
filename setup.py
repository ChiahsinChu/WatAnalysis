# SPDX-License-Identifier: LGPL-3.0-or-later
"""
Setup script for WatAnalysis
"""

from setuptools import find_packages, setup

setup(
    name="WatAnalysis",
    version="1.0",
    # include all packages in src
    packages=find_packages(),
    include_package_data=True,
)
