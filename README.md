# WatAnalysis

## Introduction

This is a package for (parallel) analysis of water structures/dynamics at **metal/water interfaces**.
Some codes here are developed based on [`MDAnalysis`](https://userguide.mdanalysis.org/2.0.0-dev0/index.html) or [`pmda`](https://www.mdanalysis.org/pmda/) package.

## Installation

```bash
python setpy.py install
# if you want to modify the code later, use the developer mode
#python setup.py develop
```

The analysis classes are adapted for the interfacial systems, which should be used in conjunction with a modified MDAnalysis package:

```bash
git clone https://github.com/ChiahsinChu/mdanalysis.git -b jxzhu_dev
cd mdanalysis && pip install package/
cd ..
```

## User Guide

1. `waterstructure`: water structure analysis

2. `waterdynamics`: water dynamics analysis

   This module is built on the basis of `MDAnalysis.analysis.waterdynamics`.

3. `temp`: instantaneous temperature of selected atoms

## Developer Guide
