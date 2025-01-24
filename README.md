# WatAnalysis

## Introduction

This is a package for (parallel) analysis of water structures/dynamics at **metal/water interfaces**.
Some codes here are developed based on [`MDAnalysis`](https://userguide.mdanalysis.org/2.0.0-dev0/index.html) or [`pmda`](https://www.mdanalysis.org/pmda/) package.

## Installation

Run the following commands in the repository root:

```bash
git clone https://github.com/ChiahsinChu/WatAnalysis.git
cd WatAnalysis
pip install .
```

or `pip install -e .` if you want the installation to update as you develop the code.

The analysis classes are adapted for the interfacial systems, which should be used in conjunction with a modified MDAnalysis package:

```bash
git clone https://github.com/ChiahsinChu/mdanalysis.git -b devel-relprop
cd mdanalysis 
pip install --upgrade package/
cd ..
```

## User Guide

1. `waterstructure`: water structure analysis

2. `waterdynamics`: water dynamics analysis

   This module is built on the basis of `MDAnalysis.analysis.waterdynamics`.

3. `temp`: instantaneous temperature of selected atoms

## Developer Guide
