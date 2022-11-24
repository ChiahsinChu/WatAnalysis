# WatAnalysis

## Introduction

This is a package for (parallel) analysis of water structures/dynamics at **metal/water interfaces**.
Some codes here are developed based on [`MD Analysis`](https://userguide.mdanalysis.org/2.0.0-dev0/index.html) or [`pmda`](https://www.mdanalysis.org/pmda/) package.

This package includes:

- [x] (region-specified) hydrogen bonds analysis

**TO DO LIST**

- [ ] code for trajectory wrapping
- [x] update region-specific function (donor/acceptor)
- [x] add z-coord of HB into results
- [x] add graph-generation module (undirected)
  - [ ] add attributes of edge(bond length)/node(z w.r.t surface)

## Installation

```bash
python setpy.py install
# if you want to modify the code later, use the developer mode
#python setup.py develop
```

## User Guide

1. `waterstructure`: water structure analysis

2. `waterdynamics`: water dynamics analysis
This module is built on the basis of `MDAnalysis.analysis.waterdynamics`. I adapt the analysis class to the interfacial systems, which is implemented in conjunction with a [modified MDAnalysis package](https://github.com/ChiahsinChu/m_MDAnalysis) (private package yet). 

- instantaneous temperature of selected atoms (AtomGroup in MDA)

## Developer Guide
