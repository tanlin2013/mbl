# MBL: many-body localization
![build](https://github.com/tanlin2013/mbl/actions/workflows/build.yml/badge.svg)
![test](https://github.com/tanlin2013/mbl/actions/workflows/test.yml/badge.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[Dashboard](https://streamlit-mbl.herokuapp.com/) |
[Documentation]()

MBL is a research program aims at studying the delocalization transitions with numerical methods,
currently with exact diagonalization (ED),
and the support of matrix product state (MPS) is under planning.  

Models
-------
* Random-field Heisenberg model

Getting Started
-------

1. Installation
   
    * using Docker (required to be built from source):
        ```
        make build & make run
        ```
    * using pip:
        ```
        pip install git+https://github.com/tanlin2013/mbl@main
        ```

2. Run a sampler
   
    with parallel runs supported by [ray](https://ray.io/).
    
    ```
    python scripts/sampler.py
    ```

License
-------
© Contributors, 2021. Licensed under a [MIT](https://github.com/tanlin2013/mbl/master/LICENSE) license.

Reference
-------
[D. A. Abanin, E. Altman, I. Bloch and M. Serbyn, Colloquium: Many-body localization, thermalization, and entanglement. Rev. Mod. Phys. 91, 021001 (2019).](https://journals.aps.org/rmp/abstract/10.1103/RevModPhys.91.021001)