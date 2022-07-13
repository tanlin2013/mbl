Data Discovery
==============

.. autosummary::
    :toctree: _autosummary

    mbl.experiment.algorithm
    mbl.experiment.random_heisenberg

|
Random Heisenberg model
-----------------------
The Hamiltonian

.. math::

    H = \sum_{i=0}^{N-1} \mathbf{S}_i\cdot\mathbf{S}_{i+1} + h_i S_i^z

with :math:`h_{i} \in [-h, h)` are sampled uniformly.
