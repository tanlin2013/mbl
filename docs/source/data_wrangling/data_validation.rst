Data Validation
===============

.. autosummary::
    :toctree: _autosummary

    mbl.schema.RandomHeisenbergEDSchema
    mbl.schema.RandomHeisenbergTSDRGSchema
    mbl.schema.RandomHeisenbergFoldingTSDRGSchema

Data extracted from the experiment is in the type
of :class:`~pandas.DataFrame`.
We use `pandera <https://pandera.readthedocs.io/en/stable/>`_
to check and validate the data in the dataframe.
