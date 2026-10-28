Getting started
===============

.. _installation:

Installation (for most users)
-----------------------------

You can install T\ :sub:`c`\ 1D in your Python environment using ``pip``.

.. code-block:: console

   pip install tc1d

Thermochronometer ages are calculated using `Tcages <https://github.com/HUGG/tcages>`_, which is installed automatically with T\ :sub:`c`\ 1D.

.. warning::

   If you choose to install T\ :sub:`c`\ 1D versions older than 0.5.0, you will also need to install the thermochronometer age prediction programs available in the `v0.2 release of Tc_core <https://github.com/HUGG/tcages/releases/tag/v0.2>`_ in order for T\ :sub:`c`\ 1D to work properly. Installing the latest version of T\ :sub:`c`\ 1D is strongly recommended.

Installation (for parallel MCMC inversions)
-------------------------------------------

You can include support for parallel Markov-Chain Monte Carlo (MCMC) inversions in T\ :sub:`c`\ 1D by installing using the following options in ``pip``.

.. code-block:: console

   pip install "tc1d[mpi]"

.. note::

   This should work in Linux and macOS, but you may need to install OpenMPI or MPICH for MPI support. In Windows, MPI is available via ms-mpi or IntelMPI, but you will need to install that software manually for MPI support.

Running a basic model
---------------------

After installation, an example model with 10 km of exhumation and default values for all other parameters can be run from the command line as follows:

.. code-block:: console

   tc1d-cli --ero-option1 10.0

.. note::

   It is also possible to define parameters and run T\ :sub:`c`\ 1D using Python function calls or an input file. These cases are described in the user guide.

Model options
-------------

A full list of options that can be used with T\ :sub:`c`\ 1D can be found by running the code with the ``--help`` flag (or no specified flags):

.. code-block:: console

   tc1d-cli --help

This will return a usage statement and list of flags the code accepts.

Example notebooks
-----------------

Additional examples of how to run T\ :sub:`c`\ 1D can be found in a set of Jupyter notebooks, which you can try online using the **launch binder** button on the `Tc1D GitHub page <https://github.com/HUGG/Tc1D>`_, for example.
To get a local copy of all of the notebooks and their data files, run

.. code-block:: console

   tc1d-notebooks

This downloads the notebooks matching your installed version of T\ :sub:`c`\ 1D into a new ``tc1d-notebooks`` directory. Use ``tc1d-notebooks --help`` to see how to choose a different location or version.
The notebooks are also available as a ZIP file attached to each `Tc1D release <https://github.com/HUGG/Tc1D/releases>`_ and the notebooks for the latest release can be `downloaded directly <https://github.com/HUGG/Tc1D/releases/latest/download/tc1d-notebooks.zip>`_.

To run the notebooks locally you will also need JupyterLab, which can be installed along with T\ :sub:`c`\ 1D using

.. code-block:: console

   pip install "tc1d[notebook]"

