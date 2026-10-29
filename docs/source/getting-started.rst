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

Inversions using the Markov-Chain Monte Carlo (MCMC) approach in T\ :sub:`c`\ 1D can be run more efficiently by running them in parallel on multiple cores or processors.
This is done using MPI, and the best approach for how to speed inversions up will depend on where you are running them.

Parallel inversions on your own computer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you are running on your own computer and do not have MPI installed already, the easiest option is to install using ``pip``:

.. code-block:: console

   pip install "tc1d[mpi]"

.. note::

   This should work in Linux and macOS, but it is possible you may need to install OpenMPI or MPICH for MPI support. In Windows, MPI is available via ms-mpi or IntelMPI and you will need to install that software manually for MPI support.

Parallel inversions on a computer cluster
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you are planning to use T\ :sub:`c`\ 1D for parallel inversions on a computer cluster (or already have MPI on your computer), it is better to install using ``pip`` after first loading the system's MPI module (assuming the cluster uses modules):

.. code-block:: console

   module load openmpi   # Or whatever MPI the system is using

After this, T\ :sub:`c`\ 1D and the needed ``schwimmbad`` library can be installed using ``pip``, as well as ``mpi4py``.

.. code-block:: console

   pip install tc1d schwimmbad
   pip install --no-cache-dir --no-binary mpi4py mpi4py

The key here is that ``mpi4py`` will be built from source using the system MPI library, which should ensure efficient performance using the tuned MPI install on the cluster.
Installing using ``pip install "tc1d[mpi]"`` will default to using the MPI provided in the ``openmpi`` dependency package, which could conflict with the cluster MPI or be slower.

Running a parallel inversion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The process for running a parallel inversion will vary depending on the system you are using.
A typical example for running on your own computer using 4 MPI processes might look like this:

.. code-block:: console

   mpirun -n 4 tc1d-cli --run-type mcmc ...

.. hint::

   On a computer cluster a similar command might be embedded in an ``sbatch`` script file.

.. note::

   One process is dedicated to coordinate the inversion, so ``-n 4`` will result in having 3 worker processes.

.. attention::

   Only MCMC inversions (``run_type: mcmc``) can be run in parallel.
   Do not use ``mpirun`` for forward models, batch runs, or NA inversions as each MPI process will run its own full copy of the model (!).

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

