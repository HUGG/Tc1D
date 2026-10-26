T\ :sub:`c`\ 1D documentation
=============================

.. figure:: gif/tc1d-0.5.0.gif
   :width: 720
   :alt: TC1D running in a terminal window

   An example Tc1D model run from the command-line interface.

T\ :sub:`c`\ 1D is a one-dimensional thermal and thermochronometer age prediction model that can simulate the effects of various geodynamic and geomorphic processes on thermochronometer ages.
It supports forward and inverse modeling of apatite and zircon (U-Th)/He and fission-track ages.

Check out the :doc:`getting started page <getting-started>` and the :doc:`user guide <user-guide>` for further information, including
details about :ref:`installation <installation>`.

What can T\ :sub:`c`\ 1D do?
----------------------------

T\ :sub:`c`\ 1D can be used to

- Explore the effects of changes in rates of burial or exhumation on apatite and zircon (U-Th)/He and fission-track thermochronometer ages
- Simulate how thermochronometer ages change in response to crustal and lithospheric thermal processes such as emplacement of magmatic intrusions or delamination of the mantle lithosphere
- Derive estimates of rates of burial and exhumation from thermochronometer datasets using data inversion
- Produce plots of measured and predicted ages for various exhumation/burial histories

More details about how T\ :sub:`c`\ 1D works and some example applications are presented in a `technical note about the code in Geochronology <https://doi.org/10.5194/gchron-8-627-2026>`_.

Citation
--------

Use of T\ :sub:`c`\ 1D can be acknowledged by citing `the relevant version of Tc1D <https://doi.org/10.5281/zenodo.7124271>`_ using the DOI from Zenodo that always points to the latest version (`10.5281/zenodo.7124271 <https://doi.org/10.5281/zenodo.7124271>`_).
For example, the citation for version 0.5.0 is

   Whipp, D. M., Gérard, B., Laaksonen, S., & Dönges, S. (2026). Tc1D - A 1D thermal and thermochronometer age prediction model (Version 0.5.0) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.23512117

We also encourage you to cite the related T\ :sub:`c`\ 1D technical note, if appropriate.

   Whipp, D. M., Gérard, B., Laaksonen, S., and Kellett, D. A. (2026). Technical note: Tc1D – a 1D thermal and thermochronometer age prediction model. *Geochronology*, 8, 627–643. https://doi.org/10.5194/gchron-8-627-2026

Acknowledgements
----------------

We acknowledge Saska Dönges from the `University of Helsinki Research Software Engineering (UH-RSE) <https://wiki.helsinki.fi/xwiki/bin/view/Research%20Software%20Engineering%20group/>`_ for scientific computing support.

License
-------

T\ :sub:`c`\ 1D is licensed under the GNU General Public License version 3: `Tc1D software license <https://github.com/HUGG/Tc1D/blob/main/LICENSE>_.

Contents
--------

.. toctree::
   :caption: Using Tc1D
   :maxdepth: 2

   getting-started
   user-guide
   example-plots

.. toctree::
   :caption: Reference
   :maxdepth: 2

   reference
