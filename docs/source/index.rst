Welcome to the ECCO Seas documentation!
=======================================

*Tools to generate regional ocean model simulations using ECCO global output.*

Author: Mike Wood

**eccoseas** is a Python library containing tools for creating files necessary for regional
ocean model simulations using the `MIT General Circulation Model <https://github.com/MITgcm/MITgcm>`_. These configurations use output
from the `ECCO Consortium <https://ecco-group.org/>`_'s global ocean state estimates including ECCOv4, ECCOv5 Alpha,
and the biogeochmistry-flavored ECCO-Darwin.

.. note::

   This project is under active development.

Motivation
----------
When creating regional ocean models, the initial conditions, boundary conditions, and 
external forcing conditions are important considerations for the model run. There are lots 
of ways these conditions can be created including using climatologies, observations,
and existing coarse resolution model. This package provides a set of tools to read in
data from ECCO's global ocean simulations and process them to generate models in a given
regional domain.



Getting Started
---------------

The Usage section has information on the :ref:`installation` of the Python package.

To see examples 
of the package in action, take a look at one of the regional model examples provided. Each model configuration showcases different
approaches to constructing model files using the tools in the **eccoseas** package. The overview page of each 
regional model provides a list of the modules and functions used from the package.

Documentation Contents 
----------------------

.. toctree::
   :maxdepth: 2
   :numbered: 5

   usage/usage
   utilities/overview
   ca_regional_model/overview
   north_atlantic_regional_model/overview
   alaskan_north_slope_regional_model/overview
