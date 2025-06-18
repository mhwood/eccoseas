Welcome to the ECCO Seas documentation!
=======================================

*Tools to generate regional ocean model simulations using ECCO global output.*

Author: Mike Wood

**eccoseas** is a Python library containing tools for creating regional
ocean model simulations using the MIT General Circulation. These configurations use output
from the ECCO Consortium's global ocean state estimates including ECCOv4, ECCOv5 Alpha,
and the biogeochmistry-flavored ECCO-Darwin.

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

The Usage section has information on the :ref:`installation` of the package.

.. note::

   This project is under active development.

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   :numbered: 5

   usage/usage
   utilities/overview
   ca_regional_model/overview
   north_atlantic_regional_model/overview
   alaskan_north_slope_regional_model/overview
