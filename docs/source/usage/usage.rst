
Usage
=====

The usage notes here provide instructions for the installation of the Python package. Use the 
Utilities tab to explore the different modules and functions of the package, and the various regional 
model examples to see the functions in action.

.. _installation:

Installation
------------

To use **eccoseas**, you can use pip to install it into you environment.

If you'd like eccoseas on your local machine, first clone the repository to your local drive:

.. code-block:: console

   git clone https://github.com/mhwood/eccoseas/tree/main


Then, use the `setup.py` file to install it into your local environment:

.. code-block:: console

   pip install eccoseas


Alternatively, you can install it directly from Github with

.. code-block:: console

   pip install git+https://github.com/mhwood/eccoseas.git

Requirements
------------

The following packages are required for **eccoseas**

.. code-block:: console

   numpy
   netCDF4
   scipy
   pyproj
   numba

In addition, if you'd like to follow along with the regional model examples in this 
documentation, then the following packages are necessary:

.. code-block:: console
   
   matplotlib
   cartopy
   cmocean
