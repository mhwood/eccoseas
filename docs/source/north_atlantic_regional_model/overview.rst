North Atlantic Regional Model
*****************************

Overview
========
This is a description of the North Atlantic model.

The domain of the regional model is shown in the following plot:

.. image:: ../images/north_atlantic_model_bathymetry.png
  :width: 400
  :alt: Bathymetry of the North Atlantic regional model

The construction of this model showcases additional components of the eccoseas
package in the downscale and ecco modules, specifically those pertaining to sea ice.

The functions used in this demo include the tasks in the following list:

1. Reading ECCO Version 5 seaice output fields
5. Reading and writing sea ice fields

For this example, the following list of files are required from the ECCO Version 5 Alpha State estimate. These
files are available on the `ECCO drive <https://ecco.jpl.nasa.gov/drive/>`_.

.. list-table:: ECCO files required to construct the California regional model
   :widths: 50 50
   :header-rows: 1

   * - Variable
     - File(s)
   * - Potential Temperature
     - THETA_1992.nc to THETA_2017.nc
   * - Salinity
     - SALT_1992.nc to SALT_2017.nc
   * - u-Component of Velocity
     - UVELMASS_2017.nc to UVELMASS_2017.nc
   * - v-Component of Velocity
     - VVELMASS_2007.nc to VVELMASS_2017.nc
   * - Sea Surface Height Anomaly
     - ETAN_1992.nc
   * - Lowngwave Downwelling Radiation
     - EIG_dlw_plus_ECCO_v4r1_ctrl (1992-2017)
   * - Shortwave Downwelling Radiation
     - EIG_dsw_plus_ECCO_v4r1_ctrl (1992-2017)
   * - u-Component of Wind
     - EIG_u10m (1992-2017)
   * - v-Component of Wind
     - EIG_v10m (1992-2017)
   * - Precipitation
     - EIG_rain_plus_ECCO_v4r1_ctrl (1992-2017)
   * - Air Temperature
     - EIG_tmp2m_degC_plus_ECCO_v4r1_ctrl (1992-2017)
   * - Specific Humidity
     - EIG_spfh2m_plus_ECCO_v4r1_ctrl (1992-2017)  
   * - Grid components for each tile
     - GRID.0001.nc through GRID.0013.nc


Initial Conditions
==================

.. toctree::

   initial_conditions


Boundary Conditions
===================

.. toctree::

   boundary_conditions
