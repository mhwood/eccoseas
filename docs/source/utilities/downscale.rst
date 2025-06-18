downscale
=========

The modules in downscale collectively form a toolkit for preparing and interpolatingmodel data, 
supporting tasks such as regridding, downscaling, and ensuring hydrological consistency. Together,
these modules enable the transformation of low-resolution model output into physically
consistent inputs for higher-resolution simulations.


bathymetry
^^^^^^^^^^
This module provides functionality for identifying and isolating hydrologically connected wet 
regions within a 2D grid representation of bathymetry data. It defines two functions: 
`generate_connected_mask`, which performs a flood-fill style operation starting from a 
specified wet cell to mark all adjacent and connected wet cells (based on 4-directional connectivity), 
and `fill_unconnected_model_regions`, which uses this connectivity mask to modify a bathymetry grid 
by filling (i.e., setting to zero) any wet regions that are not connected to a central 
reference point. The script is useful for ensuring consistency in hydrological modeling 
by filtering out isolated or disconnected wet regions that may arise due to interpolation artifacts.

hFac
^^^^
This module provides Python implementations of routines originally from the MITgcm's ini_masks_etc.F 
source file, used to compute partial cell factors for representing topography and bathymetry in ocean 
models. It defines three key functions: `create_hFacC_grid`, `create_hFacS_grid`, and `create_hFacW_grid`, 
which calculate vertical (center), south face, and west face cell fractions (hFacC, hFacS, and hFacW, 
respectively) based on input bathymetry and vertical grid resolution (`delR``). Each function emulates 
the MITgcm logic for applying geometric constraints, ensuring that cells below the ocean surface are 
appropriately represented, with provisions for minimum cell thickness (`hFacMinDr``) and minimum cell 
fraction (`hFacMin`). These masks are crucial for interpolating from coarse resolution grids to fine 
resolution grids in which the wet and dry cells may differ. Note that these functions may not reproduce
the exact hFac fields as MITgcm due to rounding difference between Python and FORTRAN.

horizonal
^^^^^^^^^
This module provides a toolkit for spatial interpolation and value propagation across 
geophysical grids, particularly focused on handling wet (oceanic or fluid) regions. It includes legacy 
and optimized implementations for horizontally spreading variable values into wet cells using 
nearest-neighbor approaches. The primary use case is in downscaling coarse-resolution model data 
onto finer-resolution grids, ensuring that missing or zero-valued cells in wet regions are filled 
sensibly. The functions assume hydrostatic equillibirum and preferentially spread horizonally where possible.
If wet values cannot be filled horizonally (i.e. they are bounded by bathymetry contours), then they are filled
vertically. Key functions leverage both NumPy and PyTorch to balance compatibility and performance, 
with options for masking and interpolation constraints. The module is designed for interpolating
oceanographic or atmospheric model inputs in multi-dimensional (2D/3D) formats.

interpolation_grid
^^^^^^^^^^^^^^^^^^
This module provides a suite of functions for constructing and exporting interpolation grids that map data 
from a source model grid (Level 0) to a target domain grid (Level 1), with a focus on ocean modeling 
applications such as ECCO (Estimating the Circulation and Climate of the Ocean). The main function, 
`create_interpolation_grid`, performs 2D spatial interpolation of a variable across vertical layers, 
followed by horizontal and vertical spreading to fill gaps in the target grid where data is missing. 
If needed, it applies a final nearest-neighbor interpolation to fill remaining holes. The wrapper 
function `create_interpolation_grids` generates interpolation mappings for multiple staggered grid 
components—typically C (cell center), S (south face), and W (west face). The resulting interpolation 
type and source-tracing metadata are written to a NetCDF file using `write_interpolation_grid_to_nc`, 
enabling reproducibility and detailed analysis of how each grid cell's value was determined. 

vertical
^^^^^^^^
This script provides a suite of Python functions designed for vertical interpolation and data 
manipulation on gridded oceanographic datasets, typically used in numerical ocean models. The main 
function, `interpolate_var_grid_faces_to_new_depth_levels`, interpolates 3D or 4D spatial fields 
(e.g., temperature or salinity) from an original set of vertical levels to a new target set, 
using linear interpolation while carefully handling missing or masked data (via a "wet grid"). 
A similar function, `interpolate_var_points_timeseries_to_new_depth_levels`, handles time series 
data at discrete point locations, interpolating them vertically in the same manner. Both functions 
ensure continuity of values through strategies such as downward filling and surface extrapolation to
mitigate gaps due to missing data. The final function, `spread_var_vertically_in_wet_grid`, vertically 
extends values from a given level downward into adjacent wet cells, applying a fixed mean vertical
difference, and is used when filling in uninitialized or partially masked regions. These tools are 
useful for regridding between different vertical coordinate systems in downscaling workflows.









