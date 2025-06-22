import os
import numpy as np
from scipy.interpolate import griddata, interp1d
import netCDF4 as nc4

def count_spreading_rows_and_cols_in_wet_grid(var_grid, source_row_grid, source_col_grid, source_level_grid, wet_grid):

    rows = np.arange(np.shape(var_grid)[0])
    cols = np.arange(np.shape(var_grid)[1])
    Cols,Rows = np.meshgrid(cols,rows)

    is_remaining = np.logical_and(var_grid==0,wet_grid==1)
    n_remaining = np.sum(is_remaining)
    continue_iter = True
    for i in range(n_remaining):
        if continue_iter:
            Wet_Rows = Rows[wet_grid == 1]
            Wet_Cols = Cols[wet_grid == 1]
            Wet_Source_Rows = source_row_grid[wet_grid==1]
            Wet_Source_Cols = source_col_grid[wet_grid == 1]
            Wet_Source_Levels = source_level_grid[wet_grid == 1]
            Wet_Vals = var_grid[wet_grid == 1]

            Wet_Rows = Wet_Rows[Wet_Vals != 0]
            Wet_Cols = Wet_Cols[Wet_Vals != 0]
            Wet_Source_Rows = Wet_Source_Rows[Wet_Vals != 0]
            Wet_Source_Cols = Wet_Source_Cols[Wet_Vals != 0]
            Wet_Source_Levels = Wet_Source_Levels[Wet_Vals != 0]
            Wet_Vals = Wet_Vals[Wet_Vals != 0]

            if len(Wet_Vals)>0:

                rows_remaining,cols_remaining = np.where(is_remaining)
                for ri in range(n_remaining):
                    row = rows_remaining[ri]
                    col = cols_remaining[ri]
                    row_col_dist = ((Wet_Rows.astype(float)-row)**2 + (Wet_Cols.astype(float)-col)**2)**0.5
                    closest_index = np.argmin(row_col_dist)
                    if row_col_dist[closest_index]<np.sqrt(2):
                        var_grid[row,col] = Wet_Vals[closest_index]
                        source_row_grid[row, col] = Wet_Source_Rows[closest_index]
                        source_col_grid[row, col] = Wet_Source_Cols[closest_index]
                        source_level_grid[row, col] = Wet_Source_Levels[closest_index]

                is_remaining = np.logical_and(var_grid == 0, wet_grid == 1)
                n_remaining_now = np.sum(is_remaining)
                if n_remaining_now<n_remaining:
                    n_remaining = n_remaining_now
                else:
                    n_remaining = n_remaining_now
                    continue_iter=False

            else:
                continue_iter = False

    return(var_grid,source_row_grid,source_col_grid,source_level_grid,n_remaining)


def count_spreading_levels_in_wet_grid(full_grid,level_grid,wet_grid,
                                       source_row_grid_full, source_row_grid,
                                       source_col_grid_full, source_col_grid,
                                       source_level_grid_full, source_level_grid,
                                       level,mean_vertical_difference):

    # if mean_vertical_difference!=0:
    #     print('Using a mean vertical difference of '+str(mean_vertical_difference))

    if level==0:
        bad_row, bad_col = np.where(np.logical_and(level_grid == 0, wet_grid == 1))
        plt.subplot(1, 2, 1)
        plt.imshow(level_grid,origin='lower')
        plt.subplot(1,2,2)
        plt.imshow(wet_grid,origin='lower')
        plt.show()
        raise ValueError('Cannot spread vertically in the surface layer e.g. at row='+str(bad_row[0])+', col='+str(bad_col[0]))

    is_remaining = np.logical_and(level_grid==0,wet_grid==1)
    rows_remaining, cols_remaining = np.where(is_remaining)
    for ri in range(len(rows_remaining)):
        row = rows_remaining[ri]
        col = cols_remaining[ri]
        level_grid[row,col] = full_grid[level-1,row,col]+mean_vertical_difference
        source_row_grid[row, col] = source_row_grid_full[level - 1, row, col]
        source_col_grid[row, col] = source_col_grid_full[level - 1, row, col]
        source_level_grid[row, col] = source_level_grid_full[level - 1, row, col]

    return(level_grid, source_row_grid, source_col_grid, source_level_grid)


def create_interpolation_grid(L0_XC, L0_YC, L0_var, L0_wet_grid, L0_wet_grid_on_L1,
                              XC_subset, YC_subset, L1_wet_grid,
                              testing=False, remove_zeros=True, printing=False, fill_downward=True,
                              mean_vertical_difference=0):
    """
    Interpolates a 3D variable from a Level 0 grid to a Level 1 grid, 
    handling gaps using horizontal and vertical spreading and optionally
    nearest-neighbor fill.

    Parameters:
        L0_XC, L0_YC : 2D arrays of source grid X/Y coordinates
        L0_var : 3D array of source variable to interpolate (levels, lat, lon)
        L0_wet_grid : 3D mask of valid ocean points on L0 grid
        L0_wet_grid_on_L1 : 3D mask of L0 wet grid mapped to L1 domain
        XC_subset, YC_subset : 2D arrays of target grid X/Y coordinates
        L1_wet_grid : 3D mask of valid ocean points on L1 grid

    Returns:
        interpolation_type_grid_full, source_row_grid_full, source_col_grid_full, source_level_grid_full
    """

    fill_with_nearest_at_end = True

    # Initialize output arrays
    shape = (L1_wet_grid.shape[0], XC_subset.shape[0], XC_subset.shape[1])
    full_grid = np.zeros(shape)
    interpolation_type_grid_full = np.zeros(shape, dtype=int)
    source_row_grid_full = np.zeros(shape, dtype=int)
    source_col_grid_full = np.zeros(shape, dtype=int)
    source_level_grid_full = np.zeros(shape, dtype=int)

    rows = np.arange(XC_subset.shape[0])
    cols = np.arange(XC_subset.shape[1])
    Cols, Rows = np.meshgrid(cols, rows)

    K = 1 if testing else L1_wet_grid.shape[0]

    for k in range(K):
        if not np.any(L1_wet_grid[k, :, :] > 0):
            grid = np.zeros_like(XC_subset, dtype=float)
        else:
            if printing:
                print(f' - Working on level {k} of {L1_wet_grid.shape[0]} ({np.sum(L1_wet_grid[k] > 0)} nonzero points found)')

            # Prepare valid interpolation points
            L0_points = np.column_stack([L0_XC.ravel(), L0_YC.ravel()])
            L0_values = L0_var[k].ravel()
            L0_wet = L0_wet_grid[k].ravel()

            mask = L0_wet != 0
            if remove_zeros:
                mask &= L0_values != 0

            L0_points = L0_points[mask]
            L0_values = L0_values[mask]

            # Interpolate using linear or nearest as fallback
            if len(L0_points) > 4:
                grid = griddata(L0_points, L0_values, (XC_subset, YC_subset), method='linear', fill_value=0)
                if not np.any(grid):
                    grid = griddata(L0_points, L0_values, (XC_subset, YC_subset), method='nearest', fill_value=0)
            else:
                grid = np.zeros_like(XC_subset, dtype=float)

            # Mask invalid areas
            grid[L0_wet_grid_on_L1[k] == 0] = 0
            grid[L1_wet_grid[k] == 0] = 0

            # Track interpolation method
            interpolation_type_grid = (grid != 0).astype(int)

            # Track source metadata
            source_row_grid = interpolation_type_grid * Rows
            source_col_grid = interpolation_type_grid * Cols
            source_level_grid = interpolation_type_grid * k

            mask_zeros = interpolation_type_grid == 0
            source_row_grid[mask_zeros] = -1
            source_col_grid[mask_zeros] = -1
            source_level_grid[mask_zeros] = -1

            # Horizontal spreading if necessary
            is_remaining = (grid == 0) & (L1_wet_grid[k] == 1)
            n_remaining = np.sum(is_remaining)
            if printing:
                print(f'   - Remaining points before horizontal spread: {n_remaining}')

            grid, source_row_grid, source_col_grid, source_level_grid, n_remaining = \
                count_spreading_rows_and_cols_in_wet_grid(grid, source_row_grid, source_col_grid, source_level_grid, L1_wet_grid[k])

            interpolation_type_grid[(source_row_grid != -1) & (interpolation_type_grid == 0)] = 2

            if printing:
                print(f'   - Remaining points before downward spread: {n_remaining}')

            # Vertical spreading
            if n_remaining > 0 and fill_downward and k > 0:
                grid, source_row_grid, source_col_grid, source_level_grid = \
                    count_spreading_levels_in_wet_grid(full_grid, grid, L1_wet_grid[k],
                                                       source_row_grid_full, source_row_grid,
                                                       source_col_grid_full, source_col_grid,
                                                       source_level_grid_full, source_level_grid,
                                                       k, mean_vertical_difference)

            interpolation_type_grid[((source_level_grid < k) & (source_level_grid != -1)) & (interpolation_type_grid == 0)] = 3

            # Final fallback: nearest neighbor fill
            if n_remaining > 0 and fill_with_nearest_at_end:
                L1_points = np.column_stack([XC_subset.ravel(), YC_subset.ravel()])
                interp_mask = interpolation_type_grid.ravel() != 0

                filled_points = L1_points[interp_mask]
                filled_values = grid.ravel()[interp_mask]
                filled_rows = source_row_grid.ravel()[interp_mask]
                filled_cols = source_col_grid.ravel()[interp_mask]
                filled_levels = source_level_grid.ravel()[interp_mask]

                fill_rows, fill_cols = np.where((grid == 0) & (L1_wet_grid[k] != 0))
                for ri in range(len(fill_rows)):
                    dists = ((XC_subset[fill_rows[ri], fill_cols[ri]] - filled_points[:, 0])**2 +
                             (YC_subset[fill_rows[ri], fill_cols[ri]] - filled_points[:, 1])**2) ** 0.5
                    ind = np.argmin(dists)
                    grid[fill_rows[ri], fill_cols[ri]] = filled_values[ind]
                    source_row_grid[fill_rows[ri], fill_cols[ri]] = filled_rows[ind]
                    source_col_grid[fill_rows[ri], fill_cols[ri]] = filled_cols[ind]
                    source_level_grid[fill_rows[ri], fill_cols[ri]] = filled_levels[ind]

                n_remaining = np.sum((grid == 0) & (L1_wet_grid[k] != 0))
                if printing:
                    print(f'   - Remaining points after nearest neighbor interpolation: {n_remaining}')

        # Save to full grid arrays
        full_grid[k] = grid
        interpolation_type_grid_full[k] = interpolation_type_grid
        source_row_grid_full[k] = source_row_grid
        source_col_grid_full[k] = source_col_grid
        source_level_grid_full[k] = source_level_grid

    return (interpolation_type_grid_full, source_row_grid_full, source_col_grid_full, source_level_grid_full)


def create_interpolation_grids(ecco_XC, ecco_YC, ecco_wet_cells, ecco_wet_cells_on_tile_domain,
                               XC, YC, domain_wet_cells_3D_C, domain_wet_cells_3D_S, domain_wet_cells_3D_W,
                               testing = False, printing=False):
    """
    Wrapper function to create interpolation grids for C, S, and W masks.

    Returns:
        All interpolation type and source grids for each component
    """
    ecco_grid = -1 * np.ones_like(ecco_wet_cells)

    # Create interpolation grid for each velocity component (C, S, W)
    return (
        create_interpolation_grid(ecco_XC, ecco_YC, ecco_grid, ecco_wet_cells, ecco_wet_cells_on_tile_domain,
                                   XC, YC, domain_wet_cells_3D_C, testing=testing, printing=printing),
        create_interpolation_grid(ecco_XC, ecco_YC, ecco_grid, ecco_wet_cells, ecco_wet_cells_on_tile_domain,
                                   XC, YC, domain_wet_cells_3D_S, testing=testing, printing=printing),
        create_interpolation_grid(ecco_XC, ecco_YC, ecco_grid, ecco_wet_cells, ecco_wet_cells_on_tile_domain,
                                   XC, YC, domain_wet_cells_3D_W, testing=testing, printing=printing)
    )


def write_interpolation_grid_to_nc(output_file_path,
                                   interpolation_type_grid_C, source_row_grid_C, source_col_grid_C, source_level_grid_C,
                                   interpolation_type_grid_S, source_row_grid_S, source_col_grid_S, source_level_grid_S,
                                   interpolation_type_grid_W, source_row_grid_W, source_col_grid_W, source_level_grid_W):
    """
    Writes the interpolation grids to a NetCDF file grouped by C, S, and W.
    """

    ds = nc4.Dataset(output_file_path, 'w')

    for name, itg, srg, scg, slg in zip(['C', 'S', 'W'],
        [interpolation_type_grid_C, interpolation_type_grid_S, interpolation_type_grid_W],
        [source_row_grid_C, source_row_grid_S, source_row_grid_W],
        [source_col_grid_C, source_col_grid_S, source_col_grid_W],
        [source_level_grid_C, source_level_grid_S, source_level_grid_W]):

        grp = ds.createGroup(name)
        grp.createDimension('levels', itg.shape[0])
        grp.createDimension('rows', itg.shape[1])
        grp.createDimension('cols', itg.shape[2])

        grp.createVariable('interp_type', 'i8', ('levels', 'rows', 'cols'))[:, :, :] = itg
        grp.createVariable('source_rows', 'i8', ('levels', 'rows', 'cols'))[:, :, :] = srg
        grp.createVariable('source_cols', 'i8', ('levels', 'rows', 'cols'))[:, :, :] = scg
        grp.createVariable('source_levels', 'i8', ('levels', 'rows', 'cols'))[:, :, :] = slg

    ds.close()

def read_interpolation_grid_from_nc(input_file_path):
    """
    Reads interpolation grids from a NetCDF file and returns them as dictionaries.
    """

    ds = nc4.Dataset(input_file_path, 'r')

    grids = {}
    for name in ['C', 'S', 'W']:
        grp = ds[name]
        grids[name] = {
            'interp_type': grp['interp_type'][:],
            'source_rows': grp['source_rows'][:],
            'source_cols': grp['source_cols'][:],
            'source_levels': grp['source_levels'][:]
        }

    ds.close()
    return grids

def read_interpolation_grid_for_variable(input_file_path, variable_name):
    """
    Reads a specific variable's interpolation grid from a NetCDF file.

    Parameters:
        input_file_path (str): Path to the NetCDF file.
        variable_name (str): Name of the variable to read

    Returns:
        dict: Dictionary containing the interpolation type and source grids.
    """

    if variable_name.lower() in ['vvel','sivice']:
        mask_name = 'S'
    elif variable_name.lower() in ['uvel','siuice']:
        mask_name = 'W'
    else:
        mask_name = 'C'

    ds = nc4.Dataset(input_file_path, 'r')
    grp = ds[mask_name]
    grids = {
        'interp_type': grp['interp_type'][:],
        'source_rows': grp['source_rows'][:],
        'source_cols': grp['source_cols'][:],
        'source_levels': grp['source_levels'][:]
    }
    ds.close()
    return grids
