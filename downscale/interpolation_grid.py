
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata, interp1d
import netCDF4 as nc4


def create_interpolation_grid(L0_XC, L0_YC, L0_var, L0_wet_grid, L0_wet_grid_on_L1,
                              XC_subset, YC_subset, L1_wet_grid):

    testing = False
    remove_zeros = True
    printing = True
    fill_downward = True
    fill_with_nearest_at_end = True
    mean_vertical_difference = 0

    full_grid = np.zeros((np.shape(L1_wet_grid)[0], np.shape(XC_subset)[0], np.shape(XC_subset)[1]))
    interpolation_type_grid_full = np.zeros((np.shape(L1_wet_grid)[0], np.shape(XC_subset)[0], np.shape(XC_subset)[1])).astype(int)
    source_row_grid_full = np.zeros((np.shape(L1_wet_grid)[0], np.shape(XC_subset)[0], np.shape(XC_subset)[1])).astype(int)
    source_col_grid_full = np.zeros((np.shape(L1_wet_grid)[0], np.shape(XC_subset)[0], np.shape(XC_subset)[1])).astype(int)
    source_level_grid_full = np.zeros((np.shape(L1_wet_grid)[0], np.shape(XC_subset)[0], np.shape(XC_subset)[1])).astype(int)

    rows = np.arange(np.shape(XC_subset)[0])
    cols = np.arange(np.shape(XC_subset)[1])
    Cols, Rows = np.meshgrid(cols, rows)

    if testing:
        K = 1
    else:
        K = np.shape(L1_wet_grid)[0]

    for k in range(K):

        continue_to_interpolation = True

        if continue_to_interpolation:
            if printing:
                print('                - Working on level ' + str(k) + ' of ' + str(
                    np.shape(L1_wet_grid)[0]) + ' (' + str(np.sum(L1_wet_grid[k, :, :] > 0)) + ' nonzero points found)')

            if np.any(L1_wet_grid[k, :, :] > 0):
                # take an initial stab at the interpolation
                L0_points = np.hstack([np.reshape(L0_XC, (np.size(L0_XC), 1)),
                                       np.reshape(L0_YC, (np.size(L0_YC), 1))])
                L0_values = np.reshape(L0_var[k, :, :], (np.size(L0_var[k, :, :]), 1))
                L0_wet_grid_vert = np.reshape(L0_wet_grid[k, :, :], (np.size(L0_wet_grid[k, :, :]), 1))
                L0_points = L0_points[L0_wet_grid_vert[:, 0] != 0, :]
                L0_values = L0_values[L0_wet_grid_vert[:, 0] != 0, :]
                if remove_zeros:
                    L0_points = L0_points[L0_values[:, 0] != 0, :]
                    L0_values = L0_values[L0_values[:, 0] != 0, :]

                if len(L0_points) > 4:
                    grid = griddata(L0_points, L0_values, (XC_subset, YC_subset), method='linear', fill_value=0)
                    grid = grid[:, :, 0]
                    # grid_nearest = griddata(L0_points, L0_values, (XC_subset, YC_subset), method='nearest', fill_value=0)
                    # grid_nearest[:,:,0]
                    if not np.any(grid != 0):
                        grid = griddata(L0_points, L0_values, (XC_subset, YC_subset), method='nearest', fill_value=0)
                        grid = grid[:, :, 0]
                else:
                    grid = np.zeros_like(XC_subset).astype(float)

                # if k==0:
                #     plt.imshow(grid,origin='lower')
                #     plt.show()

                # mask out any values which should be 0'd based on the old bathy
                grid[L0_wet_grid_on_L1[k, :, :] == 0] = 0

                # mask out any values which should be 0'd based on the new bathy
                grid[L1_wet_grid[k, :, :] == 0] = 0

                # mask a mask of where the interpolation occured
                interpolation_type_grid = (grid!=0).astype(int)

                # make arrays of rows cols and levels wherever the interpolation occured
                source_row_grid = np.copy(interpolation_type_grid) * Rows
                source_col_grid = np.copy(interpolation_type_grid) * Cols
                source_level_grid = np.copy(interpolation_type_grid) * k

                # set areas that havent been filled yet to -1 because rows, cols and levels can have index 0
                negative_interpolation_type_grid = np.copy(interpolation_type_grid)
                negative_interpolation_type_grid[interpolation_type_grid == 0] = - 1
                source_row_grid[negative_interpolation_type_grid == -1] = -1
                source_col_grid[negative_interpolation_type_grid == -1] = -1
                source_level_grid[negative_interpolation_type_grid == -1] = -1

                is_remaining = np.logical_and(grid == 0, L1_wet_grid[k, :, :] == 1)
                n_remaining = np.sum(is_remaining)
                if printing:
                    print('                  - Remaining points before horizontal spread: ' + str(n_remaining))

                # spread the the variable outward to new wet cells, keeping track of the source rows, cols, and levels as you go
                grid, source_row_grid, source_col_grid, source_level_grid, n_remaining = \
                    count_spreading_rows_and_cols_in_wet_grid(grid, source_row_grid,source_col_grid,source_level_grid, L1_wet_grid[k, :, :])

                # mark the interpolation grid to indicate the variable was spread horizontally
                interpolation_type_grid[np.logical_and(source_row_grid != -1, interpolation_type_grid == 0)] = 2

                if printing:
                    print('                  - Remaining points before downward spread: '+str(n_remaining)+' (check: filled '+str(np.sum(interpolation_type_grid==2))+')')

                # if there are still values which need to be filled, spread downward
                if n_remaining > 0 and fill_downward and k > 0:
                    grid, source_row_grid, source_col_grid, source_level_grid = \
                        count_spreading_levels_in_wet_grid(full_grid, grid,L1_wet_grid[k, :, :],
                                                           source_row_grid_full, source_row_grid,
                                                           source_col_grid_full, source_col_grid,
                                                           source_level_grid_full, source_level_grid,
                                                           k,mean_vertical_difference)

                # if k<5:
                #     C = plt.imshow(source_level_grid,origin='lower')
                #     plt.colorbar(C)
                #     plt.show()

                # mark the interpolation grid to indicate the variable was spread vertically
                interpolation_type_grid[np.logical_and(np.logical_and(source_level_grid<k,source_level_grid!=-1),
                                                       interpolation_type_grid == 0)] = 3

                n_remaining = np.sum(np.logical_and(grid==0,L1_wet_grid[k,:,:]!=0))
                if printing:
                    print('                  - Remaining points after downward spread: '+str(n_remaining)+' (check: filled '+str(np.sum(interpolation_type_grid==3))+')')

                # if, for whatever reason, there are still values to be filled, then fill em with the nearest neighbor
                # if the bathy is already "filled" then this should only pertain to the W and S masks
                # in other words, this is just used for velocity and maybe it is best to fill these with zeros in weird
                # near-coastal "holes"
                if n_remaining > 0 and fill_with_nearest_at_end:
                    if len(L0_points) > 0:

                        L1_points = np.column_stack([XC_subset.ravel(), YC_subset.ravel()])
                        interpolation_values = interpolation_type_grid.ravel()

                        filled_L1_points = L1_points[interpolation_values!=0,:]
                        filled_L1_values = grid.ravel()[interpolation_values!=0]
                        filled_source_rows = source_row_grid.ravel()[interpolation_values!=0]
                        filled_source_cols = source_col_grid.ravel()[interpolation_values != 0]
                        filled_source_levels = source_level_grid.ravel()[interpolation_values != 0]
                        fill_rows, fill_cols = np.where(np.logical_and(grid == 0, L1_wet_grid[k, :, :] != 0))

                        for ri in range(len(fill_rows)):
                            dist = ((XC_subset[fill_rows[ri],fill_cols[ri]]-filled_L1_points[:,0])**2 + \
                                   (YC_subset[fill_rows[ri],fill_cols[ri]]-filled_L1_points[:,1])**2)**0.5
                            ind = np.where(dist==np.min(dist))[0][0]
                            grid[fill_rows[ri],fill_cols[ri]] = filled_L1_values[ind]
                            source_row_grid[fill_rows[ri],fill_cols[ri]] = filled_source_rows[ind]
                            source_col_grid[fill_rows[ri], fill_cols[ri]] = filled_source_cols[ind]
                            source_level_grid[fill_rows[ri], fill_cols[ri]] = filled_source_levels[ind]

                    n_remaining = np.sum(np.logical_and(grid == 0, L1_wet_grid[k, :, :] != 0))
                    if printing:
                        print('                  - Remaining points after nearest neighbor interpolation: ' + str(n_remaining))

                # # mark the interpolation grid to indicate the variable was spread vertically
                # interpolation_type_grid[np.logical_and(source_row_grid != -1, interpolation_type_grid == 0)] = 3


            else:
                grid = np.zeros_like(XC_subset).astype(float)

        full_grid[k, :, :] = grid[:, :]
        interpolation_type_grid_full[k,:,:] = interpolation_type_grid
        source_row_grid_full[k, :, :] = source_row_grid
        source_col_grid_full[k, :, :] = source_col_grid
        source_level_grid_full[k, :, :] = source_level_grid

    return (interpolation_type_grid_full, source_row_grid_full, source_col_grid_full, source_level_grid_full)

def create_interpolation_grids(ecco_XC, ecco_YC, ecco_wet_cells, ecco_wet_cells_on_tile_domain,
                               XC, YC, domain_wet_cells_3D_C, domain_wet_cells_3D_S, domain_wet_cells_3D_W):

    # we will try to spread around a full grid of -1s
    ecco_grid = -1*np.ones_like(ecco_wet_cells)

    interpolation_type_grid_C, source_row_grid_C, source_col_grid_C, source_level_grid_C, =\
        create_interpolation_grid(ecco_XC, ecco_YC, ecco_grid, ecco_wet_cells, ecco_wet_cells_on_tile_domain,
                                  XC, YC, domain_wet_cells_3D_C)

    interpolation_type_grid_S, source_row_grid_S, source_col_grid_S, source_level_grid_S, = \
        create_interpolation_grid(ecco_XC, ecco_YC, ecco_grid, ecco_wet_cells, ecco_wet_cells_on_tile_domain,
                                  XC, YC, domain_wet_cells_3D_S)

    interpolation_type_grid_W, source_row_grid_W, source_col_grid_W, source_level_grid_W, = \
        create_interpolation_grid(ecco_XC, ecco_YC, ecco_grid, ecco_wet_cells, ecco_wet_cells_on_tile_domain,
                                  XC, YC, domain_wet_cells_3D_W)

    return(interpolation_type_grid_C, source_row_grid_C, source_col_grid_C, source_level_grid_C,
           interpolation_type_grid_S, source_row_grid_S, source_col_grid_S, source_level_grid_S,
           interpolation_type_grid_W, source_row_grid_W, source_col_grid_W, source_level_grid_W)

def write_interpolation_grid_to_nc(config_dir, model_name,
                                   interpolation_type_grid_C, source_row_grid_C, source_col_grid_C, source_level_grid_C,
                                   interpolation_type_grid_S, source_row_grid_S, source_col_grid_S, source_level_grid_S,
                                   interpolation_type_grid_W, source_row_grid_W, source_col_grid_W, source_level_grid_W):

    interpolation_grid_file = model_name + '_interpolation_grid.nc'
    ds = nc4.Dataset(os.path.join(config_dir, 'nc_grids', interpolation_grid_file), 'w')

    grp = ds.createGroup('C')
    grp.createDimension('levels', np.shape(interpolation_type_grid_C)[0])
    grp.createDimension('rows', np.shape(interpolation_type_grid_C)[1])
    grp.createDimension('cols', np.shape(interpolation_type_grid_C)[2])
    var = grp.createVariable('interp_type', 'i8', ('levels', 'rows', 'cols'))
    var[:, :, :] = interpolation_type_grid_C
    var = grp.createVariable('source_rows', 'i8', ('levels', 'rows', 'cols'))
    var[:, :, :] = source_row_grid_C
    var = grp.createVariable('source_cols', 'i8', ('levels', 'rows', 'cols'))
    var[:, :, :] = source_col_grid_C
    var = grp.createVariable('source_levels', 'i8', ('levels', 'rows', 'cols'))
    var[:, :, :] = source_level_grid_C

    grp = ds.createGroup('S')
    grp.createDimension('levels', np.shape(interpolation_type_grid_S)[0])
    grp.createDimension('rows', np.shape(interpolation_type_grid_S)[1])
    grp.createDimension('cols', np.shape(interpolation_type_grid_S)[2])
    var = grp.createVariable('interp_type', 'i8', ('levels', 'rows', 'cols'))
    var[:, :, :] = interpolation_type_grid_S
    var = grp.createVariable('source_rows', 'i8', ('levels', 'rows', 'cols'))
    var[:, :, :] = source_row_grid_S
    var = grp.createVariable('source_cols', 'i8', ('levels', 'rows', 'cols'))
    var[:, :, :] = source_col_grid_S
    var = grp.createVariable('source_levels', 'i8', ('levels', 'rows', 'cols'))
    var[:, :, :] = source_level_grid_S

    grp = ds.createGroup('W')
    grp.createDimension('levels', np.shape(interpolation_type_grid_W)[0])
    grp.createDimension('rows', np.shape(interpolation_type_grid_W)[1])
    grp.createDimension('cols', np.shape(interpolation_type_grid_W)[2])
    var = grp.createVariable('interp_type', 'i8', ('levels', 'rows', 'cols'))
    var[:, :, :] = interpolation_type_grid_W
    var = grp.createVariable('source_rows', 'i8', ('levels', 'rows', 'cols'))
    var[:, :, :] = source_row_grid_W
    var = grp.createVariable('source_cols', 'i8', ('levels', 'rows', 'cols'))
    var[:, :, :] = source_col_grid_W
    var = grp.createVariable('source_levels', 'i8', ('levels', 'rows', 'cols'))
    var[:, :, :] = source_level_grid_W

    ds.close()