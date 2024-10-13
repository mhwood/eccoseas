
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata, interp1d
import netCDF4 as nc4


def spread_var_horizontally_in_wet_grid(var_grid,wet_grid):
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
            Wet_Vals = var_grid[wet_grid == 1]
            Wet_Rows = Wet_Rows[Wet_Vals != 0]
            Wet_Cols = Wet_Cols[Wet_Vals != 0]
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

                is_remaining = np.logical_and(var_grid == 0, wet_grid == 1)
                n_remaining_now = np.sum(is_remaining)
                if n_remaining_now<n_remaining:
                    n_remaining = n_remaining_now
                else:
                    n_remaining = n_remaining_now
                    continue_iter=False

            else:
                continue_iter = False

    return(var_grid,n_remaining)

def spread_var_horizontally_in_wet_grid_with_mask(var_grid,wet_grid,spread_mask):

    rows = np.arange(np.shape(var_grid)[0])
    cols = np.arange(np.shape(var_grid)[1])
    Cols,Rows = np.meshgrid(cols,rows)

    is_remaining = np.logical_and(spread_mask==0,wet_grid==1)
    n_remaining = np.sum(is_remaining)
    continue_iter = True
    counter = 0
    for i in range(n_remaining):
        if continue_iter:
            Wet_Rows = Rows[wet_grid == 1]
            Wet_Cols = Cols[wet_grid == 1]
            Wet_Vals = var_grid[wet_grid == 1]
            Wet_Spreads = spread_mask[wet_grid == 1]

            Wet_Rows = Wet_Rows[Wet_Spreads == 0]
            Wet_Cols = Wet_Cols[Wet_Spreads == 0]
            Wet_Vals = Wet_Vals[Wet_Spreads == 0]

            if len(Wet_Vals)>0:

                rows_remaining,cols_remaining = np.where(is_remaining)
                for ri in range(n_remaining):
                    row = rows_remaining[ri]
                    col = cols_remaining[ri]
                    row_col_dist = ((Wet_Rows.astype(float)-row)**2 + (Wet_Cols.astype(float)-col)**2)**0.5
                    closest_index = np.argmin(row_col_dist)
                    if row_col_dist[closest_index]<=np.sqrt(2):
                        var_grid[row,col] = Wet_Vals[closest_index]
                        spread_mask[row,col]==1
                        counter+=1

                is_remaining = np.logical_and(spread_mask == 0, wet_grid == 1)
                n_remaining_now = np.sum(is_remaining)
                if n_remaining_now<n_remaining:
                    n_remaining = n_remaining_now
                else:
                    n_remaining = n_remaining_now
                    continue_iter=False

            else:
                continue_iter = False

    return(var_grid,n_remaining)

def spread_var_vertically_in_wet_grid(full_grid,level_grid,wet_grid,level,mean_vertical_difference):

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

    return(level_grid)


def downscale_2D_field(L0_XC, L0_YC, L0_var, L0_wet_grid, L0_wet_grid_on_L1,
                       XC_subset, YC_subset, L1_wet_grid,spread_horizontally=True,remove_zeros=True):

    # try to interpolate everything using a linear interpolation first
    L0_points = np.hstack([np.reshape(L0_XC, (np.size(L0_XC), 1)),
                              np.reshape(L0_YC, (np.size(L0_YC), 1))])
    L0_values = np.reshape(L0_var, (np.size(L0_var), 1))

    L0_wet_grid = np.reshape(L0_wet_grid, (np.size(L0_wet_grid), 1))
    if remove_zeros:
        L0_points = L0_points[L0_wet_grid[:, 0] != 0, :]
        L0_values = L0_values[L0_wet_grid[:, 0] != 0, :]

    grid = griddata(L0_points, L0_values, (XC_subset, YC_subset), method='linear',fill_value=0)
    grid = grid[:, :, 0]

    # mask out any values which should be 0'd based on the old bathy
    grid[L0_wet_grid_on_L1 == 0] = 0

    # mask out any values which should be 0'd based on the new bathy
    grid[L1_wet_grid == 0] = 0

    # plt.imshow(grid, origin='lower')
    # plt.show()

    # spread the the variable outward to new wet cells
    if spread_horizontally:
        grid, _ = spread_var_horizontally_in_wet_grid(grid, L1_wet_grid)

    # C = plt.imshow(downscaled_grid,origin='lower',
    #                vmin=np.min(downscaled_grid[downscaled_grid!=0]),vmax=np.max(downscaled_grid[downscaled_grid!=0]))
    # plt.colorbar(C)
    # plt.show()
    
    return(grid)


def downscale_3D_field(L0_XC, L0_YC, L0_var, L0_wet_grid, L0_wet_grid_on_L1,
                       XC_subset, YC_subset, L1_wet_grid,
                       mean_vertical_difference=0,fill_downward=True,
                       printing=False,remove_zeros=True, testing = False):

    # full_grid = np.zeros_like(L1_wet_grid).astype(float)
    full_grid = np.zeros((np.shape(L1_wet_grid)[0],np.shape(XC_subset)[0],np.shape(XC_subset)[1]))

    if testing:
        K=1
    else:
        K=np.shape(L1_wet_grid)[0]

    for k in range(K):

        continue_to_interpolation = True

        if continue_to_interpolation:
            if printing:
                print('                - Working on level ' + str(k) + ' of ' + str(np.shape(L1_wet_grid)[0])+' ('+str(np.sum(L1_wet_grid[k, :, :] > 0))+' nonzero points found)')
            if np.any(L1_wet_grid[k, :, :] > 0):
                # take an initial stab at the interpolation
                L0_points = np.hstack([np.reshape(L0_XC, (np.size(L0_XC), 1)),
                                       np.reshape(L0_YC, (np.size(L0_YC), 1))])
                L0_values = np.reshape(L0_var[k, :, :], (np.size(L0_var[k, :, :]), 1))
                L0_wet_grid_vert = np.reshape(L0_wet_grid[k, :, :], (np.size(L0_wet_grid[k, :, :]), 1))
                L0_points = L0_points[L0_wet_grid_vert[:,0] != 0, :]
                L0_values = L0_values[L0_wet_grid_vert[:,0] != 0, :]
                if remove_zeros:
                    L0_points = L0_points[L0_values[:, 0] != 0, :]
                    L0_values = L0_values[L0_values[:, 0] != 0, :]

                if len(L0_points)>4:
                    grid = griddata(L0_points, L0_values, (XC_subset, YC_subset), method='linear',fill_value=0)
                    grid = grid[:, :, 0]
                    # grid_nearest = griddata(L0_points, L0_values, (XC_subset, YC_subset), method='nearest', fill_value=0)
                    # grid_nearest[:,:,0]
                    if not np.any(grid!=0):
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

                # spread the the variable outward to new wet cells
                grid, n_remaining = spread_var_horizontally_in_wet_grid(grid, L1_wet_grid[k, :, :])

                # if there are still values which need to be filled, spread downward
                if n_remaining > 0 and fill_downward and k>0:
                    grid = spread_var_vertically_in_wet_grid(full_grid, grid, L1_wet_grid[k, :, :], k, mean_vertical_difference)

                # if, for whatever reason, there are still values to be filled, then fill em with the nearest neighbor
                if n_remaining>0:
                    if len(L0_points) > 0:
                        grid_nearest = griddata(L0_points, L0_values, (XC_subset, YC_subset), method='nearest', fill_value=0)
                        grid_nearest = grid_nearest[:, :, 0]
                        indices = np.logical_and(grid==0,L1_wet_grid[k, :, :] != 0)
                        grid[indices] = grid_nearest[indices]

                # # if there are any issues in the surface layer, then fill with a close point
                # if n_remaining > 0 and fill_downward and k == 0:
                #     bad_row, bad_col = np.where(np.logical_and(grid == 0, L1_wet_grid[0,:,:] == 1))
                #     good_row, good_col = np.where(np.logical_and(grid > 0, L1_wet_grid[0,:,:] == 1))
                #     for ri in range(len(bad_row)):
                #         dist = (bad_row[ri]-good_row)**2 + (bad_col[ri]-good_col)**2
                #         fill_index = np.argmin(dist)
                #         fill_val = grid[good_row[fill_index],good_col[fill_index]]
                #         # print('Filling row '+str(bad_row[ri])+', col '+str(bad_col[ri])+
                #         #       ' with value = '+str(fill_val)+' from row '+str(good_row[fill_index])+
                #         #       ', col '+str(good_col[fill_index]))
                #         grid[bad_row[ri],bad_col[ri]] = fill_val
            else:
                grid = np.zeros_like(XC_subset).astype(float)

        full_grid[k, :, :] = grid[:, :]

    return(full_grid)

def downscale_3D_field_with_interpolation_mask(L0_XC, L0_YC, L0_var, L0_wet_grid, L0_wet_grid_on_L1,
                                               XC_subset, YC_subset, L1_wet_grid,
                                               L1_interpolation_mask, L1_source_rows, L1_source_cols, L1_source_levels,
                                               mean_vertical_difference=0,fill_downward=True,
                                               printing=False, remove_zeros=True, testing = False):

    # full_grid = np.zeros_like(L1_wet_grid).astype(float)
    full_grid = np.zeros((np.shape(L1_wet_grid)[0],np.shape(XC_subset)[0],np.shape(XC_subset)[1]))

    if testing:
        K=1
    else:
        K=np.shape(L1_wet_grid)[0]

    for k in range(K):

        continue_to_interpolation = True

        tiny_value = 1e-14

        if continue_to_interpolation:
            if printing:
                print('                - Working on level ' + str(k) + ' of ' + str(np.shape(L1_wet_grid)[0])+' ('+str(np.sum(L1_wet_grid[k, :, :] > 0))+' nonzero points found)')
            if np.any(L1_wet_grid[k, :, :] > 0):
                # take an initial stab at the interpolation
                L0_points = np.hstack([np.reshape(L0_XC, (np.size(L0_XC), 1)),
                                       np.reshape(L0_YC, (np.size(L0_YC), 1))])
                L0_values = np.reshape(L0_var[k, :, :], (np.size(L0_var[k, :, :]), 1))
                L0_wet_grid_vert = np.reshape(L0_wet_grid[k, :, :], (np.size(L0_wet_grid[k, :, :]), 1))
                L0_points = L0_points[L0_wet_grid_vert[:,0] != 0, :]
                L0_values = L0_values[L0_wet_grid_vert[:,0] != 0, :]

                # fill the zeros with a very tiny value
                L0_values[L0_values[:, 0] == 0, :] = tiny_value

                if len(L0_points)>4:
                    grid = griddata(L0_points, L0_values, (XC_subset, YC_subset), method='linear',fill_value=0)
                    grid = grid[:, :, 0]
                    # grid_nearest = griddata(L0_points, L0_values, (XC_subset, YC_subset), method='nearest', fill_value=0)
                    # grid_nearest[:,:,0]
                    if not np.any(grid!=0):
                        grid = griddata(L0_points, L0_values, (XC_subset, YC_subset), method='nearest', fill_value=0)
                        grid = grid[:, :, 0]
                else:
                    grid = np.zeros_like(XC_subset).astype(float)

                # if k==0:
                #     plt.imshow(grid,origin='lower')
                #     plt.show()

                # mask out any values which should be 0'd based on the interpolation mask
                grid[L1_interpolation_mask[k, :, :] != 1] = 0

                # in this level, spread points where they should be spread
                spread_rows, spread_cols = np.where(L1_interpolation_mask[k,:,:]==2)
                for ri in range(len(spread_rows)):
                    source_row = L1_source_rows[k,spread_rows[ri],spread_cols[ri]]
                    source_col = L1_source_cols[k, spread_rows[ri], spread_cols[ri]]
                    source_level = L1_source_levels[k, spread_rows[ri], spread_cols[ri]]
                    if source_level !=k:
                         value = full_grid[source_level,source_row,source_col]
                    else:
                        value = grid[source_row,source_col]

                    # print('        - Filling in point at location '+str(k)+','+str(spread_rows[ri])+','+str(spread_cols[ri])+\
                    #       ' with point at location '+str(source_level)+','+str(source_row)+','+str(source_col)+' (value = '+str(value)+')')
                    grid[spread_rows[ri], spread_cols[ri]] = value

                # if there's any that were filled from above, then fill em
                downward_rows, downward_cols = np.where(L1_interpolation_mask[k, :, :] == 3)
                for ri in range(len(downward_rows)):
                    source_level = L1_source_levels[k, downward_rows[ri], downward_cols[ri]]
                    value = full_grid[source_level, source_row, source_col]
                    grid[downward_rows[ri], downward_cols[ri]] = value

                # now, add the end of it, make all of the tiny values actually 0
                grid[np.abs(grid) <= 2 * tiny_value] = 0

            else:
                grid = np.zeros_like(XC_subset).astype(float)

        full_grid[k, :, :] = grid[:, :]

    return(full_grid)


def downscale_3D_field_with_zeros(L0_XC, L0_YC, L0_var, L0_wet_grid, L0_wet_grid_on_L1,
                                   XC_subset, YC_subset, L1_wet_grid,
                                   mean_vertical_difference=0,fill_downward=True,printing=False,remove_zeros=True):

    # full_grid = np.zeros_like(L1_wet_grid).astype(float)
    full_grid = np.zeros((np.shape(L1_wet_grid)[0],np.shape(XC_subset)[0],np.shape(XC_subset)[1]))

    for k in range(np.shape(L1_wet_grid)[0]):

        continue_to_interpolation = True

        tiny_value = 1e-14

        if continue_to_interpolation:
            if printing:
                print('                - Working on level ' + str(k) + ' of ' + str(np.shape(L1_wet_grid)[0])+' ('+str(np.sum(L1_wet_grid[k, :, :] > 0))+' nonzero points found)')
            if np.any(L1_wet_grid[k, :, :] > 0):
                # take an initial stab at the interpolation
                L0_points = np.hstack([np.reshape(L0_XC, (np.size(L0_XC), 1)),
                                       np.reshape(L0_YC, (np.size(L0_YC), 1))])
                L0_values = np.reshape(L0_var[k, :, :], (np.size(L0_var[k, :, :]), 1))
                L0_wet_grid_vert = np.reshape(L0_wet_grid[k, :, :], (np.size(L0_wet_grid[k, :, :]), 1))
                L0_points = L0_points[L0_wet_grid_vert[:,0] != 0, :]
                L0_values = L0_values[L0_wet_grid_vert[:,0] != 0, :]

                # fill the zeros with a very tiny value
                L0_values[L0_values[:, 0] == 0, :] = tiny_value

                if len(L0_points)>4:
                    grid = griddata(L0_points, L0_values, (XC_subset, YC_subset), method='linear',fill_value=0)
                    grid = grid[:, :, 0]
                    # grid_nearest = griddata(L0_points, L0_values, (XC_subset, YC_subset), method='nearest', fill_value=0)
                    # grid_nearest[:,:,0]
                    if not np.any(grid!=0):
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

                # spread the the variable outward to new wet cells
                grid, n_remaining = spread_var_horizontally_in_wet_grid(grid, L1_wet_grid[k, :, :])

                # if there are still values which need to be filled, spread downward
                if n_remaining > 0 and fill_downward and k>0:
                    grid = spread_var_vertically_in_wet_grid(full_grid, grid, L1_wet_grid[k, :, :], k, mean_vertical_difference)

                # if, for whatever reason, there are still values to be filled, then fill em with the nearest neighbor
                if n_remaining>0:
                    if len(L0_points) > 0:
                        grid_nearest = griddata(L0_points, L0_values, (XC_subset, YC_subset), method='nearest', fill_value=0)
                        grid_nearest = grid_nearest[:, :, 0]
                        indices = np.logical_and(grid==0,L1_wet_grid[k, :, :] != 0)
                        grid[indices] = grid_nearest[indices]

                # now, add the end of it, make all of the tiny values actually 0
                grid[np.abs(grid)<=2*tiny_value] = 0

                # # if there are any issues in the surface layer, then fill with a close point
                # if n_remaining > 0 and fill_downward and k == 0:
                #     bad_row, bad_col = np.where(np.logical_and(grid == 0, L1_wet_grid[0,:,:] == 1))
                #     good_row, good_col = np.where(np.logical_and(grid > 0, L1_wet_grid[0,:,:] == 1))
                #     for ri in range(len(bad_row)):
                #         dist = (bad_row[ri]-good_row)**2 + (bad_col[ri]-good_col)**2
                #         fill_index = np.argmin(dist)
                #         fill_val = grid[good_row[fill_index],good_col[fill_index]]
                #         # print('Filling row '+str(bad_row[ri])+', col '+str(bad_col[ri])+
                #         #       ' with value = '+str(fill_val)+' from row '+str(good_row[fill_index])+
                #         #       ', col '+str(good_col[fill_index]))
                #         grid[bad_row[ri],bad_col[ri]] = fill_val
            else:
                grid = np.zeros_like(XC_subset).astype(float)

        full_grid[k, :, :] = grid[:, :]

    return(full_grid)

def downscale_3D_points(L0_points, L0_var, L0_wet_grid,
                        XC_subset, YC_subset, L1_wet_grid,
                        mean_vertical_difference=0,fill_downward=True,
                        printing=False,remove_zeros=True, testing=False):

    # full_grid = np.zeros_like(L1_wet_grid).astype(float)
    full_grid = np.zeros((np.shape(L1_wet_grid)[0],np.shape(XC_subset)[0],np.shape(XC_subset)[1]))
    all_L0_points = np.copy(L0_points)

    if testing:
        kMax = 1
    else:
        kMax = np.shape(L0_var)[0]

    for k in range(kMax):

        continue_to_interpolation = True

        if continue_to_interpolation:
            if printing:
                print('    Working on level ' + str(k) + ' of ' + str(np.shape(L0_var)[0])+' ('+str(np.sum(L1_wet_grid[k, :, :] > 0))+' nonzero points found)')
            if np.any(L1_wet_grid[k, :, :] > 0):
                # take an initial stab at the interpolation
                L0_points = np.copy(all_L0_points)
                L0_values = np.reshape(L0_var[k, :], (np.size(L0_var[k, :]), ))
                L0_wet_grid_vert = np.reshape(L0_wet_grid[k, :], (np.size(L0_wet_grid[k, :]), ))
                L0_points = L0_points[L0_wet_grid_vert != 0, :]
                L0_values = L0_values[L0_wet_grid_vert != 0]
                if remove_zeros:
                    L0_points = L0_points[L0_values != 0, :]
                    L0_values = L0_values[L0_values != 0]

                if len(L0_points)>4:
                    grid = griddata(L0_points, L0_values, (XC_subset, YC_subset), method='linear',fill_value=0)
                    # grid = grid[:, :, 0]
                    # grid_nearest = griddata(L0_points, L0_values, (XC_subset, YC_subset), method='nearest', fill_value=0)
                    # grid_nearest[:,:,0]
                    if not np.any(grid!=0):
                        grid = griddata(L0_points, L0_values, (XC_subset, YC_subset), method='nearest', fill_value=0)
                        # grid = grid[:, :, 0]
                else:
                    grid = np.zeros_like(XC_subset).astype(float)

                # if k==0:
                #     plt.imshow(grid,origin='lower')
                #     plt.show()

                # mask out any values which should be 0'd based on the old bathy
                #grid[L0_wet_grid_on_L1[k, :, :] == 0] = 0

                # mask out any values which should be 0'd based on the new bathy
                grid[L1_wet_grid[k, :, :] == 0] = 0

                # spread the the variable outward to new wet cells
                grid, n_remaining = spread_var_horizontally_in_wet_grid(grid, L1_wet_grid[k, :, :])

                # if there are still values which need to be filled, spread downward
                if n_remaining > 0 and fill_downward and k>0:
                    grid = spread_var_vertically_in_wet_grid(full_grid, grid, L1_wet_grid[k, :, :], k, mean_vertical_difference)

                # if, for whatever reason, there are still values to be filled, then fill em with the nearest neighbor
                if n_remaining>0:
                    if len(L0_points) > 0:
                        grid_nearest = griddata(L0_points, L0_values, (XC_subset, YC_subset), method='nearest', fill_value=0)
                        # grid_nearest = grid_nearest[:, :, 0]
                        indices = np.logical_and(grid==0,L1_wet_grid[k, :, :] != 0)
                        grid[indices] = grid_nearest[indices]

            else:
                grid = np.zeros_like(XC_subset).astype(float)

        full_grid[k, :, :] = grid[:, :]

    return(full_grid)

def downscale_3D_points_with_zeros(L0_points, L0_var, L0_wet_grid,
                       XC_subset, YC_subset, L1_wet_grid,
                       mean_vertical_difference=0,fill_downward=True,printing=False, testing=False):

    # full_grid = np.zeros_like(L1_wet_grid).astype(float)
    full_grid = np.zeros((np.shape(L1_wet_grid)[0],np.shape(XC_subset)[0],np.shape(XC_subset)[1]))
    all_L0_points = np.copy(L0_points)

    tiny_value = 1e-14

    for k in range(np.shape(L1_wet_grid)[0]):

        continue_to_interpolation = True

        if continue_to_interpolation:
            if printing:
                print('    Working on level ' + str(k) + ' of ' + str(np.shape(L0_var)[0])+' ('+str(np.sum(L1_wet_grid[k, :, :] > 0))+' nonzero points found)')
            if np.any(L1_wet_grid[k, :, :] > 0):
                # take an initial stab at the interpolation
                L0_points = np.copy(all_L0_points)
                L0_values = np.reshape(L0_var[k, :], (np.size(L0_var[k, :]), ))
                L0_wet_grid_vert = np.reshape(L0_wet_grid[k, :], (np.size(L0_wet_grid[k, :]), ))
                L0_points = L0_points[L0_wet_grid_vert != 0, :]
                L0_values = L0_values[L0_wet_grid_vert != 0]

                # fill the zeros with a very tiny value
                L0_values[L0_values == 0] = tiny_value

                if len(L0_points)>4:
                    grid = griddata(L0_points, L0_values, (XC_subset, YC_subset), method='linear',fill_value=0)
                    # grid = grid[:, :, 0]
                    # grid_nearest = griddata(L0_points, L0_values, (XC_subset, YC_subset), method='nearest', fill_value=0)
                    # grid_nearest[:,:,0]
                    if not np.any(grid!=0):
                        grid = griddata(L0_points, L0_values, (XC_subset, YC_subset), method='nearest', fill_value=0)
                        # grid = grid[:, :, 0]
                else:
                    grid = np.zeros_like(XC_subset).astype(float)

                # if k==0:
                #     plt.imshow(grid,origin='lower')
                #     plt.show()

                # mask out any values which should be 0'd based on the old bathy
                #grid[L0_wet_grid_on_L1[k, :, :] == 0] = 0

                # mask out any values which should be 0'd based on the new bathy
                grid[L1_wet_grid[k, :, :] == 0] = 0

                # spread the the variable outward to new wet cells
                grid, n_remaining = spread_var_horizontally_in_wet_grid(grid, L1_wet_grid[k, :, :])

                # if there are still values which need to be filled, spread downward
                if n_remaining > 0 and fill_downward and k>0:
                    grid = spread_var_vertically_in_wet_grid(full_grid, grid, L1_wet_grid[k, :, :], k, mean_vertical_difference)

                # if, for whatever reason, there are still values to be filled, then fill em with the nearest neighbor
                if n_remaining>0:
                    if len(L0_points) > 0:
                        grid_nearest = griddata(L0_points, L0_values, (XC_subset, YC_subset), method='nearest', fill_value=0)
                        # grid_nearest = grid_nearest[:, :, 0]
                        indices = np.logical_and(grid==0,L1_wet_grid[k, :, :] != 0)
                        grid[indices] = grid_nearest[indices]

                # now, add the end of it, make all of the tiny values actually 0
                grid[np.abs(grid) <= 2 * tiny_value] = 0

            else:
                grid = np.zeros_like(XC_subset).astype(float)

        full_grid[k, :, :] = grid[:, :]

    return(full_grid)

def downscale_3D_points_with_interpolation_mask(L0_points, L0_var, L0_wet_grid,
                                               XC_subset, YC_subset, L1_wet_grid,
                                               L1_interpolation_mask, L1_source_rows, L1_source_cols, L1_source_levels,
                                               printing=False, testing = False):

    # full_grid = np.zeros_like(L1_wet_grid).astype(float)
    full_grid = np.zeros((np.shape(L1_wet_grid)[0],np.shape(XC_subset)[0],np.shape(XC_subset)[1]))
    all_L0_points = np.copy(L0_points)

    if testing:
        K=1
    else:
        K=np.shape(L1_wet_grid)[0]

    for k in range(K):

        continue_to_interpolation = True

        tiny_value = 1e-14

        if continue_to_interpolation:
            if printing:
                print('                - Working on level ' + str(k) + ' of ' + str(np.shape(L1_wet_grid)[0])+' ('+str(np.sum(L1_wet_grid[k, :, :] > 0))+' nonzero points found)')
            if np.any(L1_wet_grid[k, :, :] > 0):
                # take an initial stab at the interpolation
                L0_points = np.copy(all_L0_points)
                L0_values = np.reshape(L0_var[k, :], (np.size(L0_var[k, :]),))
                L0_wet_grid_vert = np.reshape(L0_wet_grid[k, :], (np.size(L0_wet_grid[k, :]),))
                L0_points = L0_points[L0_wet_grid_vert != 0, :]
                L0_values = L0_values[L0_wet_grid_vert != 0]

                # fill the zeros with a very tiny value
                L0_values[L0_values == 0] = tiny_value

                if len(L0_points)>4:
                    grid = griddata(L0_points, L0_values, (XC_subset, YC_subset), method='linear',fill_value=0)
                    # grid_nearest = griddata(L0_points, L0_values, (XC_subset, YC_subset), method='nearest', fill_value=0)
                    # grid_nearest[:,:,0]
                    if not np.any(grid!=0):
                        grid = griddata(L0_points, L0_values, (XC_subset, YC_subset), method='nearest', fill_value=0)
                else:
                    grid = np.zeros_like(XC_subset).astype(float)

                # if k==0:
                #     plt.imshow(grid,origin='lower')
                #     plt.show()

                # mask out any values which should be 0'd based on the interpolation mask
                grid[L1_interpolation_mask[k, :, :] != 1] = 0

                # in this level, spread points where they should be spread
                spread_rows, spread_cols = np.where(L1_interpolation_mask[k,:,:]==2)
                for ri in range(len(spread_rows)):
                    source_row = L1_source_rows[k,spread_rows[ri],spread_cols[ri]]
                    source_col = L1_source_cols[k, spread_rows[ri], spread_cols[ri]]
                    source_level = L1_source_levels[k, spread_rows[ri], spread_cols[ri]]
                    if source_level !=k:
                         value = full_grid[source_level,source_row,source_col]
                    else:
                        value = grid[source_row,source_col]

                    # print('        - Filling in point at location '+str(k)+','+str(spread_rows[ri])+','+str(spread_cols[ri])+\
                    #       ' with point at location '+str(source_level)+','+str(source_row)+','+str(source_col)+' (value = '+str(value)+')')
                    grid[spread_rows[ri], spread_cols[ri]] = value

                # if there's any that were filled from above, then fill em
                downward_rows, downward_cols = np.where(L1_interpolation_mask[k, :, :] == 3)
                for ri in range(len(downward_rows)):
                    source_level = L1_source_levels[k, downward_rows[ri], downward_cols[ri]]
                    value = full_grid[source_level, source_row, source_col]
                    grid[downward_rows[ri], downward_cols[ri]] = value

                # now, add the end of it, make all of the tiny values actually 0
                grid[np.abs(grid) <= 2 * tiny_value] = 0

            else:
                grid = np.zeros_like(XC_subset).astype(float)

        full_grid[k, :, :] = grid[:, :]

    return(full_grid)



def downscale_2D_points_with_zeros(L0_points, L0_var, L0_wet_grid,
                       XC_subset, YC_subset, L1_wet_grid,
                       printing=False):


    tiny_value = 1e-14

    if np.any(L1_wet_grid > 0):
        # take an initial stab at the interpolation
        L0_values = np.reshape(L0_var, (np.size(L0_var), ))
        L0_wet_grid_vert = np.reshape(L0_wet_grid, (np.size(L0_wet_grid), ))
        L0_points = L0_points[L0_wet_grid_vert != 0, :]
        L0_values = L0_values[L0_wet_grid_vert != 0]

        # fill the zeros with a very tiny value
        L0_values[L0_values == 0] = tiny_value

        if len(L0_points)>4:
            grid = griddata(L0_points, L0_values, (XC_subset, YC_subset), method='linear',fill_value=0)
            # grid = grid[:, :, 0]
            # grid_nearest = griddata(L0_points, L0_values, (XC_subset, YC_subset), method='nearest', fill_value=0)
            # grid_nearest[:,:,0]
            if not np.any(grid!=0):
                grid = griddata(L0_points, L0_values, (XC_subset, YC_subset), method='nearest', fill_value=0)
                # grid = grid[:, :, 0]
        else:
            grid = np.zeros_like(XC_subset).astype(float)

        # if k==0:
        #     plt.imshow(grid,origin='lower')
        #     plt.show()

        # mask out any values which should be 0'd based on the old bathy
        #grid[L0_wet_grid_on_L1[k, :, :] == 0] = 0

        # mask out any values which should be 0'd based on the new bathy
        grid[L1_wet_grid == 0] = 0

        # spread the the variable outward to new wet cells
        grid, n_remaining = spread_var_horizontally_in_wet_grid(grid, L1_wet_grid)

        # if, for whatever reason, there are still values to be filled, then fill em with the nearest neighbor
        if n_remaining>0:
            if len(L0_points) > 0:
                grid_nearest = griddata(L0_points, L0_values, (XC_subset, YC_subset), method='nearest', fill_value=0)
                # grid_nearest = grid_nearest[:, :, 0]
                indices = np.logical_and(grid==0,L1_wet_grid != 0)
                grid[indices] = grid_nearest[indices]

        # now, add the end of it, make all of the tiny values actually 0
        grid[np.abs(grid) <= 2 * tiny_value] = 0

    else:
        grid = np.zeros_like(XC_subset).astype(float)

    return(grid)


def downscale_3D_boundary_field(L0_XC, L0_YC, L0_var, L0_wet_grid, L0_wet_grid_on_L1,
                                XC_subset, YC_subset, L1_wet_grid,
                                mean_vertical_difference=0,fill_downward=True,
                                printing=False,remove_zeros=True):

    # full_grid = np.zeros_like(L1_wet_grid).astype(float)
    full_grid = np.zeros((np.shape(L1_wet_grid)[0],np.shape(boundary_points)[0]))

    for k in range(np.shape(L0_var)[0]):

        only_fill_downward = False
        continue_to_interpolation = True

        if continue_to_interpolation:
            if printing:
                print('    Working on level ' + str(k) + ' of ' + str(np.shape(L0_var)[0])+' ('+str(np.sum(L1_wet_grid[k, :, :] > 0))+' nonzero points found)')
            if np.any(L1_wet_grid[k, :] > 0):
                # take an initial stab at the interpolation
                L0_points = np.hstack([np.reshape(L0_XC, (np.size(L0_XC), 1)),
                                       np.reshape(L0_YC, (np.size(L0_YC), 1))])
                L0_values = np.reshape(L0_var[k, :, :], (np.size(L0_var[k, :, :]), 1))
                L0_wet_grid_vert = np.reshape(L0_wet_grid[k, :, :], (np.size(L0_wet_grid[k, :, :]), 1))
                L0_points = L0_points[L0_wet_grid_vert[:,0] != 0, :]
                L0_values = L0_values[L0_wet_grid_vert[:,0] != 0, :]
                if remove_zeros:
                    L0_points = L0_points[L0_values[:, 0] != 0, :]
                    L0_values = L0_values[L0_values[:, 0] != 0, :]

                if len(L0_points)>4:
                    grid = griddata(L0_points, L0_values, (XC_subset, YC_subset), method='linear',fill_value=0)
                    grid = grid[:, 0]
                else:
                    grid = np.zeros((np.shape(boundary_points)[0],)).astype(float)

                # if k==0:
                #     plt.imshow(grid,origin='lower')
                #     plt.show()

                # mask out any values which should be 0'd based on the old bathy
                grid[L0_wet_grid_on_L1[k, :] == 0] = 0

                # mask out any values which should be 0'd based on the new bathy
                grid[L1_wet_grid[k, :] == 0] = 0

                # # spread the the variable outward to new wet cells
                # grid, n_remaining = spread_boundary_var_horizontally_in_wet_grid(grid, L1_wet_grid[k, :])

                # # if there are still values which need to be filled, spread downward
                # if n_remaining > 0 and fill_downward and k>0:
                #     grid = spread_var_vertically_in_wet_grid(full_grid, grid, L1_wet_grid[k, :], k,mean_vertical_difference)

        if only_fill_downward:
            grid = np.zeros_like(XC_subset).astype(float)
            grid = spread_var_vertically_in_wet_grid(full_grid, grid, L1_wet_grid[k, :, :], k,
                                                     mean_vertical_difference)

        full_grid[k, :] = grid

    return(full_grid)

def downscale_3D_seaice_field(L0_XC, L0_YC, L0_var, L0_wet_grid, L0_wet_grid_on_L1,
                       XC_subset, YC_subset, L1_wet_grid,
                       mean_vertical_difference=0,fill_downward=True,printing=False):

    # full_grid = np.zeros_like(L1_wet_grid).astype(float)
    full_grid = np.zeros((np.shape(L0_var)[0],np.shape(XC_subset)[0],np.shape(XC_subset)[1]))

    for k in range(np.shape(L0_var)[0]):

        continue_to_interpolation = True

        if continue_to_interpolation:
            if printing:
                print('    Working on level ' + str(k) + ' of ' + str(np.shape(L0_var)[0])+' ('+str(np.sum(L1_wet_grid[k, :, :] > 0))+' nonzero points found)')
            if np.any(L1_wet_grid[k, :, :] > 0):
                # take an initial stab at the interpolation
                L0_points = np.hstack([np.reshape(L0_XC, (np.size(L0_XC), 1)),
                                       np.reshape(L0_YC, (np.size(L0_YC), 1))])
                L0_values = np.reshape(L0_var[k, :, :], (np.size(L0_var[k, :, :]), 1))
                L0_wet_grid_vert = np.reshape(L0_wet_grid[k, :, :], (np.size(L0_wet_grid[k, :, :]), 1))
                L0_points = L0_points[L0_wet_grid_vert[:,0] != 0, :]
                L0_values = L0_values[L0_wet_grid_vert[:,0] != 0, :]

                if len(L0_points)>4:
                    grid = griddata(L0_points, L0_values, (XC_subset, YC_subset), method='linear',fill_value=0)
                    grid = grid[:, :, 0]
                    grid_nearest = griddata(L0_points, L0_values, (XC_subset, YC_subset), method='nearest', fill_value=0)
                    grid_nearest = grid_nearest[:, :, 0]
                else:
                    grid = np.zeros_like(XC_subset).astype(float)
                    grid_nearest = np.zeros_like(XC_subset).astype(float)

                grid[L0_wet_grid_on_L1[k, :, :] == 0] = 0
                grid[L1_wet_grid[k, :, :] == 0] = 0
                indices = np.logical_and(L1_wet_grid[k, :, :] != 0, L0_wet_grid_on_L1[k, :, :] == 0)
                grid[indices] = grid_nearest[indices]

                # # make a spreading mask (1 means we still need to spread there)
                # spread_mask = np.ones_like(grid)
                # spread_mask[L1_wet_grid[k, :, :] == 1] = 0
                #
                # spread_mask[L0_wet_grid_on_L1[k, :, :] == 0] = 1
                #
                # raise ValueError('Stop')
                #
                # # mask out any values which should be 0'd based on the old bathy
                # grid[L0_wet_grid_on_L1[k, :, :] == 0] = 0
                # spread_mask[L0_wet_grid_on_L1[k, :, :] == 0] = 1
                #
                # # mask out any values which should be 0'd based on the new bathy
                # grid[L1_wet_grid[k, :, :] == 0] = 0
                #
                # # spread the the variable outward to new wet cells
                # grid, n_remaining = spread_var_horizontally_in_wet_grid_with_mask(grid, L1_wet_grid[k, :, :], spread_mask)
                # print('   - Remaining after spread: ' + str(n_remaining))
                #
                # # if there are still values which need to be filled, spread downward
                # if n_remaining > 0 and fill_downward and k>0:
                #     grid = spread_var_vertically_in_wet_grid(full_grid, grid, L1_wet_grid[k, :, :], k,mean_vertical_difference)

                full_grid[k, :, :] = grid[:, :]

    return(full_grid)

def downscale_exf_field(L0_points, L0_values, L0_wet_grid,# L0_wet_grid_on_L1,
                        XC_subset, YC_subset, L1_wet_grid,
                        remove_zeros=True):

    # take an initial stab at the interpolation
    L0_points = L0_points[L0_wet_grid != 0, :]
    L0_values = L0_values[L0_wet_grid != 0]
    if remove_zeros:
        L0_points = L0_points[L0_values != 0, :]
        L0_values = L0_values[L0_values != 0]

    if len(L0_points)>4:
        grid = griddata(L0_points, L0_values, (XC_subset, YC_subset), method='linear',fill_value=0)
    else:
        grid = np.zeros((np.shape(boundary_points)[0],)).astype(float)

    # if k==0:
    #     plt.imshow(grid,origin='lower')
    #     plt.show()

    # mask out any values which should be 0'd based on the old bathy
    # grid[L0_wet_grid_on_L1[k, :] == 0] = 0

    # mask out any values which should be 0'd based on the new bathy
    grid[L1_wet_grid == 0] = 0

    # spread the the variable outward to new wet cells
    # grid, n_remaining = spread_var_horizontally_in_wet_grid(grid, L1_wet_grid)

    return(grid)

def downscale_3D_boundary_points(L0_points, L0_values, L0_wet_grid,
                                XC_subset, YC_subset, L1_wet_grid,
                                mean_vertical_difference=0,fill_downward=True,
                                printing=False,remove_zeros=True):

    # full_grid = np.zeros_like(L1_wet_grid).astype(float)
    full_grid = np.zeros((np.shape(L1_wet_grid)[0],np.shape(XC_subset)[0], np.shape(XC_subset)[1]))
    full_interp_mask = np.zeros((np.shape(L1_wet_grid)[0], np.shape(XC_subset)[0], np.shape(XC_subset)[1]))

    for k in range(np.shape(L0_wet_grid)[0]):

        only_fill_downward = False
        continue_to_interpolation = True

        if continue_to_interpolation:
            if printing:
                print('    Working on level ' + str(k) + ' of ' + str(np.shape(L0_var)[0])+' ('+str(np.sum(L1_wet_grid[k, :, :] > 0))+' nonzero points found)')
            if np.sum(L1_wet_grid[k, :, :] > 0)>0:
                # take an initial stab at the interpolation
                L0_points_subset = np.copy(L0_points)
                L0_values_subset = L0_values[k,:]
                L0_wet_grid_vert = np.reshape(L0_wet_grid[k, :], (np.size(L0_wet_grid[k, :]), ))
                L0_points_subset = L0_points_subset[L0_wet_grid_vert != 0, :]
                L0_values_subset = L0_values_subset[L0_wet_grid_vert != 0]
                if remove_zeros:
                    L0_points_subset = L0_points_subset[L0_values_subset != 0, :]
                    L0_values_subset = L0_values_subset[L0_values_subset != 0]

                # 1 means the point has been assigned a value (or doesn't need one)
                # 0 means the point still needs a value
                interp_mask = np.copy(1 - L1_wet_grid[k, :, :])

                if len(L0_points_subset) > 0:

                    # print(len(L0_points_subset))

                    if len(L0_points_subset) > 4:
                        grid = griddata(L0_points_subset, L0_values_subset, (XC_subset, YC_subset), method='linear',fill_value=np.nan)

                        # mask out any values which should be 0'd based on the new bathy
                        grid[L1_wet_grid[k, :, :] == 0] = 0

                        # mask out values which became nans by the interpolation
                        grid[np.isnan(grid)] = 0
                        interp_mask[np.isnan(grid)] = 0
                    else:
                        grid = np.zeros(np.shape(XC_subset))

                    # check if any points should be filled by a nearest neighbor
                    if np.any(interp_mask==0):
                        # grid = griddata(L0_points, L0_values, (XC_subset, YC_subset), method='linear',fill_value=0)
                        # grid = grid[:, :, 0]
                        grid_nearest = griddata(L0_points_subset, L0_values_subset, (XC_subset, YC_subset), method='nearest', fill_value=np.nan)
                        grid[interp_mask==0] = grid_nearest[interp_mask==0]
                        interp_mask[interp_mask==0] = 1
                else:
                    grid = np.zeros(np.shape(XC_subset))
                    # print('No points found for this layer')

            else:
                grid = np.zeros((np.shape(XC_subset)[0], np.shape(XC_subset)[1])).astype(float)
                interp_mask = np.copy(1-L1_wet_grid[k, :, :])

        full_grid[k, :, :] = grid
        full_interp_mask[k,:,:] = interp_mask

    # plt.subplot(1,2,1)
    # plt.imshow(L1_wet_grid[:,0,:])
    # plt.subplot(1,2,2)
    # plt.imshow(full_interp_mask[:,0,:])
    # plt.show()

    if np.any(full_interp_mask==0):
        for k in range(1,np.shape(full_interp_mask)[0]):
            if np.any(full_interp_mask[k,:,:]==0):
                rows,cols = np.where(full_interp_mask[k, :, :]==0)
                for ri in range(len(rows)):
                    full_grid[k,rows[ri],cols[ri]] = full_grid[k-1,rows[ri],cols[ri]]

    return(full_grid)


