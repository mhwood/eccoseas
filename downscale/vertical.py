
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata, interp1d
import netCDF4 as nc4

def interpolate_var_grid_faces_to_new_depth_levels(var_grid,wet_grid,delR_in,delR_out):

    Z_bottom_in = np.cumsum(delR_in)
    Z_top_in = np.concatenate([np.array([0]), Z_bottom_in[:-1]])
    Z_in = (Z_bottom_in + Z_top_in) / 2

    Z_bottom_out = np.cumsum(delR_out)
    Z_top_out = np.concatenate([np.array([0]), Z_bottom_out[:-1]])
    Z_out = (Z_bottom_out + Z_top_out) / 2


    if len(np.shape(var_grid))==3:
        new_var_grid = np.zeros((np.size(delR_out), np.shape(var_grid)[1],np.shape(var_grid)[2]))
        new_wet_grid = np.zeros((np.size(delR_out), np.shape(var_grid)[1], np.shape(var_grid)[2]))
        for i in range(np.shape(var_grid)[1]):
            for j in range(np.shape(var_grid)[2]):
                test_profile = var_grid[:, i, j]
                if np.sum(test_profile != 0) > 1:
                    set_int_linear = interp1d(Z_in[test_profile != 0], test_profile[test_profile != 0],
                                              bounds_error=False, fill_value=np.nan)
                    new_profile = set_int_linear(Z_out)

                    new_profile[np.abs(Z_out) < np.abs(Z_in[0])] = new_profile[~np.isnan(new_profile)][0]
                    if np.size(np.abs(Z_in[test_profile == 0])) > 0:
                        first_zero_depth = np.abs(Z_in[test_profile == 0])[0]
                        bottom_value = new_profile[~np.isnan(new_profile)][-1]
                        new_profile[np.logical_and(np.isnan(new_profile), np.abs(Z_out) < first_zero_depth)] = bottom_value

                    if np.any(np.isnan(new_profile)):
                        if np.isnan(new_profile[0]):
                            raise ValueError('The surface value is nan')
                        else:
                            for k in range(1,len(new_profile)):
                                if np.isnan(new_profile[k]):
                                    new_profile[k] = new_profile[k-1]

                    new_var_grid[:, i, j] = new_profile

                if np.sum(test_profile == 0) == 1:
                    new_var_grid[0, i, j] = var_grid[0, i, j]

    elif len(np.shape(var_grid)) == 4:
        new_var_grid = np.zeros((np.shape(var_grid)[0],np.size(delR_out), np.shape(var_grid)[2], np.shape(var_grid)[3]))
        for t in range(np.shape(var_grid)[0]):
            for i in range(np.shape(var_grid)[2]):
                for j in range(np.shape(var_grid)[3]):

                    test_profile = var_grid[t, :, i, j]
                    if np.sum(test_profile != 0) > 1:
                        set_int_linear = interp1d(Z_in[test_profile != 0], test_profile[test_profile != 0],
                                                  bounds_error=False, fill_value=np.nan)
                        new_profile = set_int_linear(Z_out)

                        new_profile[np.abs(Z_out) < np.abs(Z_in[0])] = new_profile[~np.isnan(new_profile)][0]
                        if np.size(np.abs(Z_in[test_profile == 0])) > 0:
                            first_zero_depth = np.abs(Z_in[test_profile == 0])[0]
                            bottom_value = new_profile[~np.isnan(new_profile)][-1]
                            new_profile[
                                np.logical_and(np.isnan(new_profile), np.abs(Z_out) < first_zero_depth)] = bottom_value

                        if np.any(np.isnan(new_profile)):
                            if np.isnan(new_profile[0]):
                                raise ValueError('The surface value is nan')
                            else:
                                for k in range(1, len(new_profile)):
                                    if np.isnan(new_profile[k]):
                                        new_profile[k] = new_profile[k - 1]

                        new_var_grid[t, :, i, j] = new_profile

                    if np.sum(test_profile == 0) == 1:
                        new_var_grid[t, 0, i, j] = var_grid[t, 0, i, j]
    else:
        raise ValueError('The input array should be dim 3 or 4')

    if np.shape(wet_grid)[0]!=len(delR_out):
        new_wet_grid = np.zeros((np.size(delR_out), np.shape(wet_grid)[1], np.shape(wet_grid)[2]))
        for i in range(np.shape(wet_grid)[1]):
            for j in range(np.shape(wet_grid)[2]):
                test_profile = wet_grid[:, i, j]
                if np.sum(test_profile != 0) > 1:
                    set_int_linear = interp1d(Z_in[test_profile != 0], test_profile[test_profile != 0],
                                              bounds_error=False, fill_value=np.nan)
                    new_profile = set_int_linear(Z_out)

                    new_profile[np.abs(Z_out) < np.abs(Z_in[0])] = new_profile[~np.isnan(new_profile)][0]
                    if np.size(np.abs(Z_in[test_profile == 0])) > 0:
                        first_zero_depth = np.abs(Z_in[test_profile == 0])[0]
                        bottom_value = new_profile[~np.isnan(new_profile)][-1]
                        new_profile[np.logical_and(np.isnan(new_profile), np.abs(Z_out) < first_zero_depth)] = bottom_value

                    new_wet_grid[:, i, j] = new_profile

                if np.sum(test_profile == 0) == 1:
                    new_wet_grid[0, i, j] = wet_grid[0, i, j]
        new_wet_grid[np.isnan(new_wet_grid)] = 0
        new_wet_grid = np.round(new_wet_grid).astype(int)

        new_var_grid[np.isnan(new_var_grid)] = 0
    else:
        new_wet_grid = wet_grid


    return(new_var_grid,new_wet_grid)

def interpolate_var_points_timeseries_to_new_depth_levels(var_points, wet_points, delR_in,delR_out):

    Z_bottom_in = np.cumsum(delR_in)
    Z_top_in = np.concatenate([np.array([0]), Z_bottom_in[:-1]])
    Z_in = (Z_bottom_in + Z_top_in) / 2

    Z_bottom_out = np.cumsum(delR_out)
    Z_top_out = np.concatenate([np.array([0]), Z_bottom_out[:-1]])
    Z_out = (Z_bottom_out + Z_top_out) / 2

    if len(np.shape(var_points))==3:
        new_var_points = np.zeros((np.shape(var_points)[0], np.size(delR_out), np.shape(var_points)[2]))
        for j in range(np.shape(var_points)[2]):
            for t in range(np.shape(var_points)[0]):
                test_profile = var_points[t, :, j]
                if np.sum(test_profile != 0) > 1:
                    set_int_linear = interp1d(Z_in[test_profile != 0], test_profile[test_profile != 0],
                                              bounds_error=False, fill_value=np.nan)
                    new_profile = set_int_linear(Z_out)

                    new_profile[np.abs(Z_out) < np.abs(Z_in[0])] = new_profile[~np.isnan(new_profile)][0]
                    if np.size(np.abs(Z_in[test_profile == 0])) > 0:
                        first_zero_depth = np.abs(Z_in[test_profile == 0])[0]
                        bottom_value = new_profile[~np.isnan(new_profile)][-1]
                        new_profile[np.logical_and(np.isnan(new_profile), np.abs(Z_out) < first_zero_depth)] = bottom_value

                    new_var_points[t, :, j] = new_profile

                if np.sum(test_profile == 0) == 1:
                    new_var_points[t, 0, j] = var_points[t, 0, j]
    else:
        raise ValueError('The input array should be dim 3')

    if np.shape(wet_points)[0]!=len(delR_out):
        new_wet_points = np.zeros((np.size(delR_out), np.shape(wet_points)[1]))
        for j in range(np.shape(wet_points)[1]):
            test_profile = wet_points[:, j]
            if np.sum(test_profile != 0) > 1:
                set_int_linear = interp1d(Z_in[test_profile != 0], test_profile[test_profile != 0],
                                          bounds_error=False, fill_value=np.nan)
                new_profile = set_int_linear(Z_out)

                new_profile[np.abs(Z_out) < np.abs(Z_in[0])] = new_profile[~np.isnan(new_profile)][0]
                if np.size(np.abs(Z_in[test_profile == 0])) > 0:
                    first_zero_depth = np.abs(Z_in[test_profile == 0])[0]
                    bottom_value = new_profile[~np.isnan(new_profile)][-1]
                    new_profile[np.logical_and(np.isnan(new_profile), np.abs(Z_out) < first_zero_depth)] = bottom_value

                new_wet_points[:, j] = new_profile

            if np.sum(test_profile == 0) == 1:
                new_wet_points[0, j] = wet_points[0, j]
        new_wet_points[np.isnan(new_wet_points)] = 0
        new_wet_points = np.round(new_wet_points).astype(int)

        new_var_points[np.isnan(new_var_points)] = 0
    else:
        new_wet_points = wet_points


    return(new_var_points,new_wet_points)
