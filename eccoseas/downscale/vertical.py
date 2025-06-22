import numpy as np
from scipy.interpolate import griddata, interp1d


def interpolate_var_grid_faces_to_new_depth_levels(var_grid, wet_grid, delR_in, delR_out):
    """
    Interpolates a 3D or 4D variable grid from input depth levels to a new set of output depth levels.

    Parameters:
    - var_grid (ndarray): The input variable grid (3D: depth x lat x lon, or 4D: time x depth x lat x lon).
    - wet_grid (ndarray): The wet mask grid matching the original vertical grid.
    - delR_in (ndarray): The thickness of original vertical levels.
    - delR_out (ndarray): The thickness of new vertical levels.

    Returns:
    - new_var_grid (ndarray): Interpolated variable grid at new depth levels.
    - new_wet_grid (ndarray): Interpolated wet mask at new depth levels.
    """

    # Calculate original depth midpoints
    Z_bottom_in = np.cumsum(delR_in)
    Z_top_in = np.concatenate([np.array([0]), Z_bottom_in[:-1]])
    Z_in = (Z_bottom_in + Z_top_in) / 2

    # Calculate target depth midpoints
    Z_bottom_out = np.cumsum(delR_out)
    Z_top_out = np.concatenate([np.array([0]), Z_bottom_out[:-1]])
    Z_out = (Z_bottom_out + Z_top_out) / 2

    if len(np.shape(var_grid)) == 2:
        # Handle 2D case (e.g., raveled grid)
        new_var_grid = np.zeros((len(delR_out), var_grid.shape[1]))
        new_wet_grid = np.zeros((len(delR_out), var_grid.shape[1]))

        for i in range(var_grid.shape[1]):
            test_profile = var_grid[:, i]

            if np.sum(test_profile != 0) > 1:
                set_int_linear = interp1d(Z_in[test_profile != 0], test_profile[test_profile != 0],
                                          bounds_error=False, fill_value=np.nan)
                new_profile = set_int_linear(Z_out)

                # Fill values above first valid depth with surface value
                new_profile[np.abs(Z_out) < np.abs(Z_in[0])] = new_profile[~np.isnan(new_profile)][0]

                if np.size(np.abs(Z_in[test_profile == 0])) > 0:
                    first_zero_depth = np.abs(Z_in[test_profile == 0])[0]
                    bottom_value = new_profile[~np.isnan(new_profile)][-1]
                    new_profile[
                        np.logical_and(np.isnan(new_profile), np.abs(Z_out) < first_zero_depth)] = bottom_value

                # Fill remaining NaNs downward
                if np.any(np.isnan(new_profile)):
                    if np.isnan(new_profile[0]):
                        raise ValueError('The surface value is nan')
                    for k in range(1, len(new_profile)):
                        if np.isnan(new_profile[k]):
                            new_profile[k] = new_profile[k - 1]

                new_var_grid[:, i] = new_profile

            elif np.sum(test_profile == 0) == 1:
                new_var_grid[0, i] = var_grid[0, i]

    elif len(np.shape(var_grid)) == 3:
        # Handle 3D case
        new_var_grid = np.zeros((len(delR_out), var_grid.shape[1], var_grid.shape[2]))
        new_wet_grid = np.zeros((len(delR_out), var_grid.shape[1], var_grid.shape[2]))

        for i in range(var_grid.shape[1]):
            for j in range(var_grid.shape[2]):
                test_profile = var_grid[:, i, j]

                if np.sum(test_profile != 0) > 1:
                    # Interpolate only on valid (non-zero) data
                    set_int_linear = interp1d(Z_in[test_profile != 0], test_profile[test_profile != 0],
                                              bounds_error=False, fill_value=np.nan)
                    new_profile = set_int_linear(Z_out)

                    # Fill values above first valid depth with surface value
                    new_profile[np.abs(Z_out) < np.abs(Z_in[0])] = new_profile[~np.isnan(new_profile)][0]

                    if np.size(np.abs(Z_in[test_profile == 0])) > 0:
                        first_zero_depth = np.abs(Z_in[test_profile == 0])[0]
                        bottom_value = new_profile[~np.isnan(new_profile)][-1]
                        new_profile[
                            np.logical_and(np.isnan(new_profile), np.abs(Z_out) < first_zero_depth)] = bottom_value

                    # Fill remaining NaNs downward
                    if np.any(np.isnan(new_profile)):
                        if np.isnan(new_profile[0]):
                            raise ValueError('The surface value is nan')
                        for k in range(1, len(new_profile)):
                            if np.isnan(new_profile[k]):
                                new_profile[k] = new_profile[k - 1]

                    new_var_grid[:, i, j] = new_profile

                elif np.sum(test_profile == 0) == 1:
                    new_var_grid[0, i, j] = var_grid[0, i, j]

    elif len(np.shape(var_grid)) == 4:
        # Handle 4D case (e.g., time-dependent)
        new_var_grid = np.zeros((var_grid.shape[0], len(delR_out), var_grid.shape[2], var_grid.shape[3]))

        for t in range(var_grid.shape[0]):
            for i in range(var_grid.shape[2]):
                for j in range(var_grid.shape[3]):
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
                            for k in range(1, len(new_profile)):
                                if np.isnan(new_profile[k]):
                                    new_profile[k] = new_profile[k - 1]

                        new_var_grid[t, :, i, j] = new_profile

                    elif np.sum(test_profile == 0) == 1:
                        new_var_grid[t, 0, i, j] = var_grid[t, 0, i, j]
    else:
        raise ValueError('The input array should be dim 2 or 3 or 4')


    # Handle wet grid interpolation if needed
    if wet_grid.shape[0] != len(delR_out):

        if len(np.shape(wet_grid)) == 2:
            new_wet_grid = np.zeros((len(delR_out), wet_grid.shape[1]))

            for i in range(wet_grid.shape[1]):
                test_profile = wet_grid[:, i]

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

                    new_wet_grid[:, i] = new_profile

                elif np.sum(test_profile == 0) == 1:
                    new_wet_grid[0, i] = wet_grid[0, i]

            # Clean up interpolated wet grid
            new_wet_grid[np.isnan(new_wet_grid)] = 0
            new_wet_grid = np.round(new_wet_grid).astype(int)

            # Replace NaNs in var grid
            new_var_grid[np.isnan(new_var_grid)] = 0
        elif len(np.shape(wet_grid)) == 3:
            new_wet_grid = np.zeros((len(delR_out), wet_grid.shape[1], wet_grid.shape[2]))

            for i in range(wet_grid.shape[1]):
                for j in range(wet_grid.shape[2]):
                    test_profile = wet_grid[:, i, j]

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

                        new_wet_grid[:, i, j] = new_profile

                    elif np.sum(test_profile == 0) == 1:
                        new_wet_grid[0, i, j] = wet_grid[0, i, j]

            # Clean up interpolated wet grid
            new_wet_grid[np.isnan(new_wet_grid)] = 0
            new_wet_grid = np.round(new_wet_grid).astype(int)

            # Replace NaNs in var grid
            new_var_grid[np.isnan(new_var_grid)] = 0
    else:
        new_wet_grid = wet_grid

    return new_var_grid, new_wet_grid


def interpolate_var_points_timeseries_to_new_depth_levels(var_points, wet_points, delR_in, delR_out):
    """
    Interpolates variable time series data at specific point locations to new depth levels.

    Parameters:
    - var_points (ndarray): Time series data (time x depth x point).
    - wet_points (ndarray): Corresponding wet/dry mask (depth x point).
    - delR_in (ndarray): Original depth thickness array.
    - delR_out (ndarray): Target depth thickness array.

    Returns:
    - new_var_points (ndarray): Interpolated variable data on new depth levels.
    - new_wet_points (ndarray): Interpolated wet mask on new depth levels.
    """

    Z_bottom_in = np.cumsum(delR_in)
    Z_top_in = np.concatenate([[0], Z_bottom_in[:-1]])
    Z_in = (Z_bottom_in + Z_top_in) / 2

    Z_bottom_out = np.cumsum(delR_out)
    Z_top_out = np.concatenate([[0], Z_bottom_out[:-1]])
    Z_out = (Z_bottom_out + Z_top_out) / 2

    if var_points.ndim != 3:
        raise ValueError('The input array should be dim 3')

    new_var_points = np.zeros((var_points.shape[0], len(delR_out), var_points.shape[2]))

    for j in range(var_points.shape[2]):
        for t in range(var_points.shape[0]):
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

            elif np.sum(test_profile == 0) == 1:
                new_var_points[t, 0, j] = var_points[t, 0, j]

    if wet_points.shape[0] != len(delR_out):
        new_wet_points = np.zeros((len(delR_out), wet_points.shape[1]))

        for j in range(wet_points.shape[1]):
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

            elif np.sum(test_profile == 0) == 1:
                new_wet_points[0, j] = wet_points[0, j]

        new_wet_points[np.isnan(new_wet_points)] = 0
        new_wet_points = np.round(new_wet_points).astype(int)
        new_var_points[np.isnan(new_var_points)] = 0
    else:
        new_wet_points = wet_points

    return new_var_points, new_wet_points

