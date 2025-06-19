
import os
import numpy as np
from scipy.interpolate import griddata, interp1d
#import torch
from numba import njit, prange
from scipy.spatial import Delaunay

###############################################################################################
# Spreading routines

def spread_var_horizontally_in_wet_grid_legacy(var_grid, wet_grid):
    """
    Fill zero values in var_grid using the nearest non-zero neighbors within wet areas.
    Legacy implementation using NumPy.

    Parameters:
        var_grid (ndarray): 2D array with variable data, where zeros represent missing values.
        wet_grid (ndarray): 2D mask where 1 indicates wet cells and 0 indicates dry.

    Returns:
        tuple: Updated var_grid and number of remaining unfilled wet cells.
    """
    rows = np.arange(np.shape(var_grid)[0])
    cols = np.arange(np.shape(var_grid)[1])
    Cols, Rows = np.meshgrid(cols, rows)

    is_remaining = np.logical_and(var_grid == 0, wet_grid == 1)
    n_remaining = np.sum(is_remaining)
    continue_iter = True

    for _ in range(n_remaining):
        if continue_iter:
            Wet_Rows = Rows[wet_grid == 1]
            Wet_Cols = Cols[wet_grid == 1]
            Wet_Vals = var_grid[wet_grid == 1]
            Wet_Rows = Wet_Rows[Wet_Vals != 0]
            Wet_Cols = Wet_Cols[Wet_Vals != 0]
            Wet_Vals = Wet_Vals[Wet_Vals != 0]

            if len(Wet_Vals) > 0:
                rows_remaining, cols_remaining = np.where(is_remaining)
                for ri in range(n_remaining):
                    row = rows_remaining[ri]
                    col = cols_remaining[ri]
                    row_col_dist = ((Wet_Rows.astype(float) - row) ** 2 + (Wet_Cols.astype(float) - col) ** 2) ** 0.5
                    closest_index = np.argmin(row_col_dist)
                    if row_col_dist[closest_index] < np.sqrt(2):
                        var_grid[row, col] = Wet_Vals[closest_index]

                is_remaining = np.logical_and(var_grid == 0, wet_grid == 1)
                n_remaining_now = np.sum(is_remaining)
                if n_remaining_now < n_remaining:
                    n_remaining = n_remaining_now
                else:
                    continue_iter = False
            else:
                continue_iter = False

    return var_grid, n_remaining


@njit(parallel=True)
def spread_var_horizontally_in_wet_grid(var_grid, wet_grid):
    var_grid = var_grid.astype(np.float32)
    wet_grid = wet_grid.astype(np.uint8)

    H, W = var_grid.shape

    # Flatten coordinate grids
    total = H * W
    Rows = np.empty(total, dtype=np.int32)
    Cols = np.empty(total, dtype=np.int32)
    for i in range(H):
        for j in range(W):
            idx = i * W + j
            Rows[idx] = i
            Cols[idx] = j

    # Initial remaining mask
    is_remaining = (var_grid == 0) & (wet_grid == 1)
    n_remaining = np.sum(is_remaining)
    continue_iter = True

    while continue_iter:
        # Flatten wet_mask and get flat indices
        wet_mask = (wet_grid == 1).flatten()
        val_mask = (var_grid.flatten() != 0) & wet_mask
        idxs = np.where(val_mask)[0]

        if idxs.size == 0:
            break

        Wet_Rows = Rows[idxs]
        Wet_Cols = Cols[idxs]
        Wet_Vals = var_grid.flatten()[idxs]

        rem_rows, rem_cols = np.where((var_grid == 0) & (wet_grid == 1))
        n_rem = rem_rows.shape[0]

        if n_rem == 0:
            break

        # Parallel loop over all remaining cells
        for idx in prange(n_rem):
            r = rem_rows[idx]
            c = rem_cols[idx]
            min_dist = 1e6
            min_val = 0.0

            for j in range(Wet_Vals.size):
                dr = float(Wet_Rows[j]) - r
                dc = float(Wet_Cols[j]) - c
                dist = (dr * dr + dc * dc) ** 0.5
                if dist < min_dist:
                    min_dist = dist
                    min_val = Wet_Vals[j]

            if min_dist < 2 ** 0.5:
                var_grid[r, c] = min_val

        # Recompute remaining
        is_remaining_new = (var_grid == 0) & (wet_grid == 1)
        n_remaining_new = np.sum(is_remaining_new)

        if n_remaining_new < n_remaining:
            n_remaining = n_remaining_new
        else:
            continue_iter = False

    return var_grid, n_remaining

def spread_var_horizontally_in_wet_grid_torch(var_grid_np, wet_grid_np):
    """
    GPU-accelerated version of horizontal spreading using PyTorch.
    Fills zero cells in var_grid_np by copying nearest valid values from neighboring wet cells.

    Parameters:
        var_grid_np (ndarray): 2D array with variable data (zero represents missing values).
        wet_grid_np (ndarray): 2D array indicating wet areas (1 for wet, 0 for dry).

    Returns:
        tuple: (filled variable grid as ndarray, number of unfilled cells after process).
    """
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    var_grid = torch.tensor(var_grid_np, dtype=torch.float32, device=device)
    wet_grid = torch.tensor(wet_grid_np, dtype=torch.float32, device=device)

    H, W = var_grid.shape
    rows = torch.arange(H, device=device)
    cols = torch.arange(W, device=device)
    Cols, Rows = torch.meshgrid(cols, rows, indexing='xy')
    Cols = Cols.contiguous()
    Rows = Rows.contiguous()

    is_remaining = (var_grid == 0) & (wet_grid == 1)
    n_remaining = is_remaining.sum().item()
    continue_iter = True

    while continue_iter:
        Wet_Rows = Rows[wet_grid == 1]
        Wet_Cols = Cols[wet_grid == 1]
        Wet_Vals = var_grid[wet_grid == 1]

        valid_mask = Wet_Vals != 0
        Wet_Rows = Wet_Rows[valid_mask]
        Wet_Cols = Wet_Cols[valid_mask]
        Wet_Vals = Wet_Vals[valid_mask]

        if Wet_Vals.numel() == 0:
            break

        rows_remaining, cols_remaining = torch.where(is_remaining)
        if rows_remaining.numel() == 0:
            break

        # Expand dimensions for broadcasting
        target_r = rows_remaining.unsqueeze(1).float()
        target_c = cols_remaining.unsqueeze(1).float()
        source_r = Wet_Rows.unsqueeze(0).float()
        source_c = Wet_Cols.unsqueeze(0).float()

        # Compute distances
        dists = torch.sqrt((source_r - target_r) ** 2 + (source_c - target_c) ** 2)
        min_dists, min_indices = torch.min(dists, dim=1)

        # Only update if within sqrt(2)
        close_mask = min_dists < (2 ** 0.5)
        update_rows = rows_remaining[close_mask]
        update_cols = cols_remaining[close_mask]
        update_vals = Wet_Vals[min_indices[close_mask]]

        var_grid[update_rows, update_cols] = update_vals

        # Update remaining mask
        new_is_remaining = (var_grid == 0) & (wet_grid == 1)
        new_n_remaining = new_is_remaining.sum().item()

        if new_n_remaining < n_remaining:
            n_remaining = new_n_remaining
            is_remaining = new_is_remaining
        else:
            continue_iter = False

    return var_grid.cpu().numpy(), n_remaining

###############################################################################################
# 2D routines

def downscale_2D_field(L0_XC, L0_YC, L0_var, L0_wet_grid, L0_wet_grid_on_L1,
                       XC_subset, YC_subset, L1_wet_grid, spread_horizontally=True, remove_zeros=True):
    """
    Downscale a 2D variable field from a coarse to a fine grid using linear interpolation,
    optional spreading, and wet grid masking.

    Parameters:
        L0_XC, L0_YC (ndarray): Coordinates for the coarse grid.
        L0_var (ndarray): Variable values on the coarse grid.
        L0_wet_grid (ndarray): Wet mask for coarse grid.
        L0_wet_grid_on_L1 (ndarray): Coarse wet mask regridded to fine grid.
        XC_subset, YC_subset (ndarray): Coordinates for the fine grid.
        L1_wet_grid (ndarray): Wet mask for the fine grid.
        spread_horizontally (bool): Whether to use horizontal spreading for filling missing values.
        remove_zeros (bool): Whether to ignore zero values during interpolation.

    Returns:
        ndarray: Interpolated and optionally filled 2D variable field.
    """
    L0_points = np.hstack([np.reshape(L0_XC, (np.size(L0_XC), 1)),
                           np.reshape(L0_YC, (np.size(L0_YC), 1))])
    L0_values = np.reshape(L0_var, (np.size(L0_var), 1))

    L0_wet_grid = np.reshape(L0_wet_grid, (np.size(L0_wet_grid), 1))
    if remove_zeros:
        L0_points = L0_points[L0_wet_grid[:, 0] != 0, :]
        L0_values = L0_values[L0_wet_grid[:, 0] != 0, :]

    grid = griddata(L0_points, L0_values, (XC_subset, YC_subset), method='linear', fill_value=0)
    grid = grid[:, :, 0]

    grid[L0_wet_grid_on_L1 == 0] = 0
    grid[L1_wet_grid == 0] = 0

    if spread_horizontally:
        grid, _ = spread_var_horizontally_in_wet_grid(grid, L1_wet_grid)

    return grid

def downscale_2D_points_with_zeros(L0_points, L0_var, L0_wet_grid,
                                    XC_subset, YC_subset, L1_wet_grid,
                                    printing=False):
    """
    Downscale a 2D variable field from scattered data points to a finer grid with zero handling.

    Parameters:
        L0_points (ndarray): Coordinates of the original scattered points.
        L0_var (ndarray): Corresponding variable values.
        L0_wet_grid (ndarray): Wet mask at original resolution.
        XC_subset, YC_subset (ndarray): Target grid coordinates.
        L1_wet_grid (ndarray): Wet mask for the target grid.
        printing (bool): If True, enable debug visualization.

    Returns:
        ndarray: Interpolated and filled field over the fine grid.
    """
    tiny_value = 1e-14

    if np.any(L1_wet_grid > 0):
        L0_values = np.reshape(L0_var, (np.size(L0_var), ))
        L0_wet_grid_vert = np.reshape(L0_wet_grid, (np.size(L0_wet_grid), ))
        L0_points = L0_points[L0_wet_grid_vert != 0, :]
        L0_values = L0_values[L0_wet_grid_vert != 0]
        L0_values[L0_values == 0] = tiny_value

        if len(L0_points) > 4:
            grid = griddata(L0_points, L0_values, (XC_subset, YC_subset), method='linear', fill_value=0)
            if not np.any(grid != 0):
                grid = griddata(L0_points, L0_values, (XC_subset, YC_subset), method='nearest', fill_value=0)
        else:
            grid = np.zeros_like(XC_subset).astype(float)

        grid[L1_wet_grid == 0] = 0
        grid, n_remaining = spread_var_horizontally_in_wet_grid(grid, L1_wet_grid)

        if n_remaining > 0:
            if len(L0_points) > 0:
                grid_nearest = griddata(L0_points, L0_values, (XC_subset, YC_subset), method='nearest', fill_value=0)
                indices = np.logical_and(grid == 0, L1_wet_grid != 0)
                grid[indices] = grid_nearest[indices]

        grid[np.abs(grid) <= 2 * tiny_value] = 0
    else:
        grid = np.zeros_like(XC_subset).astype(float)

    return grid

def downscale_exf_field(L0_points, exf_grid,
                        XC_subset, YC_subset, L1_wet_grid, tri=None, printing=False):
    """
    Interpolates a forcing field to a fine grid using Delaunay-based linear interpolation.

    Parameters:
        L0_points (ndarray): Coordinates of the original data (n_points, 2).
        L0_values (ndarray): Corresponding values (n_points,).
        L0_wet_grid (ndarray): Wet mask for the original field.
        XC_subset, YC_subset (ndarray): Target grid coordinates.
        L1_wet_grid (ndarray): Wet mask on the fine grid.

    Returns:
        ndarray: Interpolated field with invalid regions masked.
    """

    def interpolate_timestep(L0_values, vertices, bary_coords):
        # Perform interpolation
        interpolated = np.zeros(query_points.shape[0], dtype=float)
        interpolated[valid] = np.einsum('ij,ij->i', L0_values[vertices], bary_coords)
        # Reshape
        grid = interpolated.reshape(XC_subset.shape)
        return(grid)

    # Compute triangulation only if not provided
    if tri is None:
        if printing:
            print('       - Computing the Delunay triangulation for interpolation')
        tri = Delaunay(L0_points)

    # create a grid of zeros to fill in
    interpolated_grid = np.zeros((np.shape(exf_grid)[0], np.shape(XC_subset)[0], np.shape(XC_subset)[1]))

    # Prepare query points
    query_points = np.vstack([XC_subset.ravel(), YC_subset.ravel()]).T
    simplices = tri.find_simplex(query_points)
    valid = simplices >= 0

    # Get simplex vertex indices and transform matrices
    vertices = tri.simplices[simplices[valid]]
    T = tri.transform[simplices[valid], :2]
    delta = query_points[valid] - tri.transform[simplices[valid], 2]
    bary = np.einsum('ijk,ik->ij', T, delta)
    bary_coords = np.c_[bary, 1 - bary.sum(axis=1)]

    # loop through the timesteps and apply the interpolation
    for i in range(np.shape(interpolated_grid)[0]):
        if printing:
            if i%100==0:
                print('       - Working on timestep '+str(i)+' of '+str(np.shape(interpolated_grid)[0]))
        ecco_values = exf_grid[i, :, :].ravel()
        timestep_grid = interpolate_timestep(ecco_values, vertices, bary_coords)
        timestep_grid[L1_wet_grid==0]=0
        interpolated_grid[i,:,:] = timestep_grid

    return interpolated_grid, tri

###############################################################################################
# 3D routines

def downscale_3D_field(L0_XC, L0_YC, L0_var, L0_wet_grid, L0_wet_grid_on_L1,
                       XC_subset, YC_subset, L1_wet_grid,
                       mean_vertical_difference=0, fill_downward=True,
                       printing=False, remove_zeros=True, testing=False):
    """
    Downscale a 3D field from coarse to fine grid with optional vertical spreading and zero handling.

    Parameters:
        L0_XC, L0_YC (ndarray): Coordinates for coarse grid.
        L0_var (ndarray): 3D variable values [levels, y, x].
        L0_wet_grid (ndarray): Wet mask for L0 grid.
        L0_wet_grid_on_L1 (ndarray): L0 wet grid remapped onto L1 resolution.
        XC_subset, YC_subset (ndarray): Fine grid coordinates.
        L1_wet_grid (ndarray): Fine resolution wet mask [levels, y, x].
        mean_vertical_difference (float): Value to add during vertical fill.
        fill_downward (bool): Whether to apply vertical spreading.
        printing (bool): Enable status output per level.
        remove_zeros (bool): Remove zero-values before interpolation.
        testing (bool): Run only first vertical level for testing.

    Returns:
        ndarray: Downscaled 3D field [levels, y, x].
    """
    full_grid = np.zeros((L1_wet_grid.shape[0], XC_subset.shape[0], XC_subset.shape[1]))
    K = 1 if testing else L1_wet_grid.shape[0]

    for k in range(K):
        if printing:
            print(f"                - Working on level {k} of {L1_wet_grid.shape[0]} ({np.sum(L1_wet_grid[k] > 0)} nonzero points found)")

        if np.any(L1_wet_grid[k] > 0):
            # Flatten inputs
            L0_points = np.hstack([L0_XC.reshape(-1, 1), L0_YC.reshape(-1, 1)])
            L0_values = L0_var[k].reshape(-1, 1)
            L0_mask = L0_wet_grid[k].reshape(-1, 1)

            L0_points = L0_points[L0_mask[:, 0] != 0]
            L0_values = L0_values[L0_mask[:, 0] != 0]
            if remove_zeros:
                L0_points = L0_points[L0_values[:, 0] != 0]
                L0_values = L0_values[L0_values[:, 0] != 0]

            if len(L0_points) > 4:
                grid = griddata(L0_points, L0_values, (XC_subset, YC_subset), method='linear', fill_value=0)[:, :, 0]
                if not np.any(grid != 0):
                    grid = griddata(L0_points, L0_values, (XC_subset, YC_subset), method='nearest', fill_value=0)[:, :, 0]
            else:
                grid = np.zeros_like(XC_subset).astype(float)

            grid[L0_wet_grid_on_L1[k] == 0] = 0
            grid[L1_wet_grid[k] == 0] = 0

            # Horizontal spreading
            grid, n_remaining = spread_var_horizontally_in_wet_grid(grid, L1_wet_grid[k])

            # Vertical spreading if needed
            if n_remaining > 0 and fill_downward and k > 0:
                grid = full_grid[k - 1] + mean_vertical_difference

            # Nearest fallback
            if n_remaining > 0 and len(L0_points) > 0:
                grid_nearest = griddata(L0_points, L0_values, (XC_subset, YC_subset), method='nearest', fill_value=0)[:, :, 0]
                grid[grid == 0] = grid_nearest[grid == 0]
        else:
            grid = np.zeros_like(XC_subset).astype(float)

        full_grid[k] = grid

    return full_grid

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


