import numpy as np

def generate_connected_mask(start_row, start_col, wet_grid):
    """
    Generate a mask identifying wet regions connected to a specified starting point.

    Parameters:
    start_row (int): Row index of the starting wet cell.
    start_col (int): Column index of the starting wet cell.
    wet_grid (ndarray): 2D numpy array where 1 represents wet cells and 0 represents dry cells.

    Returns:
    ndarray: A mask grid where
             0 = unverified wet (disconnected),
             1 = dry,
             2 = verified wet (connected to the start location).

    Raises:
    ValueError: If the starting location is a dry cell.
    """

    if wet_grid[start_row, start_col] == 0:
        raise ValueError('The start row/col location is dry')

    # Generate coordinate grids
    rows = np.arange(np.shape(wet_grid)[0])
    cols = np.arange(np.shape(wet_grid)[1])
    Cols, Rows = np.meshgrid(cols, rows)

    # Initialize the mask: 1 for dry, 0 for unverified wet
    mask_grid = 1 - np.copy(wet_grid)
    mask_grid[start_row, start_col] = 2  # Mark the starting cell as verified wet

    # Boolean array indicating unverified wet cells
    is_remaining = np.logical_and(mask_grid == 0, wet_grid == 1)
    n_remaining = np.sum(is_remaining)
    continue_iter = True

    for i in range(n_remaining):
        if continue_iter:
            # Get coordinates of current unverified wet cells
            Wet_Rows = Rows[wet_grid == 1]
            Wet_Cols = Cols[wet_grid == 1]
            Mask_Vals = mask_grid[wet_grid == 1]

            # Focus on unverified wet cells
            Wet_Rows = Wet_Rows[Mask_Vals == 0]
            Wet_Cols = Wet_Cols[Mask_Vals == 0]
            Mask_Vals = Mask_Vals[Mask_Vals == 0]

            if len(Mask_Vals) > 0:
                rows_remaining, cols_remaining = np.where(is_remaining)
                for ri in range(n_remaining):
                    row = rows_remaining[ri]
                    col = cols_remaining[ri]

                    # Check connectivity in 4 directions: up, down, left, right
                    if row < np.shape(wet_grid)[0] - 1 and mask_grid[row + 1, col] == 2:
                        mask_grid[row, col] = 2
                    if row > 0 and mask_grid[row - 1, col] == 2:
                        mask_grid[row, col] = 2
                    if col < np.shape(wet_grid)[1] - 1 and mask_grid[row, col + 1] == 2:
                        mask_grid[row, col] = 2
                    if col > 0 and mask_grid[row, col - 1] == 2:
                        mask_grid[row, col] = 2

                # Update remaining unverified wet cells
                is_remaining = np.logical_and(mask_grid == 0, wet_grid == 1)
                n_remaining_now = np.sum(is_remaining)

                # Stop if no further cells have been marked
                if n_remaining_now < n_remaining:
                    n_remaining = n_remaining_now
                else:
                    n_remaining = n_remaining_now
                    continue_iter = False
            else:
                continue_iter = False

    return mask_grid

def fill_unconnected_model_regions(bathymetry_grid, central_wet_row, central_wet_col):
    """
    Fill (set to 0) all unconnected wet regions in a bathymetry grid.

    Parameters:
    bathymetry_grid (ndarray): 2D numpy array representing bathymetry values. Negative = wet, non-negative = dry.
    central_wet_row (int): Row index of a known valid central wet cell.
    central_wet_col (int): Column index of a known valid central wet cell.

    Returns:
    ndarray: Modified bathymetry grid with unconnected wet cells filled (set to 0).
    """

    # Create binary wet/dry grid from bathymetry
    wet_grid = np.copy(bathymetry_grid)
    wet_grid[wet_grid >= 0] = 0
    wet_grid[wet_grid < 0] = 1

    # Get the connectivity mask
    mask_grid = generate_connected_mask(central_wet_row, central_wet_col, wet_grid)

    # Set unconnected wet areas (mask=0) to 0 in the original grid
    bathymetry_grid[mask_grid == 0] = 0

    return bathymetry_grid
