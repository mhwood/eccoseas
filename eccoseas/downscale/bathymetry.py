import numpy as np

def generate_connected_mask(start_row, start_col, wet_grid):

    if wet_grid[start_row,start_col]==0:
        raise ValueError(' The start row/col location is  dry')

    rows = np.arange(np.shape(wet_grid)[0])
    cols = np.arange(np.shape(wet_grid)[1])
    Cols,Rows = np.meshgrid(cols,rows)

    mask_grid = 1-np.copy(wet_grid)
    mask_grid[start_row,start_col] = 2
    # in the mask, 0 means unverified
    # 1 is verified dry
    # 2 is verified wet

    # plt.imshow(mask_grid)
    # plt.show()

    is_remaining = np.logical_and(mask_grid==0,wet_grid==1)
    n_remaining = np.sum(is_remaining)
    # print(n_remaining)
    continue_iter = True
    for i in range(n_remaining):
        if continue_iter:
            # get the wet rows, cols, and their current mask values
            Wet_Rows = Rows[wet_grid == 1]
            Wet_Cols = Cols[wet_grid == 1]
            Mask_Vals = mask_grid[wet_grid == 1]

            # reduce these to the ones that havent been verified yet
            Wet_Rows = Wet_Rows[Mask_Vals == 0]
            Wet_Cols = Wet_Cols[Mask_Vals == 0]
            Mask_Vals = Mask_Vals[Mask_Vals == 0]

            if len(Mask_Vals)>0:

                # for each row/col, see if its connected to one we've verified is connected
                rows_remaining,cols_remaining = np.where(is_remaining)
                for ri in range(n_remaining):
                    row = rows_remaining[ri]
                    col = cols_remaining[ri]

                    # # this bit allows for diagonal spreading
                    # row_col_dist = ((Wet_Rows.astype(float)-row)**2 + (Wet_Cols.astype(float)-col)**2)**0.5
                    # closest_index = np.argmin(row_col_dist)
                    # if row_col_dist[closest_index]<np.sqrt(2):
                    #     var_grid[row,col] = Wet_Vals[closest_index]

                    # this bit allows for only up/dow/left/right spreading
                    if row<np.shape(wet_grid)[0]-1:
                        if mask_grid[row+1,col] == 2:
                            mask_grid[row,col] = 2
                    if row > 0:
                        if mask_grid[row - 1, col] == 2:
                            mask_grid[row,col] = 2
                    if col<np.shape(wet_grid)[1]-1:
                        if mask_grid[row,col+1] == 2:
                            mask_grid[row,col] = 2
                    if col > 0:
                        if mask_grid[row, col-1] == 2:
                            mask_grid[row,col] = 2

                is_remaining = np.logical_and(mask_grid == 0, wet_grid == 1)
                n_remaining_now = np.sum(is_remaining)

                # plt.subplot(1,2,1)
                # plt.imshow(wet_grid,cmap='Greys_r')
                # plt.subplot(1, 2, 2)
                # plt.imshow(mask_grid)
                # plt.show()

                if n_remaining_now<n_remaining:
                    n_remaining = n_remaining_now
                else:
                    n_remaining = n_remaining_now
                    continue_iter=False
            else:
                continue_iter = False

    return(mask_grid)

def fill_unconnected_model_regions(bathymetry_grid, central_wet_row, central_wet_col):

    wet_grid = np.copy(bathymetry_grid)
    wet_grid[wet_grid >=0 ] = 0
    wet_grid[wet_grid<0] = 1

    mask_grid = generate_connected_mask(central_wet_row, central_wet_col, wet_grid)

    bathymetry_grid[mask_grid == 0] = 0

    return(bathymetry_grid)


