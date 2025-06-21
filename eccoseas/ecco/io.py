
import os
import numpy as np
import netCDF4 as nc4


def read_ecco_field_to_faces(file_path, llc, dim, Nr,
                             dtype='>f4'):

    grid_array = np.fromfile(file_path,dtype=dtype)
    N = 13*llc*llc

    field_faces = {}

    if dim == 2:
        points_counted = 0
        for i in range(1, 6):
            if i < 3:
                n_points = 3 * llc * llc
                grid = grid_array[points_counted:points_counted + n_points]
                grid = np.reshape(grid, (3 * llc, llc))
            if i == 3:
                n_points = llc * llc
                grid = grid_array[points_counted:points_counted + n_points]
                grid = np.reshape(grid, (llc, llc))
            if i > 3:
                n_points = 3 * llc * llc
                grid = grid_array[points_counted:points_counted + n_points]
                grid = np.reshape(grid, (llc, 3 * llc))
            field_faces[i] = grid
            points_counted += n_points

    if dim==3:

        for i in range(1, 6):
            if i < 3:
                face_grid = np.zeros((Nr, 3 * llc, llc))
            elif i == 3:
                face_grid = np.zeros((Nr, llc, llc))
            if i > 3:
                face_grid = np.zeros((Nr, llc, 3 * llc))
            field_faces[i]=face_grid

        for nr in range(Nr):
            points_counted = 0
            level_grid = grid_array[nr * N:(nr + 1) * N]
            for i in range(1,6):
                if i < 3:
                    n_points = 3*llc*llc
                    grid = level_grid[points_counted:points_counted+n_points]
                    grid = np.reshape(grid,(3*llc,llc))
                if i == 3:
                    n_points = llc * llc
                    grid = level_grid[points_counted:points_counted + n_points]
                    grid = np.reshape(grid, (llc, llc))
                if i > 3:
                    n_points = 3 * llc * llc
                    grid = level_grid[points_counted:points_counted + n_points]
                    grid = np.reshape(grid, (llc, 3*llc))
                field_faces[i][nr,:,:] = grid

                points_counted += n_points

    return(field_faces)

def read_ecco_faces_to_tiles(ecco_faces, llc, dim):
    ecco_tiles = {}
    if dim==2:
        ecco_tiles[1] = ecco_faces[1][:llc,:]
        ecco_tiles[2] = ecco_faces[1][llc:2*llc, :]
        ecco_tiles[3] = ecco_faces[1][2*llc:, :]
        ecco_tiles[4] = ecco_faces[2][:llc, :]
        ecco_tiles[5] = ecco_faces[2][llc:2*llc, :]
        ecco_tiles[6] = ecco_faces[2][2*llc:, :]
        ecco_tiles[7] = ecco_faces[3][:,:]
        ecco_tiles[8] = ecco_faces[4][:, :llc]
        ecco_tiles[9] = ecco_faces[4][:, llc:2*llc]
        ecco_tiles[10] = ecco_faces[4][:, 2*llc:]
        ecco_tiles[11] = ecco_faces[5][:, :llc]
        ecco_tiles[12] = ecco_faces[5][:, llc:2*llc]
        ecco_tiles[13] = ecco_faces[5][:, 2*llc:]
    if dim==3:
        ecco_tiles[1] = ecco_faces[1][:, :llc,:]
        ecco_tiles[2] = ecco_faces[1][:, llc:2*llc, :]
        ecco_tiles[3] = ecco_faces[1][:, 2*llc:, :]
        ecco_tiles[4] = ecco_faces[2][:, :llc, :]
        ecco_tiles[5] = ecco_faces[2][:, llc:2*llc, :]
        ecco_tiles[6] = ecco_faces[2][:, 2*llc:, :]
        ecco_tiles[7] = ecco_faces[3]
        ecco_tiles[8] = ecco_faces[4][:, :, :llc]
        ecco_tiles[9] = ecco_faces[4][:, :, llc:2*llc]
        ecco_tiles[10] = ecco_faces[4][:, :, 2*llc:]
        ecco_tiles[11] = ecco_faces[5][:, :, :llc]
        ecco_tiles[12] = ecco_faces[5][:, :, llc:2*llc]
        ecco_tiles[13] = ecco_faces[5][:, :, 2*llc:]
    return(ecco_tiles)

def read_ecco_grid_tiles_from_nc(grid_dir, var_name):
    ecco_tiles = {}

    for tile_number in range(1,14):
        ds = nc4.Dataset(os.path.join(grid_dir,'GRID.'+'{:04d}'.format(tile_number)+'.nc'))
        if var_name in ['hFacC','hFacS','hFacW']:
            grid = ds.variables[var_name][:, :, :]
        elif var_name in ['DRC','DRF','RC','RF']:
            grid = ds.variables[var_name][:]
        else:
            grid = ds.variables[var_name][:, :]
        ds.close()
        ecco_tiles[tile_number] = np.array(grid)
    return(ecco_tiles)

def read_ecco_geometry_to_faces(ecco_dir,llc, Nr):

    XC_faces = {}
    YC_faces = {}
    for i in [1, 2, 3, 4, 5]:
        if i < 3:
            grid = np.fromfile(os.path.join(ecco_dir, 'tile00' + str(i) + '.mitgrid'), '>f8')
            grid = np.reshape(grid, (16, 3 * llc+1, llc+1))
        if i == 3:
            grid = np.fromfile(os.path.join(ecco_dir, 'tile00' + str(i) + '.mitgrid'), '>f8')
            grid = np.reshape(grid, (16, llc+1, llc+1))
        if i > 3:
            grid = np.fromfile(os.path.join(ecco_dir, 'tile00' + str(i) + '.mitgrid'), '>f8')
            grid = np.reshape(grid, (16, llc+1, 3 * llc+1))
        XC_face = grid[0,:-1,:-1]
        YC_face = grid[1,:-1,:-1]
        XC_faces[i] = XC_face
        YC_faces[i] = YC_face

    angleCS_path = os.path.join(ecco_dir,'AngleCS.data')
    AngleCS_faces = read_ecco_field_to_faces(angleCS_path, llc, dim=2, Nr=Nr)

    angleSN_path = os.path.join(ecco_dir,'AngleSN.data')
    AngleSN_faces = read_ecco_field_to_faces(angleSN_path, llc, dim=2, Nr=Nr)

    hFacC_path = os.path.join(ecco_dir, 'hFacC.data')
    hFacC_faces = read_ecco_field_to_faces(hFacC_path, llc, dim=3, Nr=Nr)

    hFacW_path = os.path.join(ecco_dir, 'hFacW.data')
    hFacW_faces = read_ecco_field_to_faces(hFacW_path, llc, dim=3, Nr=Nr)

    hFacS_path = os.path.join(ecco_dir, 'hFacS.data')
    hFacS_faces = read_ecco_field_to_faces(hFacS_path, llc, dim=3, Nr=Nr)

    return(XC_faces, YC_faces, AngleCS_faces, AngleSN_faces, hFacC_faces, hFacW_faces, hFacS_faces)