
import os
import numpy as np
import simplegrid as sg
import matplotlib.pyplot as plt
import netCDF4 as nc4
from MITgcmutils import mds


#################################################################################################
# pickup functions

def read_pickup_file_to_compact(pickup_file_path):

    Nr = 50
    print('      Reading from '+pickup_file_path)
    global_data, _, global_metadata = mds.rdmds(pickup_file_path, returnmeta=True)

    has_Nr = {'uvel': True, 'vvel': True, 'theta': True,
              'salt': True, 'gunm1': True, 'gvnm1': True,
              'gunm2': True, 'gvnm2': True, 'etan': False,
              'detahdt': False, 'etah': False}

    var_names = []
    row_bounds = []
    var_grids = []

    start_row = 0
    for var_name in global_metadata['fldlist']:
        if has_Nr[var_name.strip().lower()]:
            end_row = start_row + Nr
        else:
            end_row = start_row + 1
        var_grid = global_data[start_row:end_row,:,:]
        var_grids.append(var_grid)
        row_bounds.append([start_row,end_row])
        start_row=end_row
        var_names.append(var_name.strip())

    return(var_names,row_bounds,var_grids,global_metadata)

#################################################################################################
# seaice pickup functions

def read_seaice_pickup_file_to_compact(pickup_file_path):

    print('      Reading from '+pickup_file_path)
    global_data, _, global_metadata = mds.rdmds(pickup_file_path, returnmeta=True)

    var_names = []
    row_bounds = []
    var_grids = []

    start_row = 0
    for var_name in global_metadata['fldlist']:
        end_row = start_row + 1
        var_grid = global_data[start_row:end_row,:,:]
        var_grids.append(var_grid)
        row_bounds.append([start_row,end_row])
        start_row=end_row
        var_names.append(var_name.strip())

    return(var_names,row_bounds,var_grids,global_metadata)

def read_ecco_seaice_pickup_to_stiched_grid(pickup_file_path,ordered_ecco_tiles,ordered_ecco_tile_rotations):

    ecco_sNx = 90
    ecco_sNy = 90

    # pickup_file_path = os.path.join(ecco_dir,'LLC'+str(llc)+'_Files','input_init',pickup_file)
    var_names,row_bounds,compact_var_grids,global_metadata = read_seaice_pickup_file_to_compact(pickup_file_path)

    var_grids = []

    for vn in range(len(var_names)):
        compact_var_grid = compact_var_grids[vn]
        ecco_grid = np.zeros((1, ecco_sNy * len(ordered_ecco_tiles), ecco_sNx * len(ordered_ecco_tiles[0])))

        for r in range(len(ordered_ecco_tiles)):
            ecco_tile_row = ordered_ecco_tiles[r]
            ecco_rotation_row = ordered_ecco_tile_rotations[r]
            for c in range(len(ordered_ecco_tiles[r])):

                # get the variable grid
                var_grid = read_tile_from_ECCO_compact(compact_var_grid, tile_number=ecco_tile_row[c])

                # rotate things as necessary
                for n in range(ecco_rotation_row[c]):
                    var_grid = np.rot90(var_grid, axes=(1, 2))

                # put it into the big grid
                ecco_grid[:, r * ecco_sNy:(r + 1) * ecco_sNy, c * ecco_sNx:(c + 1) * ecco_sNx] = var_grid

        var_grids.append(ecco_grid)

    return(var_names, var_grids, global_metadata)

def rotate_ecco_seaice_grids_to_natural_grids(var_names, var_grids, ecco_AngleCS, ecco_AngleSN):

    def rotate_velocity_vectors_to_natural(angle_cos, angle_sin, uvel, vvel):
        zonal_velocity = np.zeros_like(uvel)
        meridional_velocity = np.zeros_like(vvel)
        for k in range(np.shape(uvel)[0]):
            zonal_velocity[k,:,:] = angle_cos * uvel[k,:,:] - angle_sin * vvel[k,:,:]
            meridional_velocity[k,:,:] = angle_sin * uvel[k,:,:] + angle_cos * vvel[k,:,:]
        return (zonal_velocity, meridional_velocity)

    uvel_grid = var_grids[var_names.index('siUICE')]
    vvel_grid = var_grids[var_names.index('siVICE')]
    natural_uvel_grid, natural_vvel_grid = rotate_velocity_vectors_to_natural(ecco_AngleCS, ecco_AngleSN, uvel_grid, vvel_grid)
    var_grids[var_names.index('siUICE')] = natural_uvel_grid
    var_grids[var_names.index('siVICE')] = natural_vvel_grid

    return(var_grids)

#################################################################################################
# ptracer pickup functions

def read_ptracer_pickup_file_to_compact(pickup_file_path):

    Nr = 50
    print('      Reading from '+pickup_file_path)
    global_data, _, global_metadata = mds.rdmds(pickup_file_path, returnmeta=True)

    var_names = []
    var_grids = []

    for vn in range(len(global_metadata['fldlist'])):
        var_grid = global_data[vn,:,:,:]
        var_grids.append(var_grid)
        var_names.append(global_metadata['fldlist'][vn].strip())

    return(var_names,var_grids,global_metadata)

def read_ecco_ptracer_pickup_to_stiched_grid(pickup_file_path,ordered_ecco_tiles,ordered_ecco_tile_rotations):

    ecco_Nr = 50
    ecco_sNx = 90
    ecco_sNy = 90

    # pickup_file_path = os.path.join(ecco_dir,'LLC'+str(llc)+'_Files','input_init',pickup_file)
    var_names,compact_var_grids,global_metadata = read_ptracer_pickup_file_to_compact(pickup_file_path)

    var_grids = []

    for vn in range(len(var_names)):
        compact_var_grid = compact_var_grids[vn]
        ecco_grid = np.zeros((ecco_Nr, ecco_sNy * len(ordered_ecco_tiles), ecco_sNx * len(ordered_ecco_tiles[0])))

        for r in range(len(ordered_ecco_tiles)):
            ecco_tile_row = ordered_ecco_tiles[r]
            ecco_rotation_row = ordered_ecco_tile_rotations[r]
            for c in range(len(ordered_ecco_tiles[r])):

                # get the variable grid
                var_grid = read_tile_from_ECCO_compact(compact_var_grid, tile_number=ecco_tile_row[c])

                # rotate things as necessary
                for n in range(ecco_rotation_row[c]):
                    var_grid = np.rot90(var_grid, axes=(1, 2))

                # put it into the big grid
                ecco_grid[:, r * ecco_sNy:(r + 1) * ecco_sNy, c * ecco_sNx:(c + 1) * ecco_sNx] = var_grid

        var_grids.append(ecco_grid)

    return(var_names, var_grids, global_metadata)

#################################################################################################
# darwin pickup function

def read_darwin_pickup_file_to_compact(pickup_file_path):

    Nr = 50
    print('      Reading from '+pickup_file_path)
    global_data, _, global_metadata = mds.rdmds(pickup_file_path, returnmeta=True)

    # there is only one field in the darwin pickup

    var_names = [global_metadata['fldlist'][0]]
    var_grids = [global_data]

    return(var_names,var_grids,global_metadata)

def read_ecco_darwin_pickup_to_stiched_grid(pickup_file_path,ordered_ecco_tiles,ordered_ecco_tile_rotations):

    ecco_Nr = 50
    ecco_sNx = 90
    ecco_sNy = 90

    # pickup_file_path = os.path.join(ecco_dir,'LLC'+str(llc)+'_Files','input_init',pickup_file)
    var_names,compact_var_grids,global_metadata = read_darwin_pickup_file_to_compact(pickup_file_path)

    var_grids = []

    for vn in range(len(var_names)):
        compact_var_grid = compact_var_grids[vn]
        ecco_grid = np.zeros((ecco_Nr, ecco_sNy * len(ordered_ecco_tiles), ecco_sNx * len(ordered_ecco_tiles[0])))

        for r in range(len(ordered_ecco_tiles)):
            ecco_tile_row = ordered_ecco_tiles[r]
            ecco_rotation_row = ordered_ecco_tile_rotations[r]
            for c in range(len(ordered_ecco_tiles[r])):

                # get the variable grid
                var_grid = read_tile_from_ECCO_compact(compact_var_grid, tile_number=ecco_tile_row[c])

                # rotate things as necessary
                for n in range(ecco_rotation_row[c]):
                    var_grid = np.rot90(var_grid, axes=(1, 2))

                # put it into the big grid
                ecco_grid[:, r * ecco_sNy:(r + 1) * ecco_sNy, c * ecco_sNx:(c + 1) * ecco_sNx] = var_grid

        var_grids.append(ecco_grid)

    return(var_names, var_grids, global_metadata)

#################################################################################################
# VELMASS to VEL conversion

def convert_velmass_to_vel(ecco_dir, llc, var_name, year):
    # to convert to UVEL and VVEL, we will use the relationship explained here:
    # https://ecco-v4-python-tutorial.readthedocs.io/ECCO_v4_Volume_budget_closure.html

    #######################################################################################
    # read in the fields

    # get the velmass file
    ds = nc4.Dataset(os.path.join(ecco_dir,'LLC'+str(llc)+'_Files','nctiles_monthly', var_name + 'MASS', var_name + 'MASS_' + str(year) + '.nc'))
    if var_name == 'UVEL':
        i = ds.variables['i_g'][:]
        j = ds.variables['j'][:]
    if var_name == 'VVEL':
        i = ds.variables['i'][:]
        j = ds.variables['j_g'][:]
    k = ds.variables['k'][:]
    tile_var = ds.variables['tile'][:]
    time = ds.variables['time'][:]
    timestep_var = ds.variables['timestep'][:]
    velmass = ds.variables[var_name + 'MASS'][:, :, :, :, :]
    ds.close()

    # get the ETAN field
    ds = nc4.Dataset(os.path.join(ecco_dir,'LLC'+str(llc)+'_Files','nctiles_monthly','ETAN','ETAN_' + str(year) + '.nc'))
    etan = ds.variables['ETAN'][:, :, :, :]
    ds.close()

    if var_name=='UVEL':
        file_path = os.path.join(ecco_dir,'LLC'+str(llc)+'_Files','input_init','hFacW.data')
    if var_name=='VVEL':
        file_path = os.path.join(ecco_dir,'LLC'+str(llc)+'_Files','input_init','hFacS.data')
    hFac_faces = read_ecco_field_to_faces(file_path, llc, dim=3)
    hFac_tiles = read_ecco_faces_to_tiles(hFac_faces, llc, dim=3)

    depth_file_path = os.path.join(ecco_dir, 'LLC' + str(llc) + '_Files', 'input_init', 'bathy270_filled_noCaspian_r4')
    depth_faces = read_ecco_field_to_faces(depth_file_path, llc, dim=2)
    depth_tiles = read_ecco_faces_to_tiles(depth_faces, llc, dim=2)

    #######################################################################################
    # do the conversion

    vel = np.zeros_like(velmass)
    for timestep in range(np.shape(velmass)[0]):
        for tile in range(np.shape(velmass)[2]):
            s_star = 1 + etan[timestep, tile, :, :] / depth_tiles[tile+1][:, :]
            vel[timestep, :, tile, :, :] = velmass[timestep, :, tile, :, :] / (hFac_tiles[tile+1] * s_star)

    # double check its masked
    for timestep in range(np.shape(velmass)[0]):
        for tile in range(np.shape(velmass)[2]):
            subset = vel[timestep, :, tile, :, :]
            subset[hFac_tiles[tile+1][:, :, :] == 0] = 0
            vel[timestep, :, tile, :, :] = subset

    # #######################################################################################
    # # maybe plot a sanity check
    # plt.subplot(1,2,1)
    # C = plt.imshow(velmass[3,0,7,:,:],origin='lower')
    # plt.colorbar(C)
    # plt.title('velmass')
    # plt.subplot(1, 2, 2)
    # C = plt.imshow(vel[3, 0, 7, :, :], origin='lower')
    # plt.colorbar(C)
    # plt.title('vel')
    # plt.show()

    #######################################################################################
    # write out the field
    if var_name not in os.listdir(os.path.join(ecco_dir,'LLC'+str(llc)+'_Files','nctiles_monthly')):
        os.mkdir(os.path.join(ecco_dir,'LLC'+str(llc)+'_Files','nctiles_monthly',var_name))

    ds = nc4.Dataset(os.path.join(ecco_dir,'LLC'+str(llc)+'_Files','nctiles_monthly', var_name, var_name + '_' + str(year) + '.nc'), 'w')

    if var_name=='UVEL':
        ds.createDimension('i_g', np.size(i))
        ds.createDimension('j', np.size(j))
    if var_name=='VVEL':
        ds.createDimension('i', np.size(i))
        ds.createDimension('j_g', np.size(j))
    ds.createDimension('k', np.size(k))
    ds.createDimension('tile', np.size(tile_var))
    ds.createDimension('time', np.size(time))

    if var_name=='UVEL':
        var = ds.createVariable('i_g', 'f4', ('i_g',))
        var[:] = i
        var = ds.createVariable('j', 'f4', ('j',))
        var[:] = j
    if var_name=='VVEL':
        var = ds.createVariable('i', 'f4', ('i',))
        var[:] = i
        var = ds.createVariable('j_g', 'f4', ('j_g',))
        var[:] = j

    var = ds.createVariable('k', 'f4', ('k',))
    var[:] = k

    var = ds.createVariable('tile', 'f4', ('tile',))
    var[:] = tile_var

    var = ds.createVariable('time', 'f4', ('time',))
    var[:] = time

    var = ds.createVariable('timestep', 'f4', ('time',))
    var[:] = timestep_var

    if var_name=='UVEL':
        var = ds.createVariable('UVEL', 'f4', ('time', 'k', 'tile', 'i_g', 'j'))
        var[:, :, :, :, :] = vel
    if var_name=='VVEL':
        var = ds.createVariable('VVEL', 'f4', ('time', 'k', 'tile', 'i', 'j_g'))
        var[:, :, :, :, :] = vel

    ds.close()





