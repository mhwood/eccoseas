
import os
import numpy as np
import simplegrid as sg
import matplotlib.pyplot as plt
import netCDF4 as nc4
from MITgcmutils import mds

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





