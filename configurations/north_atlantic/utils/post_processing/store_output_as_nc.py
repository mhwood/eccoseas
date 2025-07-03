import os
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc4
from datetime import datetime, timedelta
import argparse
import ast
import sys
import shutil

def iter_number_to_year_month(iter_number,seconds_per_iter):
    total_seconds = iter_number*seconds_per_iter
    date = datetime(1992,1,15,12) + timedelta(seconds=total_seconds)
    year = date.year
    month=date.month
    year_month = str(year)+'{:02d}'.format(month)
    return(year_month)

def date_to_iter_number(date,seconds_per_iter):
    total_seconds = (date-datetime(1992,1,15,12)).total_seconds()
    iter_number = total_seconds/seconds_per_iter
    return(iter_number)

def iter_number_to_date(iter_number,seconds_per_iter):

    total_seconds = iter_number*seconds_per_iter
    date = datetime(1992,1,15,12) + timedelta(seconds=total_seconds)
    return(date)

def read_grid_geometry_from_nc(config_dir, model_name):
    file_path = os.path.join(config_dir, 'input', model_name+'_grid.nc')
    ds = nc4.Dataset(file_path)
    XC = ds.variables['XC'][:,:]
    YC = ds.variables['YC'][:,:]
    drF = ds.variables['drF'][:]
    ds.close()
    Z_bottom = np.cumsum(drF)
    Z_top = np.concatenate([np.array([0]), Z_bottom[:-1]])
    Z = (Z_bottom + Z_top) / 2
    return(XC, YC, Z)

def get_variable_list(subset):

    if subset == 'EtaN_day_mean':
        var_names = ['EtaN']
    elif subset == 'EtaN_day_snap':
        var_names = ['EtaN']
    elif subset == 'EtaN_mon_mean':
        var_names = ['EtaN']
    elif subset=='SI_daily_mean':
        var_names = ['SIarea','SIheff','SIhsnow','SIuice','SIvice']
    elif subset=='SI_daily_snap':
        var_names = ['SIarea','SIheff','SIhsnow','SIuice','SIvice']
    elif subset == 'TS_surf_daily_snap':
        var_names = ['Theta_surf','Salt_surf']
    elif subset == 'vel_surf_daily_snap':
        var_names = ['Uvel_surf','Vvel_surf']
    elif subset == 'TS_AW_daily_snap':
        var_names = ['Theta_AW','Salt_AW']
    elif subset == 'state_3D_day_mean':
        var_names = ['Theta', 'Salt']
    elif subset == 'vel_3D_day_mean':
        var_names = ['Uvel','Vvel','Wvel']
    elif subset == 'vol_flux':
        var_names = ['UVELMASS','VVELMASS','WVELMASS']
    elif subset == 'heat_flux_adv':
        var_names = ['ADVx_TH','ADVy_TH','ADVr_TH']
    elif subset == 'heat_flux_diff':
        var_names = ['DFrI_TH','KPPg_TH']
    elif subset == 'heat_flux_surf':
        var_names = ['oceQsw','WTHMASS','TFLUX']
    elif subset == 'salt_flux_adv':
        var_names = ['ADVx_SLT','ADVy_SLT','ADVr_SLT']
    elif subset== 'salt_flux_diff':
        var_names = ['DFrI_SLT','KPPg_SLT']
    elif subset== 'salt_flux_surf':
        var_names = ['oceFWflx','WSLTMASS','SFLUX']
    else:
        raise ValueError('Variable not yet implemented')

    return(var_names)

def get_variable_classifications(subset):
    if subset == 'EtaN_day_mean':
        classification = 'daily_mean'
    elif subset == 'EtaN_day_snap':
        classification = 'daily_snapshot'
    elif subset == 'EtaN_mon_mean':
        classification = 'monthly_mean'
    elif subset == 'SI_daily_mean':
        classification = 'daily_mean'
    elif subset == 'TS_surf_daily_snap':
        classification = 'daily_snapshot'
    elif subset == 'SI_daily_snap':
        classification = 'daily_snapshot'
    elif subset == 'vel_surf_daily_snap':
        classification = 'daily_snapshot'
    elif subset == 'TS_AW_daily_snap':
        classification = 'daily_snapshot'
    elif subset == 'state_3D_day_mean':
        classification = 'daily_mean'
    elif subset == 'vel_3D_day_mean':
        classification = 'daily_mean'
    elif subset == 'vol_flux':
        classification = 'monthly_mean'
    elif subset == 'heat_flux_adv':
        classification = 'monthly_mean'
    elif subset == 'heat_flux_diff':
        classification = 'monthly_mean'
    elif subset == 'heat_flux_surf':
        classification = 'monthly_mean'
    elif subset == 'salt_flux_adv':
        classification = 'monthly_mean'
    elif subset == 'salt_flux_adv':
        classification = 'monthly_mean'
    elif subset == 'salt_flux_diff':
        classification = 'monthly_mean'
    elif subset == 'salt_flux_surf':
        classification = 'monthly_mean'
    else:
        raise ValueError('Variable not yet implemented')

    return(classification)

def get_list_of_iter_numbers(year, month, subset, seconds_per_iter, classification):
    iter_numbers = []

    if month in [1, 3, 5, 7, 8, 10, 12]:
        n_days = 31
    elif month in [4, 6, 9, 11]:
        n_days = 30
    else:
        if year % 4 == 0:
            n_days = 29
        else:
            n_days = 28

    if year == 1992 and month == 1:
        start_day = 15
    else:
        start_day = 1

    if 'daily' in classification or 'day' in classification:
        for day in range(start_day, n_days + 1):
            if subset in ['TS_AW_daily_snap','vel_surf_daily_snap','TS_surf_daily_snap',
                          'SI_daily_snap','EtaN_day_snap']:
                iter_number = date_to_iter_number(datetime(year, month, day) + timedelta(days=1),
                                                  seconds_per_iter)
            else:
                iter_number = date_to_iter_number(datetime(year, month, day) + timedelta(days=0.5), seconds_per_iter)
            iter_numbers.append(int(iter_number))
    elif 'monthly_mean' in classification:
        if month<12:
            iter_number = date_to_iter_number(datetime(year, month+1, 1), seconds_per_iter)
        else:
            iter_number = date_to_iter_number(datetime(year+1, 1, 1), seconds_per_iter)
        iter_numbers.append(int(iter_number))
    else:
        raise ValueError('No iters defined for this classification')

    return(iter_numbers)

def check_if_files_are_present(run_dir, subset, iter_numbers):

    all_found = True
    count = 0
    for iter_number in iter_numbers:
        file_name = subset + '.' + '{:010d}'.format(iter_number) + '.data'
        if file_name not in os.listdir(os.path.join(run_dir, 'diags', subset)):
            all_found = False
        else:
            count+=1
    print('            - Found '+str(count)+' of '+str(len(iter_numbers)))

    return(all_found)

def stack_files_to_nc(run_dir, subset, var_names, output_dir, year, month, classification, var_name, iter_numbers, XC, YC, Z):

    Nr = len(Z)
    var_index = var_names.index(var_name)

    if subset in ['SI_daily_mean','EtaN_day_snap','TS_AW_daily_snap','EtaN_day_mean','SI_daily_snap',
                  'TS_surf_daily_snap', 'EtaN_mon_mean','heat_flux_surf','salt_flux_surf','vel_surf_daily_snap']:
        has_Nr = False
    else:
        has_Nr = True

    if has_Nr:
        output_array = np.zeros((len(iter_numbers),len(Z),np.shape(XC)[0],np.shape(XC)[1]))
    else:
        output_array = np.zeros((len(iter_numbers), np.shape(XC)[0], np.shape(XC)[1]))


    counter = 0
    for iter_number in iter_numbers:
        file_path = os.path.join(run_dir,'diags',subset, subset + '.' + '{:010d}'.format(iter_number) + '.data')
        print('                - Reading from '+subset + '.' + '{:010d}'.format(iter_number) + '.data')
        grid = np.fromfile(file_path, '>f4')

        if has_Nr:
            grid = np.reshape(grid, (len(var_names)*Nr, np.shape(XC)[0], np.shape(XC)[1]))
            grid = grid[var_index*Nr:(var_index+1)*Nr,:,:]
            output_array[counter, :, :, :] = grid
        else:
            grid = np.reshape(grid, (len(var_names), np.shape(XC)[0], np.shape(XC)[1]))
            grid = grid[var_index,:,:]
            output_array[counter, :, :] = grid

        counter += 1

    ds = nc4.Dataset(os.path.join(output_dir, classification, var_name,var_name+'_'+str(year)+'{:02d}'.format(month)+'.nc'), 'w')
    ds.createDimension('iterations', len(iter_numbers))
    ds.createDimension('rows', np.shape(XC)[0])
    ds.createDimension('cols', np.shape(XC)[1])
    if has_Nr:
        ds.createDimension('depths', len(Z))

    ivar = ds.createVariable('iterations', 'f4', ('iterations',))
    ivar[:] = iter_numbers

    if has_Nr:
        evar = ds.createVariable(var_name, 'f4', ('iterations', 'depths', 'rows', 'cols'))
        evar[:, :, :, :] = output_array
    else:
        evar = ds.createVariable(var_name, 'f4', ('iterations', 'rows', 'cols'))
        evar[:, :, :] = output_array


    xvar = ds.createVariable('longitude', 'f4', ('rows', 'cols'))
    xvar[:, :] = XC

    yvar = ds.createVariable('latitude', 'f4', ('rows', 'cols'))
    yvar[:, :] = YC

    if has_Nr:
        zvar = ds.createVariable('depths', 'f4', ('depths',))
        zvar[:] = Z

    ds.close()

def stack_mds_output_to_nc(config_dir, run_dir_path, subsets, output_dir,XC, YC, Z, seconds_per_iter):
    delete_file_list = []
    check_if_exists = True

    for subset in subsets:

        print('       - Creating datasets for the ' + subset + ' subset')

        classification = get_variable_classifications(subset)

        print('          - Classification: ' + str(classification))
        if classification not in os.listdir(output_dir):
            os.mkdir(os.path.join(output_dir, classification))

        var_names = get_variable_list(subset)
        for var_name in var_names:
            if var_name not in os.listdir(os.path.join(output_dir, classification)):
                os.mkdir(os.path.join(output_dir, classification, var_name))

        for year in range(1992, 1993):
            for month in range(1, 13):

                if subset in os.listdir(os.path.join(run_dir_path, 'diags')):

                    print('        - Working on files for ' + str(year) + '/' + str(month))
                    iter_numbers = get_list_of_iter_numbers(year, month, subset, seconds_per_iter, classification)

                    all_files_present = check_if_files_are_present(run_dir_path, subset, iter_numbers)

                    if all_files_present:
                        print('            - All files found for ' + subset + ' in month ' + str(month))

                        for var_name in var_names:

                            if check_if_exists:
                                make_file = False
                                if var_names[-1] + '_' + str(year) + '{:02d}'.format(month) + '.nc' not in os.listdir(
                                        os.path.join(output_dir, classification, var_names[-1])):
                                    make_file = True
                            else:
                                make_file = True

                            if make_file:
                                stack_files_to_nc(run_dir_path, subset, var_names, output_dir, year, month,
                                                  classification, var_name,
                                                  iter_numbers, XC, YC, Z)
                                if var_name == var_names[-1]:
                                    for i in range(len(iter_numbers)):
                                        delete_file_list.append([subset, subset + '.' + '{:010d}'.format(iter_numbers[i])])
                            else:
                                print('        - Skipping ' + var_name + ' in ' + str(year) + '/' + str(
                                    month) + ' - already made')



########################################################################################################################

def stack_data_to_nc(config_dir, subset):

    experiments = ['noIB']

    for experiment in experiments:
        print('    - Creating the files for the '+experiment+' experiment')
        run_dir = 'run_' + experiment
        results_dir = 'results_' + experiment
        seconds_per_iter = 60

        run_dir_path = os.path.join(config_dir,run_dir)
        output_dir = os.path.join(config_dir,results_dir)

        if subset=='All':
            subsets = ['EtaN_day_snap','TS_surf_daily_snap', 'SI_daily_snap',
                       'vel_surf_daily_snap']
        else:
            subsets = [subset]

        XC, YC, Z = read_grid_geometry_from_nc(config_dir, 'north_atlantic')

        if results_dir not in os.listdir(os.path.join(config_dir)):
            os.mkdir(os.path.join(config_dir, results_dir))

        stack_mds_output_to_nc(config_dir, run_dir_path, subsets, output_dir,XC, YC, Z, seconds_per_iter)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--config_dir", action="store",
                        help="The directory where the L2, L2, and L3 configurations are stored.", dest="config_dir",
                        type=str, required=True)

    parser.add_argument("-s", "--subset", action="store",
                        help="The subset to stack (e.g. surfDiag, awDiag, seaiceDiag, dynDiag).", dest="subset",
                        type=str, required=False, default='All')

    # parser.add_argument("-r", "--run_dir", action="store",
    #                     help="Choose the run directory.", dest="run_dir",
    #                     type=str, required=True)

    args = parser.parse_args()
    config_dir = args.config_dir
    subset = args.subset

    stack_data_to_nc(config_dir, subset)



