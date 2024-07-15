
import os
import numpy as np

#################################################################################################
# exf functions


def read_ecco_exf_file(data_dir, file_prefix, year):
    """
        Read an external forcing field used in ECCO Version 5 Alpha.
        (e.g. EIG_dsw_plus_ECCO_v4r1_ctrl_2008)

        Parameters
        ----------
        data_dir : str
            A file path to the directory where the exf files are stored.
        file_prefix : float
            The prefix of the file omitting the year (e.g. EIG_dsw_plus_ECCO_v4r1_ctrl)
        year : int
            The year of the file (e.g. 2008)

        Returns
        -------
        lon : 1-d numpy array
            An array of longitude values corresponding to the geometry of the grid.
        lat : 1-d numpy array
            An array of latitude values corresponding to the geometry of the grid.
        lon : 3-d numpy array
            An array of external forcing values corresponding to file requested.
        """

    lon = np.arange(0,360,0.7031250)
    lon[lon>180] -= 360
    first_index = np.where(lon<0)[0][0]
    lon = np.concatenate([lon[first_index:],lon[:first_index]])

    del_lat = np.concatenate([np.array([0]),
                              np.array([0.6958694]),
                              np.array([0.6999817]),
                              np.array([0.7009048]),
                              np.array([0.7012634]),
                              np.array([0.7014313]),
                              0.7017418*np.ones((245,)),
                              np.array([0.7014313]),
                              np.array([0.7012634]),
                              np.array([0.7009048]),
                              np.array([0.6999817]),
                              np.array([0.6958694]),
                              ])
    lat = np.cumsum(del_lat)+-89.4628220

    grid = np.fromfile(os.path.join(data_dir, file_prefix + '_' + str(year)), '>f4')
    n_timesteps = int(np.size(grid)/(len(lon)*len(lat)))
    grid = np.reshape(grid,(n_timesteps,len(lat),len(lon)))

    grid = np.concatenate([grid[:,:,first_index:],grid[:,:,:first_index]],axis=2)

    return(lon,lat,grid)
