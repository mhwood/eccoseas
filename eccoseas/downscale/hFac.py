import numpy as np

def create_hFacC_grid(bathy, delR, hFacMin=0.2, hFacMinDr=5.0):
    # This is from MITgcm inside ini_masks_etc.F
    # Note: R_low is just the bathy
    # Note: drF is the grid spacing (provided)
    # Note: recip_drF is the reciprocal of drF
    # Note: Ro_surf is the z coord of the surface (essentially always 0 for the ocean?)
    # C--   Calculate lopping factor hFacC : over-estimate the part inside of the domain
    # C     taking into account the lower_R Boundary (Bathymetry / Top of Atmos)
    #         DO k=1, Nr
    #          hFacMnSz = MAX( hFacMin, MIN(hFacMinDr*recip_drF(k),oneRL) )
    #          DO j=1-OLy,sNy+OLy
    #           DO i=1-OLx,sNx+OLx
    # C      o Non-dimensional distance between grid bound. and domain lower_R bound.
    #            hFac_loc = (rF(k)-R_low(i,j,bi,bj))*recip_drF(k)
    # C      o Select between, closed, open or partial (0,1,0-1)
    #            hFac_loc = MIN( MAX( hFac_loc, zeroRL ) , oneRL )
    # C      o Impose minimum fraction and/or size (dimensional)
    #            IF ( hFac_loc.LT.hFacMnSz*halfRL .OR.
    #      &          R_low(i,j,bi,bj).GE.Ro_surf(i,j,bi,bj) ) THEN
    #              hFacC(i,j,k,bi,bj) = zeroRS
    #            ELSE
    #              hFacC(i,j,k,bi,bj) = MAX( hFac_loc, hFacMnSz )
    #            ENDIF
    #           ENDDO
    #          ENDDO
    #         ENDDO

    """
    Create the vertical cell fraction (hFacC) grid for MITgcm based on bathymetry.

    Parameters:
        bathy (ndarray): 2D array of bathymetric depths (positive down).
        delR (ndarray): 1D array of vertical layer thicknesses.
        hFacMin (float): Minimum cell fraction.
        hFacMinDr (float): Minimum physical cell thickness.

    Returns:
        ndarray: 3D array representing hFacC values.
    """
    RU = -1 * np.cumsum(delR)
    RL = np.concatenate([[0], RU[:-1]])
    R_low = bathy
    drF = RL - RU
    recip_drF = 1 / drF

    hFacC = np.zeros((len(RL), *bathy.shape))
    for k in range(len(RL)):
        hFacMnSz = np.max([hFacMin, np.min([hFacMinDr * recip_drF[k], 1])])
        for i in range(bathy.shape[0]):
            for j in range(bathy.shape[1]):
                hFac_loc = (RL[k] - R_low[i, j]) * recip_drF[k]
                hFac_loc = np.min([np.max([hFac_loc, 0]), 1])
                if hFac_loc <= hFacMnSz * 0.5 or R_low[i, j] >= 0:
                    hFacC[k, i, j] = 0
                else:
                    hFacC[k, i, j] = np.max([hFac_loc, hFacMnSz])

    return hFacC

def create_hFacS_grid(bathy, delR, hFacMin=0.2, hFacMinDr=5.0):
    # This is from MITgcm inside ini_masks_etc.F
    # Note: R_low is just the bathy
    # Note: drF is the grid spacing (provided)
    # Note: recip_drF is the reciprocal of drF
    # Note: Ro_surf is the z coord of the surface (essentially always 0 for the ocean?)

    """
    Create the hFacS grid (cell face values on south face) for MITgcm.

    Parameters:
        bathy (ndarray): 2D array of bathymetric depths.
        delR (ndarray): 1D array of vertical layer thicknesses.
        hFacMin (float): Minimum cell fraction.
        hFacMinDr (float): Minimum physical cell thickness.

    Returns:
        ndarray: 3D array of hFacS values.
    """
    RU = -1 * np.cumsum(delR)
    RL = np.concatenate([[0], RU[:-1]])
    R_low = bathy
    drF = RL - RU
    recip_drF = 1 / drF

    rLowS = np.copy(bathy)
    for j in range(1, rLowS.shape[0]):
        for i in range(rLowS.shape[1]):
            rLowS[j, i] = np.max([R_low[j - 1, i], R_low[j, i]])

    hFacS = np.zeros((len(delR), *bathy.shape))
    for k in range(len(delR)):
        hFacMnSz = np.max([hFacMin, np.min([hFacMinDr * recip_drF[k], 1])])
        for j in range(rLowS.shape[0]):
            for i in range(rLowS.shape[1]):
                hFac1tmp = (RL[k] - rLowS[j, i]) * recip_drF[k]
                hFac_loc = np.min([hFac1tmp, 1])
                if hFac_loc < hFacMnSz * 0.5 or rLowS[j, i] >= 0:
                    hFac1tmp = 0
                else:
                    hFac1tmp = np.max([hFac_loc, hFacMnSz])
                hFac2tmp = RL[k] * recip_drF[k]
                hFac_loc = hFac1tmp - np.max([hFac2tmp, 0])
                if hFac_loc < hFacMnSz * 0.5:
                    hFacS[k, j, i] = 0
                else:
                    hFacS[k, j, i] = np.max([hFac_loc, hFacMnSz])

    return hFacS

def create_hFacW_grid(bathy, delR, hFacMin=0.2, hFacMinDr=5.0):
    # This is from MITgcm inside ini_masks_etc.F

    """
    Create the hFacW grid (cell face values on west face) for MITgcm.

    Parameters:
        bathy (ndarray): 2D array of bathymetric depths.
        delR (ndarray): 1D array of vertical layer thicknesses.
        hFacMin (float): Minimum cell fraction.
        hFacMinDr (float): Minimum physical cell thickness.

    Returns:
        ndarray: 3D array of hFacW values.
    """
    RU = -1 * np.cumsum(delR)
    RL = np.concatenate([[0], RU[:-1]])
    R_low = bathy
    drF = RL - RU
    recip_drF = 1 / drF

    rLowW = np.copy(bathy)
    for j in range(rLowW.shape[0]):
        for i in range(1, rLowW.shape[1]):
            rLowW[j, i] = np.max([R_low[j, i - 1], R_low[j, i]])

    hFacW = np.zeros((len(delR), *bathy.shape))
    for k in range(len(delR)):
        hFacMnSz = np.max([hFacMin, np.min([hFacMinDr * recip_drF[k], 1])])
        for j in range(rLowW.shape[0]):
            for i in range(rLowW.shape[1]):
                hFac1tmp = (RL[k] - rLowW[j, i]) * recip_drF[k]
                hFac_loc = np.min([hFac1tmp, 1])
                if hFac_loc < hFacMnSz * 0.5 or rLowW[j, i] >= 0:
                    hFac1tmp = 0
                else:
                    hFac1tmp = np.max([hFac_loc, hFacMnSz])
                hFac2tmp = RL[k] * recip_drF[k]
                hFac_loc = hFac1tmp - np.max([hFac2tmp, 0])
                if hFac_loc < hFacMnSz * 0.5:
                    hFacW[k, j, i] = 0
                else:
                    hFacW[k, j, i] = np.max([hFac_loc, hFacMnSz])

    return hFacW
