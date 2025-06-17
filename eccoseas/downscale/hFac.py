
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata, interp1d
import netCDF4 as nc4



def create_hFacC_grid(bathy,delR, hFacMin=0.2, hFacMinDr=5.0):
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

    # Define grids with same names as those in MITgcm
    RU = -1 * np.cumsum(delR)
    RL = np.concatenate([[0], RU[:-1]])
    R_low = bathy
    drF = RL - RU
    recip_drF = 1 / drF

    # Pythonize the above loops
    hFacC = np.zeros((len(RL), np.shape(bathy)[0], np.shape(bathy)[1]))
    for k in range(len(RL)):
        # if k%5==0:
        #     print('     - Calculating hFacC for depth cells '+str(k)+' to '+str(k+5))
        hFacMnSz = np.max([hFacMin, np.min([hFacMinDr * recip_drF[k], 1])])
        for i in range(np.shape(bathy)[0]):
            for j in range(np.shape(bathy)[1]):
                #      o Non-dimensional distance between grid bound. and domain lower_R bound.
                hFac_loc = (RL[k] - R_low[i, j]) * recip_drF[k]
                #      o Select between, closed, open or partial (0,1,0-1)
                hFac_loc = np.min([np.max([hFac_loc, 0]), 1])
                #      o Impose minimum fraction and/or size (dimensional)
                if hFac_loc <= hFacMnSz * 0.5 or R_low[i, j] >= 0:
                    hFacC[k, i, j] = 0
                else:
                    hFacC[k, i, j] = np.max([hFac_loc, hFacMnSz])

    return(hFacC)

def create_hFacS_grid(bathy,delR, hFacMin=0.2, hFacMinDr=5.0):
    # This is from MITgcm inside ini_masks_etc.F
    # Note: R_low is just the bathy
    # Note: drF is the grid spacing (provided)
    # Note: recip_drF is the reciprocal of drF
    # Note: Ro_surf is the z coord of the surface (essentially always 0 for the ocean?)

    RU = -1 * np.cumsum(delR)
    RL = np.concatenate([[0], RU[:-1]])
    R_low = bathy
    drF = RL - RU
    recip_drF = 1 / drF

    # rLowS = np.zeros((np.shape(bathy)[0], np.shape(bathy)[1]))
    rLowS = np.copy(bathy)
    for j in range(1,np.shape(rLowS)[0]):
        for i in range(np.shape(rLowS)[1]):
            rLowS[j,i] = np.max([R_low[j-1,i],R_low[j,i]])

    hFacS = np.zeros((len(delR), np.shape(bathy)[0], np.shape(bathy)[1]))
    for k in range(len(delR)):
        # if k%5==0:
        #     print('     - Calculating hFacS for depth cells '+str(k)+' to '+str(k+5))
        hFacMnSz = np.max([hFacMin, np.min([hFacMinDr * recip_drF[k], 1])])
        for j in range(np.shape(rLowS)[0]):
            for i in range(np.shape(rLowS)[1]):
                hFac1tmp = (RL[k] - rLowS[j,i]) * recip_drF[k]
                hFac_loc = np.min([hFac1tmp, 1])
    #      o Impose minimum fraction and/or size (dimensional)
                if hFac_loc<hFacMnSz*0.5 or rLowS[j,i]>=0:
                    hFac1tmp = 0
                else:
                    hFac1tmp = np.max([hFac_loc, hFacMnSz])
    #      o Reduce the previous fraction : substract the outside fraction
    #        (i.e., beyond reference (=at rest) surface position rSurfS)
                hFac2tmp = ( RL[k]-0 )*recip_drF[k]
                hFac_loc = hFac1tmp - np.max([hFac2tmp, 0])
    #      o Impose minimum fraction and/or size (dimensional)
                if hFac_loc<hFacMnSz*0.5:
                    hFacS[k,j,i]=0
                else:
                    hFacS[k,j,i]=np.max([hFac_loc, hFacMnSz])

    return(hFacS)

def create_hFacW_grid(bathy,delR, hFacMin=0.2, hFacMinDr=5.0):
    # This is from MITgcm inside ini_masks_etc.F

    RU = -1 * np.cumsum(delR)
    RL = np.concatenate([[0], RU[:-1]])
    R_low = bathy
    drF = RL - RU
    recip_drF = 1 / drF

    rLowW = np.copy(bathy)
    for j in range(np.shape(rLowW)[0]):
        for i in range(1,np.shape(rLowW)[1]):
            rLowW[j,i] = np.max([R_low[j,i-1],R_low[j,i]])

    hFacW = np.zeros((len(delR), np.shape(bathy)[0], np.shape(bathy)[1]))
    for k in range(len(delR)):
        # if k%5==0:
        #     print('     - Calculating hFacW for depth cells '+str(k)+' to '+str(k+5))
        hFacMnSz = np.max([hFacMin, np.min([hFacMinDr * recip_drF[k], 1])])
        for j in range(np.shape(rLowW)[0]):
            for i in range(np.shape(rLowW)[1]):
                hFac1tmp = (RL[k] - rLowW[j,i]) * recip_drF[k]
                hFac_loc = np.min([hFac1tmp, 1])
    #      o Impose minimum fraction and/or size (dimensional)
                if hFac_loc<hFacMnSz*0.5 or rLowW[j,i]>=0:
                    hFac1tmp = 0
                else:
                    hFac1tmp = np.max([hFac_loc, hFacMnSz])
    #      o Reduce the previous fraction : substract the outside fraction
    #        (i.e., beyond reference (=at rest) surface position rSurfS)
                hFac2tmp = ( RL[k]-0 )*recip_drF[k]
                hFac_loc = hFac1tmp - np.max([hFac2tmp, 0])
    #      o Impose minimum fraction and/or size (dimensional)
                if hFac_loc<hFacMnSz*0.5:
                    hFacW[k,j,i]=0
                else:
                    hFacW[k,j,i]=np.max([hFac_loc, hFacMnSz])

    return(hFacW)

def create_surface_hFacC_grid(bathy, delR, hFacMin=0.2, hFacMinDr=5.0):
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

    # Define grids with same names as those in MITgcm
    RU = -1 * np.cumsum(delR)
    RL = np.concatenate([[0], RU[:-1]])
    R_low = bathy
    drF = RL - RU
    recip_drF = 1 / drF

    # Pythonize the above loops
    hFacC = np.zeros((np.shape(bathy)[0], np.shape(bathy)[1]))
    # for k in range(len(RL)):
    # if k%5==0:
    #     print('     - Calculating hFacC for depth cells '+str(k)+' to '+str(k+5))
    k=0
    hFacMnSz = np.max([hFacMin, np.min([hFacMinDr * recip_drF[k], 1])])
    for i in range(np.shape(bathy)[0]):
        for j in range(np.shape(bathy)[1]):
            #      o Non-dimensional distance between grid bound. and domain lower_R bound.
            hFac_loc = (RL[k] - R_low[i, j]) * recip_drF[k]
            #      o Select between, closed, open or partial (0,1,0-1)
            hFac_loc = np.min([np.max([hFac_loc, 0]), 1])
            #      o Impose minimum fraction and/or size (dimensional)
            if hFac_loc <= hFacMnSz * 0.5 or R_low[i, j] >= 0:
                hFacC[i, j] = 0
            else:
                hFacC[i, j] = np.max([hFac_loc, hFacMnSz])

    return(hFacC)