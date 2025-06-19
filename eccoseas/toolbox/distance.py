import numpy as np

def great_circle_distance(lon_ref, lat_ref, Lon, Lat):
    """
    Calculates the great-circle distance between a reference point and
    one or more destination points using the haversine formula.

    Parameters:
    ----------
    lon_ref : float
        Longitude of the reference point in degrees.
    lat_ref : float
        Latitude of the reference point in degrees.
    Lon : float or array-like
        Longitude(s) of the destination point(s) in degrees.
    Lat : float or array-like
        Latitude(s) of the destination point(s) in degrees.

    Returns:
    -------
    h : float or ndarray
        Great-circle distance(s) in meters between the reference point
        and the destination point(s).
    """
    earth_radius = 6371000  # Radius of the Earth in meters

    # Convert degrees to radians
    lon_ref_radians = np.radians(lon_ref)
    lat_ref_radians = np.radians(lat_ref)
    lons_radians = np.radians(Lon)
    lats_radians = np.radians(Lat)

    # Compute differences in coordinates
    lat_diff = lats_radians - lat_ref_radians
    lon_diff = lons_radians - lon_ref_radians

    # Haversine formula to calculate the central angle
    d = np.sin(lat_diff * 0.5) ** 2 + \
        np.cos(lat_ref_radians) * np.cos(lats_radians) * np.sin(lon_diff * 0.5) ** 2

    # Convert central angle to distance
    h = 2 * earth_radius * np.arcsin(np.sqrt(d))

    return h