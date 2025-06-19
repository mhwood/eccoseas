
from pyproj import Transformer
import numpy as np

def reproject_polygon(polygon_array, inputCRS, outputCRS, x_column=0, y_column=1):
    """
    Reprojects a 2D NumPy array of polygon coordinates from one coordinate
    reference system (CRS) to another using pyproj.

    Special handling is included for known coordinate order quirks in pyproj,
    especially when transforming between EPSG:4326 and EPSG:3413.

    Parameters:
    ----------
    polygon_array : ndarray
        A 2D NumPy array where each row is a coordinate [x, y] (or in other columns).
    inputCRS : int
        EPSG code of the input coordinate reference system.
    outputCRS : int
        EPSG code of the desired output coordinate reference system.
    x_column : int, optional
        Index of the column containing x-coordinates (default is 0).
    y_column : int, optional
        Index of the column containing y-coordinates (default is 1).

    Returns:
    -------
    output_polygon : ndarray
        A copy of the input polygon array with reprojected coordinates.
    """

    # Initialize the transformer from inputCRS to outputCRS
    transformer = Transformer.from_crs('EPSG:' + str(inputCRS), 'EPSG:' + str(outputCRS))

    # pyproj may mix up coordinate order depending on the CRS transformation
    # Known patterns are handled explicitly to ensure correctness

    if inputCRS == 4326 and outputCRS == 3413:
        # Input in (lat, lon) order, returns (x, y)
        x2, y2 = transformer.transform(polygon_array[:, y_column], polygon_array[:, x_column])
    elif inputCRS == 3413 and outputCRS == 4326:
        # Input in (x, y), returns (lat, lon) in reverse
        y2, x2 = transformer.transform(polygon_array[:, x_column], polygon_array[:, y_column])
    elif str(inputCRS).startswith('326') and outputCRS == 3413:
        # UTM to Polar Stereographic projection, also needs (lat, lon) input order
        x2, y2 = transformer.transform(polygon_array[:, y_column], polygon_array[:, x_column])
    else:
        # Fail fast if the transformation pair is not explicitly tested
        raise ValueError('Reprojection with this EPSG pair is not safe - no test for validity has been implemented.')

    # Ensure the results are NumPy arrays
    x2 = np.array(x2)
    y2 = np.array(y2)

    # Create a copy of the original array and update coordinates
    output_polygon = np.copy(polygon_array)
    output_polygon[:, x_column] = x2
    output_polygon[:, y_column] = y2

    return output_polygon