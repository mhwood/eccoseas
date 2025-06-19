toolbox
=======

The toolbox of the eccoseas package has a few helpful modules when working with
various datasets.


distance
^^^^^^^^
This module defines a function `great_circle_distance` that calculates
the shortest distance over the Earth's surface between
a fixed reference point and one or more target coordinates using the haversine
formula. This method accounts for the Earth's curvature and returns distances
in meters. 

reprojection
^^^^^^^^^^^^

This module defines a function `reproject_polygon` that reprojects
a 2D array of polygon coordinates from one coordinate reference system (CRS)
to another using the pyproj library. Due to known inconsistencies in coordinate
order during certain transformations (e.g., EPSG:4326 to 3413), the function
includes conditional logic to handle specific EPSG code pairs safely. As of now, only
EPSG 4326, 3413, and 326?? are handled, but other CRSs can be added easily
with the addition of new if statement blocks.












