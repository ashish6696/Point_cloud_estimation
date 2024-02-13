import numpy as np
def calculate_rms(checkerboard_points, triangulated_points):
    """
    Calculate the Root Mean Square (RMS) error between checkerboard points and triangulated points.

    Args:
    - checkerboard_points: numpy array of shape (N, 3) containing checkerboard 3D points
    - triangulated_points: numpy array of shape (N, 3) containing triangulated 3D points

    Returns:
    - rms: the RMS error value
    """
    # Ensure the inputs are numpy arrays
    checkerboard_points = np.array(checkerboard_points)
    triangulated_points = np.array(triangulated_points)
    
    # Calculate the squared differences between corresponding points
    squared_diff = np.sum((checkerboard_points - triangulated_points) ** 2, axis=1) 
    
    # Calculate the mean squared error
    mean_squared_error = np.mean(squared_diff)
    
    # Calculate the RMS error
    rms = round(np.sqrt(mean_squared_error),4)
    return rms