import json
import numpy as np
from utils.coord import checkerboard
def load_config(config_file_path):
    """
    Load the configuration from the given JSON file.

    Parameters:
    - config_file_path: Path to the JSON configuration file.

    Returns:
    - checkerboard_points: Nx3 numpy array of checkerboard points in the world frame.
    - camera_matrix: Intrinsic camera matrix. 3x3 numpy array.
    - dist_coeffs: Distortion coefficients. 1x4 numpy array.
    """
    # Load the configuration from the JSON file
    with open(config_file_path, 'r') as f:
        config = json.load(f)
    # Extract checkerboard configuration
    checkerboard_config = config['checkerboard']
    width = checkerboard_config['width']
    height = checkerboard_config['height']
    cols = checkerboard_config['cols']
    rows = checkerboard_config['rows']   

    # Extract camera intrinsic parameters
    camera_config = config['camera']
    focal_length = camera_config['focal_length']
    principal_point = tuple(camera_config['principal_point'])
    num_cameras = camera_config['num_cameras']

    # Generate world coordinates for checkerboard points
    checkerboard_points = checkerboard(width, height, cols, rows)

    camera_matrix = np.array([[focal_length, 0, principal_point[0]],
                              [0, focal_length, principal_point[1]],
                              [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.zeros(4)  # Assuming no lens distortion

    return checkerboard_points,num_cameras, camera_matrix, dist_coeffs