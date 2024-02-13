import cv2
import numpy as np
def pnp(checkerboard_points, image_points, intrinsic_matrix, dist_coeffs):
    """
    Solve Perspective-n-Point (PnP) problem for the given checkerboard points and image points.
    
    Parameters:
    - checkerboard_points: Nx3 numpy 
    array of checkerboard points in the world frame.
    - image_points: Nx2 numpy array of image points in the camera frame.
    - intrinsic_matrix: Intrinsic camera matrix. 3x3 matrix.
    - dist_coeffs: Distortion coefficients. 1x5 matrix.
    
    Returns:
    - extrinsic_matrix: 3x4 numpy array of extrinsic parameters (rotation and translation).
    """
    # Solve PnP problem
    retval, rvec, tvec = cv2.solvePnP(checkerboard_points, image_points, intrinsic_matrix, dist_coeffs)
    # Convert rotation vector to rotation matrix
    R, _ = cv2.Rodrigues(rvec)
    # Apply the extrinsic parameters (rotation and translation)
    extrinsic_matrix = np.hstack((R, tvec))   # [R | t] 3x4 matrix

    return extrinsic_matrix