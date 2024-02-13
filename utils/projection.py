import numpy as np

def project_points_to_image_plane(object_points, extrinsic_matrix, intrinsic_matrix, distCoeffs=None):
    """
    Project 3D points to 2D image plane using intrinsic and extrinsic camera parameters.
    
    Parameters:
    - object_points: Nx1x3 numpy array of object points.
    - extrinsic_matrix: Extrinsic parameters matrix combining rotation and translation. 3x4 matrix.
    - intrinsic_matrix: Intrinsic camera matrix. 3x3 matrix.
    - distCoeffs: Distortion coefficients (not used in this simplified version).
    
    Returns:
    - projected_points: Nx2 numpy array of projected points on the image plane.
    """
    # Ensure object_points is in the correct shape (Nx3)
    object_points = object_points.reshape(-1, 3)
    
    # Convert object points to homogeneous coordinates (Nx4)
    object_points_homogeneous = np.hstack((object_points, np.ones((object_points.shape[0], 1)))) # Nx4 matrix
    
    # Project 3D points to the camera coordinate system using extrinsic parameters
    camera_points = np.dot(extrinsic_matrix, object_points_homogeneous.T)  # 4xN matrix
    
    # Apply the intrinsic parameters
    projected_points_homogeneous = np.dot(intrinsic_matrix, camera_points[:3, :])  # 3xN matrix
    
    # Convert from homogeneous coordinates to 2D
    projected_points = (projected_points_homogeneous[:2, :] / projected_points_homogeneous[2, :]).T  # Nx2 matrix
    
    return projected_points