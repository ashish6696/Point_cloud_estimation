import numpy as np
import cv2
import scipy.linalg as la
def get_extrinsics_from_homography(homography, intrinsic_matrix):
    """
    Calculate the extrinsic camera matrix from a homography matrix and an intrinsic camera matrix.

    :param homography: A 3x3 homography matrix.
    :param intrinsic_matrix: A 3x3 intrinsic camera matrix.
    :return: A 3x4 extrinsic matrix consisting of a rotation matrix and a translation vector.
    """
    # Normalize the homography matrix with respect to the intrinsic matrix
    normalized_homography = np.linalg.inv(intrinsic_matrix) @ homography

    # Compute the normalized direction vectors from the normalized homography matrix
    direction_vector_1 = normalized_homography[:, 0] / np.linalg.norm(normalized_homography[:, 0])
    direction_vector_2 = normalized_homography[:, 1] / np.linalg.norm(normalized_homography[:, 1])
    
    # Compute the orthogonal vector to the plane spanned by the first two direction vectors
    orthogonal_vector = np.cross(direction_vector_1, direction_vector_2)

    # Construct a matrix using the direction vectors and the orthogonal vector
    matrix_M = np.column_stack([direction_vector_1, direction_vector_2, orthogonal_vector])

    # Perform Singular Value Decomposition (SVD) to enforce orthogonality
    U, _, VT = la.svd(matrix_M, full_matrices=False)

    # Enforce a proper rotation matrix by ensuring the determinant is 1
    rotation_matrix = U @ np.diag([1, 1, np.linalg.det(U @ VT)]) @ VT

    # Compute the scale factor based on the normalization
    scale_factor = 1 / np.linalg.norm(normalized_homography[:, 0])

    # Compute the translation vector using the scale factor
    translation_vector = scale_factor * normalized_homography[:, 2]

    # Combine the rotation matrix and the translation vector into the extrinsic matrix
    extrinsic_matrix = np.hstack((rotation_matrix, translation_vector.reshape(3, 1)))

    return extrinsic_matrix

def find_camera_extrinsics(checkerboard_points, image_points, intrinsic_matrix, dist_coeffs):
    """
    Use homography and its decomposition to find camera extrinsics.
    """
    # Ensure points are suitable for findHomography
    checkerboard_points_homogeneous = np.hstack((checkerboard_points[:, :2], np.ones((checkerboard_points.shape[0], 1))))
    
    # Find homography
    H, _ = cv2.findHomography(checkerboard_points_homogeneous[:, :2], image_points[:, :2])
    
    # Decompose the homography matrix
    extrinsics = get_extrinsics_from_homography(H, intrinsic_matrix)
    return extrinsics

 # other experiments
def select_correct_homography_decomposition(decompositions, intrinsic_matrix, image_points, checkerboard_points):
    """
    Select the correct homography decomposition based on the lowest reprojection error.
    
    Parameters:
    - decompositions: The possible rotations, translations, and normals from homography decomposition.
    - intrinsic_matrix: The camera's intrinsic matrix.
    - image_points: The observed 2D points in the image.
    - checkerboard_points: The corresponding 3D points in the world coordinate system.
    
    Returns:
    - The best extrinsic matrix (R|t) based on reprojection error.
    """
    min_error = np.inf
    best_extrinsic_matrix = None

    for rotation, translation, _ in zip(*decompositions):
        # Construct the extrinsic matrix
        extrinsic_matrix = np.hstack((rotation, translation.reshape(-1, 1)))
        
        # Project the checkerboard points using the current R, t
        projected_points = cv2.projectPoints(checkerboard_points, rotation, translation, intrinsic_matrix, distCoeffs=None)[0].squeeze()

        # Calculate reprojection error
        error = np.linalg.norm(image_points - projected_points, axis=1).mean()

        if error < min_error:
            min_error = error
            best_extrinsic_matrix = extrinsic_matrix

    return best_extrinsic_matrix