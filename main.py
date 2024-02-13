import numpy as np
import cv2
import scipy.linalg as la

from utils.config import load_config
from utils.visualization import plot_camera_and_world_frame,plot_camera_views
from utils.projection import project_points_to_image_plane
from utils.error import calculate_rms
from libs.homography_decomp import find_camera_extrinsics
from libs.pnp import pnp
from libs.traingulation import MultiViewTriangulation
from libs.bundle_adjustment_axisangle_fix import BundleAdjustmentAxisAngle
from libs.bundle_adjustment_quat_fix import BundleAdjustmentQuaternion




def main():

    checkerboard_points, num_cameras, intrinsic_matrix, dist_coeffs = load_config('data//checkerboard_camera_config.json')

    # Prepare the figure for plotting all cameras
    image_points_calib_list = []
    image_points_list = []
    reprojected_points_list = []
    camera_pose =[]

    # Solve homography for each camera 
    for cam in range(num_cameras):
        file_path = f'data/image-cam-{cam}-image.txt'  # Update this to your file path
        image_points = np.loadtxt(file_path, delimiter=',', usecols=(1, 2), max_rows=60).astype(np.float32)
        image_points_calib = image_points[:50]
        image_points_calib_list.append(image_points_calib)
        image_points_list.append(image_points)

        # Solve homography for the camera
        extrinsic_matrix = find_camera_extrinsics(checkerboard_points[:, 1:4].astype(np.float32), image_points_calib, intrinsic_matrix, dist_coeffs)
        camera_pose.append(extrinsic_matrix)

        # Project 3D checkerboard points back onto the image plane
        reprojected_points = project_points_to_image_plane(checkerboard_points[:, 1:4], extrinsic_matrix, intrinsic_matrix)
        reprojected_points_list.append(reprojected_points)    

    # Triangulate a single 3D point from its observations in multiple camera views
    triangulator = MultiViewTriangulation(intrinsic_matrix, camera_pose, image_points_list)
    triangulated_coordinates = triangulator.triangulate()
    triangulated_coordinates = np.vstack(triangulated_coordinates)
    #TODO: TRAINGULATION CHECK MATRICS

    # Bundle Adjustment
    triangulated_coordinates[:50] = checkerboard_points[:, 1:4] # make checkboard point fixed
    ba = BundleAdjustmentAxisAngle(intrinsic_matrix, camera_pose, image_points_list, triangulated_coordinates)
    # ba = BundleAdjustmentQuaternion(intrinsic_matrix, camera_pose, image_points_list, triangulated_coordinates)
    optimized_camera_params, optimized_points_3d = ba.run()




    # Results 

    # Save the triangulated 3D points and optimized 3D points to a file
    np.savetxt('results/Triangulation.txt', triangulated_coordinates, delimiter=',', fmt='%.2f')
    np.savetxt('results/Optimized_points.txt', optimized_points_3d, delimiter=',', fmt='%.2f')

    # Plot checkerboard points, camera in world frame
    # plot_camera_and_world_frame(checkerboard_points[:, 1:4], camera_pose)      # after homography
    # plot_camera_and_world_frame(triangulated_coordinates , camera_pose)        # after traingulation
    plot_camera_and_world_frame(optimized_points_3d, optimized_camera_params)    # after bundle adjustment

    # Visualize the original and reprojected image points for each camera
    plot_camera_views(image_points_calib_list, reprojected_points_list, num_cameras)  # after homography
    
    
    

if __name__ == "__main__":
    main()
