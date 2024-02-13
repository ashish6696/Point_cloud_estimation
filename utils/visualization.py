import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
def plot_camera_and_world_frame(checkerboard_points, camera_poses):
    """
    Visualizes the world frame, camera frames, and checkerboard points in 3D using Plotly.
    
    Parameters:
    - checkerboard_points: Nx3 numpy array of checkerboard points in the world frame.
    - camera_poses: List of 4x4 numpy arrays, each representing the transformation matrix from the world to a camera frame.
    - camera_matrix: Camera intrinsic matrix, not directly used in this visualization but required for a complete setup.
    """
    fig = go.Figure()

    # Define world axes
    world_axes = {
        'x': {'point': np.array([0.3, 0, 0]), 'color': 'red'},
        'y': {'point': np.array([0, 0.3, 0]), 'color': 'green'},
        'z': {'point': np.array([0, 0, 0.3]), 'color': 'blue'}}

    # Plot world axes
    for axis, info in world_axes.items():
        fig.add_trace(go.Scatter3d(x=[0, info['point'][0]], y=[0, info['point'][1]], z=[0, info['point'][2]],
                                   mode='lines', line=dict(color=info['color'], width=5), name=f'World {axis.upper()}'))

    # Plot checkerboard points
    fig.add_trace(go.Scatter3d(x=checkerboard_points[:, 0], y=checkerboard_points[:, 1], z=checkerboard_points[:, 2],
                               mode='markers', marker=dict(size=5, color='blue'), name='Checkerboard Points'))

    # Plot each camera frame
    for i, pose in enumerate(camera_poses):
        pose = np.vstack([pose, [0, 0, 0, 1]])

        pose = np.linalg.inv(pose)


        camera_position = pose[:3, 3]
        R = pose[:3, :3]  # Extract rotation matrix

        # Calculate and plot camera axes
        axis_length = 0.1
        axes_colors = ['red', 'green', 'blue']
        for axis, color in zip([R[:, 0], R[:, 1], R[:, 2]], axes_colors):
            world_axis = camera_position + axis_length * axis
            fig.add_trace(go.Scatter3d(x=[camera_position[0], world_axis[0]], y=[camera_position[1], world_axis[1]], z=[camera_position[2], world_axis[2]],
                                       mode='lines', line=dict(color=color, width=4), name=f'Cam {i} {color} axis'))

        # Add camera position marker
        fig.add_trace(go.Scatter3d(x=[camera_position[0]], y=[camera_position[1]], z=[camera_position[2]],
                                   mode='markers+text', marker=dict(size=5, color='black'), name=f'Camera {i}',
                                   text=[f'Cam {i}'], textposition="bottom center"))

    # Setting the scene
    fig.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
                      title='3D Visualization of World Frame, Camera Frames, and Checkerboard Points')

    fig.show()

def plot_camera_views(image_points_list, reprojected_points_list, num_cameras):
    """
    Visualizes the original and reprojected image points for each camera using Matplotlib.
    
    Parameters:
    - original_image_points: List of 2D numpy arrays, each containing the original image points for a camera.
    - reprojected_image_points: List of 2D numpy arrays, each containing the reprojected image points for a camera.
    - num_cameras: Number of cameras.
    """
    # Prepare the figure for plotting all cameras
    fig, axs = plt.subplots(2, 5, figsize=(20, 8))  # Adjust subplot layout as needed
    for cam in range(num_cameras):
        # Plot original image points for the camera
        image_points_ = image_points_list[cam]
        ax = axs[0, cam]
        ax.scatter(image_points_[:, 0], image_points_[:, 1], color='blue', label='Original Image Points')
        ax.set_title(f'Cam {cam} Original')
        ax.axis('equal')

        # Plot reprojected points for the camera
        reprojected_points_ = reprojected_points_list[cam]
        ax = axs[1, cam]
        ax.scatter(reprojected_points_[:, 0], reprojected_points_[:, 1], color='red', label='Reprojected Points', alpha=0.6)
        ax.set_title(f'Cam {cam} Reprojected')
        ax.axis('equal')
    plt.tight_layout()
    plt.show()