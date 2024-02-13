import numpy as np

class MultiViewTriangulation:
    def __init__(self, intrinsic_matrix, extrinsic_matrices, points_cam_views):
        self.K = intrinsic_matrix
        self.extrinsic_matrices = extrinsic_matrices
        self.points_cam_views = points_cam_views

    def unproject_points(self, point, camera_matrix):
        """
        Converts 2D image points to 3D rays in world coordinates.

        Parameters:
        - point: The 2D image point coordinates (x, y).
        - camera_matrix: Camera matrix including rotation and translation.

        Returns:
        - A list of tuples representing unprojected rays in world coordinates, 
        each tuple contains the ray origin and direction.
        """
        unprojected_rays = []
        x, y = point[0], point[1]
        homog_point = np.array([x, y, 1.0])
        
        # Unproject point
        ray = np.linalg.inv(self.K).dot(homog_point)
        ray /= np.linalg.norm(ray)  # Normalize the ray direction

        # Calculate the camera center in world coordinates
        camera_center_world = -camera_matrix[:, :3].T.dot(camera_matrix[:, 3]) # take a one dim example to understand the reasoning

        # Rotate the ray direction from camera to world coordinates
        ray_world = camera_matrix[:, :3].T.dot(ray)  

        # Store the ray as a tuple (origin, direction)
        unprojected_rays.append((camera_center_world, ray_world))

        return unprojected_rays
    
    def triangulate(self):
        """
        Triangulates 3D points from their projections in multiple camera views.

        Returns:
        - A list of triangulated 3D point coordinates.
        """
        triangulated_points = []
        num_points = len(self.points_cam_views[0])

        for point_idx in range(num_points):  # Assuming 60 points indexed from 0 to 59
            A = []
            b = np.zeros((3, 1))

            for cam_idx, camera_matrix in enumerate(self.extrinsic_matrices):
                # Retrieve the corresponding 2D point for each camera view
                point_2d = self.points_cam_views[cam_idx][point_idx][1:]
                ray_origin, ray_direction = self.unproject_points(self.points_cam_views[cam_idx][point_idx], camera_matrix)[0]

                # Compute Ai and bi components for the least squares system
                # Ai is derived from the outer product of the ray direction
                I = np.eye(3)
                Ai = I - np.outer(ray_direction, ray_direction)  # (I- d*d.T) 3*3 Matrix
                A.append(Ai)                                     

                # bi is the projection of the camera center onto the plane orthogonal to the ray direction
                bi = Ai.dot(ray_origin)  # (I- d*d.T)*C 3*1 VECTOR
                b= np.concatenate((b, bi.reshape(-1, 1)), axis=0)  # 3x1 vector
                

            # Solve the least squares problem using numpy's lstsq function
            A = np.vstack(A)   # (n*3)x3 matrix
            b = b[3:,:]        # (n*3)x1 matrix

            P, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None) # AX=B   X is 3x1 vector

            # append points location as a list
            triangulated_points.append(P.flatten())  

        return triangulated_points
    

 

def main():
    pass

if __name__ == "__main__":
    main()
