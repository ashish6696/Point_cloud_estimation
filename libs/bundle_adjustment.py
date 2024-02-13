import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
import cv2

class BundleAdjustmentAxisAngle:
    def __init__(self, intrinsic_matrix, extrinsic_matrices, points_cam_views, triangulated_coordinates):
        self.K = intrinsic_matrix
        self.extrinsics = extrinsic_matrices  # Assuming extrinsic_matrices are rotation matrices and translation vectors
        self.points_2d = [np.array(view) for view in points_cam_views]  # list of N x 2 matrices for each camera view
        self.point_3d_estimate = triangulated_coordinates  # Initial estimate of 3D points
        self.num_cameras = len(extrinsic_matrices)
        self.num_points = len(triangulated_coordinates)
        self.all_params = self.pack_parameters()

    def matrix_to_axis(self,rotation_matrix):
        rotation_vector, _ = cv2.Rodrigues(rotation_matrix)
        return rotation_vector

    def axis_to_matrix(self,rotation_vector):
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        return rotation_matrix
    
    def projection(self, r_t):
        if r_t.shape ==(6,):
            # Extract rotation and translation from r_t_vec
            rvec = r_t[:3]
            tvec = r_t[3:]
            # Convert rotation vector to rotation matrix
            R_mat = self.axis_to_matrix(rvec)

        else:
            R_mat = r_t[:,:3]
            tvec  = r_t[:,3]
        # Project points
        projected_points = []
        for point in self.point_3d_estimate:
            # Transform point to camera coordinate system
            point_cam = R_mat @ point + tvec
            # Normalize
            point_cam_normalized = point_cam / point_cam[2]
            # Project points using intrinsic matrix
            p_img = self.K @ point_cam_normalized
            projected_points.append(p_img[:2])
        
        projected_points = np.array(projected_points)
        return projected_points

    def pack_parameters(self):
        cam_params = []
        for extrinsic in self.extrinsics:
            # The rotation part is the upper left 3x3 submatrix of the 3x4 extrinsic matrix
            R = extrinsic[:, :3]
            # The translation vector is the last column of the extrinsic matrix
            t = extrinsic[:, 3]

            # Convert rotation matrix to axis-angle representation
            axis_angle = self.matrix_to_axis(R)
            
            # Concatenate axis-angle representation with translation vector
            cam_param = np.hstack((axis_angle.reshape(3), t))
            cam_params.append(cam_param)
        
        # Flatten the camera parameters list into a single array
        cam_params_flat = np.hstack(cam_params)
        
        # Combine flattened camera parameters with flattened 3D point coordinates
        all_params = np.hstack((cam_params_flat,self.point_3d_estimate.flatten()))
        
        # all_params = [r_t_vec0, ... ,r_t_vec5, point0 , ..., point59]
        return all_params
    
    def unpack_parameters(self,params):
        r_t_vecs = params[:self.num_cameras * 6].reshape((self.num_cameras, 6))
        points_3d = params[self.num_cameras * 6:].reshape((self.num_points, 3))
        
        extrinsics = []
        for r_t_vec in r_t_vecs:
            axis_angle = r_t_vec[:3]
            t = r_t_vec[3:]
            R = self.axis_to_matrix(axis_angle)  # Convert back to rotation matrix
            extrinsic = np.hstack((R, t.reshape((3, 1))))  # Reconstruct transformation matrix
            extrinsics.append(extrinsic)
        
        return extrinsics, points_3d
    
    def residuals(self,params):
        # print(params.shape) # 210 = num_cameras * 6 + num_points * 3
        # Unpack parameters
        extrinsics, points_3d = self.unpack_parameters(params)
        residuals = []
        for i in range(self.num_cameras):
            projected_points = self.projection(extrinsics[i])
            residuals.append(projected_points - self.points_2d[i])
            # print((projected_points-point2d_views[i]).shape) # 60x2

        # print(np.hstack(residuals).ravel())
        return np.hstack(residuals).ravel() #600  = num_points * 2 * num_cameras

    def build_bundle_adjustment_sparsity(self):
        m =  self.num_points * 2 * self.num_cameras
        n = self.num_cameras * 6 +  self.num_points * 3
        A = lil_matrix((m, n), dtype=int)
        point_offset = self.num_cameras * 6
        for i in range(self.num_points):
            for j in range(self.num_cameras):
                row_start = 2 * (j *  self.num_points + i )     # row = [cam0_point0, cam0_point1 .... cam0_point59,cam1_point0 ,....   cam4_point59]
                row_end   = 2 * (j *  self.num_points + (i+1))  # 2 raw each for point's u,v
                A[row_start :row_end, 6 * j:6 * (j + 1)] = 1
                A[row_start :row_end, point_offset + 3 * i:point_offset + 3 * (i + 1)] = 1
        return A


    def run(self):
        # Pack initial parameters
        initial_params = self.pack_parameters()

        # Jacobian sparsity pattern
        A = self.build_bundle_adjustment_sparsity()

        # Optimize using Levenberg-Marquardt algorithm
        result = least_squares(self.residuals, initial_params, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-4, method='trf', args=())

        # Unpack optimized parameters
        self.extrinsics, self.point_3d_estimate = self.unpack_parameters(result.x)

        return self.extrinsics,self.point_3d_estimate
