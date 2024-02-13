import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation as R


class BundleAdjustmentQuaternion:
    def __init__(self, intrinsic_matrix, extrinsic_matrices, points_cam_views, triangulated_coordinates):
        self.K = intrinsic_matrix
        self.extrinsics = extrinsic_matrices  # Assuming extrinsic_matrices are rotation matrices and translation vectors
        self.points_2d = [np.array(view) for view in points_cam_views]  # list of N x 2 matrices for each camera view
        self.point_3d_estimate = triangulated_coordinates  # Initial estimate of 3D points
        self.num_cameras = len(extrinsic_matrices)
        self.num_points = len(triangulated_coordinates)
        self.all_params = self.pack_parameters()

    def matrix_to_quat(self,rotation_matrix):
        """
        Converts a rotation matrix to a quaternion.

        Parameters:
        - rotation_matrix: A 3x3 rotation matrix.

        Returns:
        - A quaternion [x, y, z, w] representing the same rotation.
        """
        # Convert rotation matrix to quaternion
        rotation = R.from_matrix(rotation_matrix)
        quat = rotation.as_quat()  # Returns quaternion [x, y, z, w]
        return quat

    def quat_to_matrix(self,quat):
        """
        Converts a quaternion to a rotation matrix 

        Parameters:
        - rotation_vector: quaternion.

        Returns:
        - The corresponding rotation matrix.
        """
        # Convert quaternion to rotation matrix
        rotation = R.from_quat(quat)
        R_mat = rotation.as_matrix()
        return R_mat
    
    def projection(self, r_t):
        """
        Projects 3D points onto 2D using camera parameters.

        Parameters:
        - r_t: Either a quaternion + translation (7 elements) or an extrinsic matrix (3x4).

        Returns:
        - 2D projections of 3D points as an array.
        """
        if r_t.shape ==(7,):
            # Extract rotation and translation from r_t_vec
            rvec = r_t[:4]
            tvec = r_t[3:]
            # Convert rotation vector to rotation matrix
            R_mat = self.quat_to_matrix(rvec)

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
        """
        Packs camera extrinsics and 3D point coordinates into a single array.

        Converts each camera's extrinsic matrix into quaternion and translation vector,
        then concatenates these with all 3D point coordinates into one flattened array.

        Returns:
        - A single flattened array containing all camera parameters followed by
        3D point coordinates.
        """
        cam_params = []
        for extrinsic in self.extrinsics:
            # The rotation part is the upper left 3x3 submatrix of the 3x4 extrinsic matrix
            R = extrinsic[:, :3]
            # The translation vector is the last column of the extrinsic matrix
            t = extrinsic[:, 3]

            # Convert rotation matrix to quaternion
            quat = self.matrix_to_quat(R)
            
            # Concatenate quaternion representation with translation vector
            cam_param = np.hstack((quat.reshape(4), t))
            cam_params.append(cam_param)
        
        # Flatten the camera parameters list into a single array
        cam_params_flat = np.hstack(cam_params)
        
        # Combine flattened camera parameters with flattened 3D point coordinates
        all_params = np.hstack((cam_params_flat,self.point_3d_estimate.flatten()))
        
        # all_params = [r_t_vec0, ... ,r_t_vec5, point0 , ..., point59]
        return all_params
    
    def unpack_parameters(self,params):
        """
        Unpacks parameters into camera extrinsics and 3D point coordinates.

        Parameters:
        - params: Flattened array of camera parameters and 3D point coordinates.

        Returns:
        - A list of extrinsic matrices (3, 4) for each camera, combining rotation and translation.
        - An array of 3D point coordinates (N, 3), where N is the number of points.
        """
        r_t_vecs = params[:self.num_cameras * 7].reshape((self.num_cameras, 7))
        points_3d = params[self.num_cameras * 7:].reshape((self.num_points, 3))
        
        extrinsics = []
        for r_t_vec in r_t_vecs:
            quat = r_t_vec[:4]
            t = r_t_vec[4:]
            R = self.quat_to_matrix(quat)  # Convert back to rotation matrix
            extrinsic = np.hstack((R, t.reshape((3, 1))))  # Reconstruct transformation matrix
            extrinsics.append(extrinsic)
        
        return extrinsics, points_3d
    
    def combine_fixpoint(self, params, params_fix):
        """
        Combine parameters from the optimized set and the fixed set into a full parameter set.

        :param params: The optimized parameters, excluding the first 50 fixed points.
        :param params_fix: The parameters of the first 50 fixed points.
        :return: The combined full parameter set.
        """
        num_camera_params = len(self.extrinsics) * 7  # 7 params per camera: 4 for rotation (quat), 3 for translation
        cam_params = params[:num_camera_params]
        non_fixed_point_params = params[num_camera_params:]
        combined_params = np.hstack((cam_params, params_fix, non_fixed_point_params)) 
        return combined_params

    def separate_fixpoints(self, all_params):
        """
        Separate the fixed point parameters from the full parameter set.

        :param all_params: The combined full parameter set, including camera parameters and all 3D points.
        :return: A tuple containing:
            - Camera parameters
            - Fixed point parameters
            - Non-fixed point parameters
        """
        num_camera_params = len(self.extrinsics) * 7  # 7 params per camera
        num_fixed_points = 50  # Number of fixed points
        num_point_params = 3  # Parameters per point (x, y, z)


        cam_params = all_params[:num_camera_params]
        start_points_index = num_camera_params
        fixed_points_length = num_fixed_points * num_point_params
        fixed_point_params = all_params[start_points_index:start_points_index + fixed_points_length]
        non_fixed_point_params = all_params[start_points_index + fixed_points_length:]
        param_opt = np.hstack((cam_params, non_fixed_point_params))
        
        return param_opt, fixed_point_params

    
    def residuals(self,params,params_fix):
        """
        Computes residuals for bundle adjustment optimization.

        Parameters:
        - params: Array of current optimized parameters.
        - params_fix: Array of fixed parameters not being optimized.

        Returns:
        - Flattened array of residuals between observed and projected points.
        """
        # print(params.shape) # num_cameras * 7 + num_points * 3
        # Unpack parameters
        params = self.combine_fixpoint(params,params_fix)
        extrinsics, points_3d = self.unpack_parameters(params)
        residuals = []
        for i in range(self.num_cameras):
            projected_points = self.projection(extrinsics[i])
            residuals.append(projected_points - self.points_2d[i])
            # print((projected_points-point2d_views[i]).shape) # 60x2

        # print(np.hstack(residuals).ravel())
        return np.hstack(residuals).ravel() #600  = num_points * 2 * num_cameras

    def build_bundle_adjustment_sparsity(self):
        """
        Constructs the sparsity pattern of the Jacobian for bundle adjustment.

        Defines which parameters affect which observations to speed up Jacobian computations.

        Returns:
        - A sparse matrix representing the Jacobian's sparsity pattern.
        """
        m = self.num_points * 2 * self.num_cameras
        n = self.num_cameras * 7 +  (self.num_points - 50) * 3
        A = lil_matrix((m, n), dtype=int)
        point_offset = self.num_cameras * 7
        for i in range(self.num_points - 50):
            for j in range(self.num_cameras):
                row_start = 2 * (j *  self.num_points  + i )     # row = [cam0_point0, cam0_point1 .... cam0_point59,cam1_point0 ,....   cam4_point59]
                row_end   = 2 * (j *  self.num_points  + (i+1))  # 2 raw each for point's u,v
                A[row_start :row_end, 7 * j:7 * (j + 1)] = 1
                A[row_start :row_end, point_offset + 3 * i:point_offset + 3 * (i + 1)] = 1
        return A


    def run(self):
        """
        Executes the bundle adjustment optimization process.

        Combines, optimizes parameters using least squares, and updates class attributes with optimized values.

        Returns:
        - Updated extrinsics and 3D point estimates after optimization.
        """
        # Pack initial parameters
        initial_params = self.pack_parameters()
        initial_params, params_fix = self.separate_fixpoints(initial_params)

        # Jacobian sparsity pattern
        A = self.build_bundle_adjustment_sparsity()

        # Optimize using Levenberg-Marquardt algorithm
        result = least_squares(self.residuals, initial_params, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-4, method='trf', args=(params_fix,))

        # Unpack optimized parameters
        params = self.combine_fixpoint(result.x, params_fix)
        self.extrinsics, self.point_3d_estimate = self.unpack_parameters(params)

        return self.extrinsics,self.point_3d_estimate
