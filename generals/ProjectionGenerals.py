"""local programs"""
from generals.ScannerGenerals import ScannerGenerals

"""public packages"""
import numpy as np
from joblib import Parallel, delayed


class ProjectionGenerals(ScannerGenerals):
    def __init__(self, source_to_iso_center, source_to_detector, detector_offset_x, detector_offset_y, source_offset_x,
                 source_offset_y, in_plane_angle, out_of_plane_angle, proj_dim, proj_spacing):
        # scanner settings
        ScannerGenerals.__init__(self, source_to_iso_center, source_to_detector, detector_offset_x, detector_offset_y,
                                 source_offset_x, source_offset_y, in_plane_angle, out_of_plane_angle,
                                 proj_dim[2])

        # projection settings
        self.proj_dim = proj_dim
        self.proj_spacing = proj_spacing
        self.proj_origin = - (proj_dim - 1) / 2 * proj_spacing

        self.gantry_angles = np.arange(0, 360, 360/proj_dim[2])  # 这里做了临时修改
        self.proj_info = None
        self.water_pre_correction_coefs = None

    def get_each_rotation_matrix(self, current_gantry_angle):
        """
        Generate the rotation matrix for transformation from detector index to physical location
        :return: 2D matrix
        """
        in_plane_angle = np.deg2rad(self.in_plane_angle)
        out_of_plane_angle = np.deg2rad(self.out_of_plane_angle)
        gantry_angle = np.deg2rad(current_gantry_angle)

        in_plane_angle_matrix = np.array([
            [np.cos(in_plane_angle), -np.sin(in_plane_angle), 0, 0],
            [np.sin(in_plane_angle), np.cos(in_plane_angle), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        out_of_plane_angle_matrix = np.array([
            [1, 0, 0, 0],
            [0, np.cos(out_of_plane_angle), -np.sin(out_of_plane_angle), 0],
            [0, np.sin(out_of_plane_angle), np.cos(out_of_plane_angle), 0],
            [0, 0, 0, 1]
        ])

        gantry_angle_matrix = np.array([
            [np.cos(gantry_angle), 0, np.sin(gantry_angle), 0],
            [0, 1, 0, 0],
            [-np.sin(gantry_angle), 0, np.cos(gantry_angle), 0],
            [0, 0, 0, 1]
        ])
        rotation_matrix = np.dot(np.dot(in_plane_angle_matrix, out_of_plane_angle_matrix), gantry_angle_matrix)
        return rotation_matrix.astype(np.float32)

    def get_translation_matrix(self):
        source_translation_matrix = np.array([
            [1, 0, 0, self.source_offset_x],
            [0, 1, 0, self.source_offset_y],
            [0, 0, 1, self.source_to_iso_center],
            [0, 0, 0, 1]

        ])

        detector_translation_matrix = np.array([
            [1, 0, 0, self.detector_offset_x],
            [0, 1, 0, self.detector_offset_y],
            [0, 0, 1, self.source_to_iso_center - self.source_to_detector],
            [0, 0, 0, 1]
        ])

        return source_translation_matrix, detector_translation_matrix

    def get_each_info(self, gantry_angle):
        source_translation_matrix, detector_translation_matrix = self.get_translation_matrix()

        # source position
        current_sp = (np.dot(
            self.get_each_rotation_matrix(gantry_angle),
            np.dot(
                source_translation_matrix,
                np.array([0, 0, 0, 1]).reshape([-1, 1])
            )
        ).T)[0, :3]

        # detector position
        x = (np.arange(self.proj_dim[0]) + 0.5 - self.proj_dim[0] / 2) * self.proj_spacing[0]
        y = (np.arange(self.proj_dim[1]) + 0.5 - self.proj_dim[1] / 2) * self.proj_spacing[1]

        grid_x, grid_y = np.meshgrid(x, y)
        detector_position = np.vstack((
            grid_x.reshape(-1),
            grid_y.reshape(-1),
            np.zeros(self.proj_dim[0] * self.proj_dim[1]),
            np.ones(self.proj_dim[0] * self.proj_dim[1])
        ))

        current_dp = (np.dot(
            self.get_each_rotation_matrix(gantry_angle),
            np.dot(
                detector_translation_matrix,
                detector_position
            )
        ).T)[:, :3]

        # info structure: [sp_x, sp_y, sp_z, dp_x, dp_y, dp_z]
        info = np.hstack((
            np.array([current_sp] * current_dp.shape[0]),
            current_dp,
        ))
        return info.astype(np.float32)

    def get_all_info(self):
        infos = Parallel(n_jobs=30)(
            delayed(self.get_each_info)(angle) for angle in self.gantry_angles)
        proj_info = np.array(infos)
        return proj_info.astype(np.float32)

    def get_each_rotation_matrices(self, param):
        pass


if __name__ == "__main__":
    pass
    # projections = ProjectionGenerals(
    #     source_to_iso_center=500,
    #     source_to_detector=1000,
    #     detector_offset_x=0,
    #     detector_offset_y=0,
    #     source_offset_x=0,
    #     source_offset_y=0,
    #     in_plane_angle=0,
    #     out_of_plane_angle=0,
    #     projection_img=sitk.ReadImage(
    #         r"D:\linyuejie\temp_files\szj_gain_correction\20241029\water_phantom_projections\bin0_corrected_water_phantom_projection.mhd"))