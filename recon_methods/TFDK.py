# import array_api_compat.numpy as xp
import array_api_compat.cupy as xp
# import array_api_compat.torch as xp

import matplotlib.pyplot as plt
import numpy as np
import parallelproj
from tqdm import tqdm
import time

from generals.ProjectionGenerals import ProjectionGenerals
from generals.ReconGenerals import ReconGenerals


class TFDKProjection(ProjectionGenerals):
    def __init__(self, source_to_iso_center, source_to_detector, detector_offset_x, detector_offset_y, source_offset_x,
                 source_offset_y, in_plane_angle, out_of_plane_angle, proj_dim, proj_spacing):
        ProjectionGenerals.__init__(self, source_to_iso_center, source_to_detector, detector_offset_x,
                                    detector_offset_y, source_offset_x,
                                    source_offset_y, in_plane_angle, out_of_plane_angle, proj_dim, proj_spacing)

    def get_each_rotation_matrices(self, current_gantry_angles):
        """
        Generate the rotation matrix for transformation from detector index to physical location
        :return: 2D matrix
        """
        in_plane_angle = np.deg2rad(self.in_plane_angle)
        out_of_plane_angle = np.deg2rad(self.out_of_plane_angle)
        gantry_angles = np.deg2rad(current_gantry_angles)

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

        gantry_angle_matrix = np.array([[
                [np.cos(angle), 0, np.sin(angle), 0],
                [0, 1, 0, 0],
                [-np.sin(angle), 0, np.cos(angle), 0],
                [0, 0, 0, 1]
        ] for angle in gantry_angles])

        rotation_matrix = np.matmul(in_plane_angle_matrix, out_of_plane_angle_matrix)
        final_matrix = xp.asarray(np.matmul(rotation_matrix, gantry_angle_matrix))

        return final_matrix


class TFDKReconstruction(ReconGenerals):
    def __init__(self, projection:TFDKProjection, projection_img, recon_dim, recon_spacing):
        ReconGenerals.__init__(self, projection, projection_img, recon_dim, recon_spacing)
        self.recon_img = xp.zeros(self.recon_dim)
        self.rebin_info = np.zeros([np.prod(self.projection.proj_dim), 3])
        self.projection_rebin()
        self.get_tfdk_weights()

        # cpu coefs in extended z (xFDK)
        self.cz = np.zeros(self.recon_dim[2])
        self.r = np.zeros(self.recon_dim[:2])
        self.weight_t = np.zeros(self.recon_dim)
        # gpu coefs in extend z (xFDK)
        self.theta_1 = xp.zeros(self.recon_dim)
        self.theta_2 = xp.zeros(self.recon_dim)
        self.delta_theta = xp.zeros(self.recon_dim)
        self.init_extend_z()

    def init_extend_z(self):
        self.get_coefs_in_extend_z()
        self.get_weight_transition_in_extend_z()

    def get_coefs_in_extend_z(self):
        x = ((np.arange(self.recon_dim[0]) + 0.5) - self.recon_dim[0] / 2) * self.recon_spacing[0]
        y = -((np.arange(self.recon_dim[1]) + 0.5) - self.recon_dim[1] / 2) * self.recon_spacing[1]
        z = np.abs(((np.arange(self.recon_dim[2]) + 0.5) - self.recon_dim[2] / 2) * self.recon_spacing[2])

        grid_x, grid_y = np.meshgrid(x, y)
        self.r = np.sqrt(grid_x ** 2 + grid_y ** 2)
        phi = xp.asarray(np.arctan2(-grid_x, grid_y))

        beta_max, gamma_max, _ = np.arctan(((self.projection.proj_dim - 1) / 2 * self.projection.proj_spacing) / self.projection.source_to_detector)
        c = 1 / np.tan(gamma_max)
        self.cz = c * z

        xp_cz = xp.array([np.full([self.recon_dim[0], self.recon_dim[1]], i) for i in self.cz])
        rf = self.projection.source_to_iso_center
        xp_r = xp.tile(xp.asarray(self.r), (self.recon_dim[2], 1, 1))

        self.theta_1 = xp.tile(phi - xp.pi / 2, (self.recon_dim[0], 1, 1))
        self.theta_2 = xp.tile(phi + xp.pi / 2, (self.recon_dim[0], 1, 1))
        self.delta_theta = xp.arcsin((rf-xp_cz) / xp_r) - xp.arctan(xp.sqrt(xp_r ** 2 - (rf-xp_cz) ** 2) / xp_cz)
        self.delta_theta[(xp_r <= xp.abs(rf-xp_cz))] = (xp.pi / 2 * xp.sign(rf-xp_cz))[(xp_r <= xp.abs(rf-xp_cz))]

    def get_weight_transition_in_extend_z(self):
        r = np.tile(self.r, (self.recon_dim[2], 1, 1))
        cz = np.array([np.full([self.recon_dim[0], self.recon_dim[1]], i) for i in self.cz])

        rf = self.projection.source_to_iso_center
        beta_max, gamma_max, _ = np.arctan(((self.projection.proj_dim - 1) / 2 * self.projection.proj_spacing) / self.projection.source_to_detector)
        rm = self.projection.source_to_iso_center * np.sin(beta_max)
        delta_r = rm ** 2 / (2 * rf)
        r0 = rf - cz - delta_r
        weight_t = np.zeros(self.recon_dim)
        option_1 = (r >= r0 - delta_r) & (r < r0 + delta_r)
        weight_t[option_1] = 1 + np.sin(np.pi / 2 * ((r - r0) / delta_r))[option_1]
        option_2 = (r >= r0 + delta_r)
        weight_t[option_2] = 2
        weight_t /= 2
        self.weight_t = weight_t

    def get_extend_z_weights(self, theta_angle):  # angle = theta angle
        theta = xp.deg2rad(theta_angle)

        weight_ps = xp.zeros(self.recon_dim)
        theta_minus_2pi = theta-xp.pi*2
        theta_1_minus_delta_theta = self.theta_1 - self.delta_theta
        theta_1_add_delta_theta = self.theta_1 + self.delta_theta
        theta_2_minus_delta_theta = self.theta_2 - self.delta_theta
        theta_2_add_delta_theta = self.theta_2 + self.delta_theta

        option_1 = (theta >= theta_1_minus_delta_theta)
        option_2 = (theta < theta_1_add_delta_theta) | ((theta_minus_2pi < theta_1_add_delta_theta) & (theta_minus_2pi >= theta_1_minus_delta_theta))
        judgement = option_1 & option_2
        modified_theta = xp.ones(self.recon_dim) * theta
        modified_theta[((theta_minus_2pi < theta_1_add_delta_theta) & (theta_minus_2pi >= theta_1_minus_delta_theta))] = theta_minus_2pi
        weight_ps[judgement] = 1 + xp.sin(np.pi / 2 * ((modified_theta - self.theta_1) / self.delta_theta))[judgement]

        option_1 = (theta >= theta_1_add_delta_theta)
        option_2 = (theta < theta_2_minus_delta_theta) | ((theta_minus_2pi < theta_2_minus_delta_theta) & (theta_minus_2pi >= theta_1_add_delta_theta))
        judgement = option_1 & option_2
        weight_ps[judgement] = 2

        option_1 = (theta >= theta_2_minus_delta_theta)
        option_2 = (theta < theta_2_add_delta_theta) | ((theta_minus_2pi < theta_2_add_delta_theta) & (theta_minus_2pi >= theta_2_minus_delta_theta))
        judgement = option_1 & option_2
        modified_theta = xp.ones(self.recon_dim) * theta
        modified_theta[((theta_minus_2pi < theta_2_add_delta_theta) & (theta_minus_2pi >= theta_2_minus_delta_theta))] = theta_minus_2pi
        weight_ps[judgement] = 1 - xp.sin(np.pi / 2 * ((modified_theta - self.theta_2) / self.delta_theta))[judgement]

        weight_fs = 1
        weight_c = (1 - xp.asarray(self.weight_t)) * weight_fs + xp.asarray(self.weight_t) * weight_ps
        return weight_c.transpose([2, 0, 1])

    def get_each_info(self, gantry_angle):
        current_index = int(gantry_angle / (360 / self.projection.num_of_projections))
        source_translation_matrix, detector_translation_matrix = self.projection.get_translation_matrix()

        # source position
        source_position_with_translation = xp.matmul(
            xp.asarray(source_translation_matrix),
            xp.array([0, 0, 0, 1]).reshape([-1, 1])
        )
        rotation_matrices = self.projection.get_each_rotation_matrices((self.rebin_info[:, 0].reshape(self.projection.proj_dim[::-1]))[current_index, :, :].flatten())
        current_sp = xp.squeeze(xp.matmul(rotation_matrices, source_position_with_translation))[:, :3]

        # detector position
        detector_position = np.vstack((
            (((self.rebin_info[:, 2] + 0.5) - self.projection.proj_dim[0] / 2) * self.projection.proj_spacing[0]).reshape(self.projection.proj_dim[::-1])[current_index, :, :].flatten(),
            (((self.rebin_info[:, 1] + 0.5) - self.projection.proj_dim[1] / 2) * self.projection.proj_spacing[1]).reshape(self.projection.proj_dim[::-1])[current_index, :, :].flatten(),
            np.zeros(self.projection.proj_dim[0] * self.projection.proj_dim[1]),
            np.ones(self.projection.proj_dim[0] * self.projection.proj_dim[1])
        ))

        detector_position_with_translation = xp.matmul(xp.asarray(detector_translation_matrix), xp.asarray(detector_position))
        current_dp = xp.squeeze(xp.einsum('ijk,kli->ijl', rotation_matrices, xp.expand_dims(detector_position_with_translation, axis=1)))[:, :3]

        # info structure: [sp_x, sp_y, sp_z, dp_x, dp_y, dp_z]
        info = xp.hstack((
            current_sp, current_dp
        ))
        return info

    def nearest_interpolation(self, index):
        nan_tag = (np.isnan(index[:, 1])) | (np.isnan(index[:, 2]))
        index = np.nan_to_num(index, nan=0)

        x = np.round(index[:, 2], decimals=0).astype(int)
        y = np.round(index[:, 1], decimals=0).astype(int)
        z = np.round(index[:, 0], decimals=0).astype(int)

        x[x >= self.projection.proj_dim[0]] -= 1
        y[y >= self.projection.proj_dim[1]] -= 1
        z[z >= self.projection.proj_dim[2]] -= 1

        c = self.projection_img[z, y, x]
        c[nan_tag] = 0
        return c

    def trilinear_interpolation(self, index):
        index[:, 2][(index[:, 2] >= self.projection.proj_dim[0]) | (index[:, 2] < 0)] = np.nan
        index[:, 1][(index[:, 1] >= self.projection.proj_dim[1]) | (index[:, 1] < 0)] = np.nan
        nan_tag = (np.isnan(index[:, 1])) | (np.isnan(index[:, 2]))
        index = np.nan_to_num(index, nan=0)

        # 只适用于单位长度为 1 的情况
        xd = index[:, 2] % 1
        x0 = index[:, 2].astype(int)
        x1 = (index[:, 2] + 1).astype(int)
        yd = index[:, 1] % 1
        y0 = (index[:, 1]).astype(int)
        y1 = (index[:, 1] + 1).astype(int)
        zd = index[:, 0] % 1
        z0 = (index[:, 0]).astype(int)
        z1 = (index[:, 0] + 1).astype(int)

        z1[z1 >= self.projection_img.shape[0]] = self.projection_img.shape[0] - 1
        y1[y1 >= self.projection_img.shape[1]] = self.projection_img.shape[1] - 1
        x1[x1 >= self.projection_img.shape[2]] = self.projection_img.shape[2] - 1

        # x-direction
        c_00 = self.projection_img[z0, y0, x0] * (1 - xd) + self.projection_img[z0, y0, x1] * xd
        c_01 = self.projection_img[z1, y0, x0] * (1 - xd) + self.projection_img[z1, y0, x1] * xd
        c_10 = self.projection_img[z0, y1, x0] * (1 - xd) + self.projection_img[z0, y1, x1] * xd
        c_11 = self.projection_img[z1, y1, x0] * (1 - xd) + self.projection_img[z1, y1, x1] * xd

        # y-direction
        c_0 = c_00 * (1 - yd) + c_10 * yd
        c_1 = c_01 * (1 - yd) + c_11 * yd

        # z-direction
        c = c_0 * (1 - zd) + c_1 * zd
        c[nan_tag] = 0
        return c

    def projection_rebin(self):
        ratio = self.projection.source_to_iso_center / self.projection.source_to_detector

        # parallel beams
        parallel_theta = np.array([np.full((self.projection.proj_dim[1], self.projection.proj_dim[0]), i) for i in np.deg2rad(self.projection.gantry_angles)], dtype=np.float32)
        parallel_t = ((np.arange(self.projection.proj_dim[0]) + 0.5 - self.projection.proj_dim[0] / 2) * self.projection.proj_spacing[0]).astype(np.float32) * ratio
        parallel_s = ((np.arange(self.projection.proj_dim[1]) + 0.5 - self.projection.proj_dim[1] / 2) * self.projection.proj_spacing[1]).astype(np.float32) * ratio
        parallel_t, parallel_s = np.meshgrid(parallel_t, parallel_s)

        parallel_t = np.tile(parallel_t, (self.projection.proj_dim[2], 1, 1))
        parallel_s = np.tile(parallel_s, (self.projection.proj_dim[2], 1, 1))

        r = self.projection.source_to_iso_center
        rebin_beta = np.rad2deg(parallel_theta - np.arcsin(parallel_t / r))
        rebin_m = (parallel_t * r) / np.sqrt(r ** 2 - parallel_t ** 2) / ratio / self.projection.proj_spacing[0] + self.projection.proj_dim[0] / 2
        rebin_n = (parallel_s * r ** 2) / (r ** 2 - parallel_t ** 2) / ratio / self.projection.proj_spacing[1] + self.projection.proj_dim[1] / 2

        del parallel_theta
        del parallel_t
        del parallel_s

        # 清理异常值
        rebin_beta[rebin_beta < 0] += 360
        rebin_beta[rebin_beta >= 360] -= 360
        rebin_index = np.stack((rebin_beta.flatten(), rebin_n.flatten(), rebin_m.flatten()), axis=-1).astype(np.float32)

        self.rebin_info = rebin_index.copy()
        del rebin_beta
        del rebin_m
        del rebin_n

        # 并行三线性插值
        self.projection_img = np.array(self.trilinear_interpolation(rebin_index)).reshape(self.projection.proj_dim[::-1])

    def get_tfdk_weights(self):
        ratio = self.projection.source_to_iso_center / self.projection.source_to_detector
        x = (np.arange(self.projection.proj_dim[0]) * self.projection.proj_spacing[0] + self.projection.proj_origin[0]) * ratio
        y = (np.arange(self.projection.proj_dim[1]) * self.projection.proj_spacing[1] + self.projection.proj_origin[1]) * ratio
        grid_x, grid_y = np.meshgrid(x, y)

        weights = np.sqrt(
            (self.projection.source_to_iso_center ** 2 - grid_x ** 2) / (self.projection.source_to_iso_center ** 2 - grid_x ** 2 + grid_y ** 2)
        )
        weights = np.tile(weights, (self.projection.proj_dim[2], 1, 1))
        self.projection_img *= weights

    def get_each_fdk_recon(self, gantry_angle):
        start = time.time()
        current_index = int(gantry_angle / (360 / self.projection.num_of_projections))
        current_proj_info = self.get_each_info(gantry_angle)
        print("%ds" % (time.time() - start))
        current_proj_counts = xp.asarray(self.projection_img[current_index, :, :]).reshape(-1)
        weights = self.get_extend_z_weights(gantry_angle)  # fdk weights
        self.recon_img += parallelproj.backend.joseph3d_back(
            xstart=current_proj_info[:, :3],
            xend=current_proj_info[:, 3:6],
            img_shape=self.recon_dim,
            img_origin=xp.asarray(self.recon_origin),
            voxsize=xp.asarray(self.recon_spacing),
            img_fwd=current_proj_counts,
            threadsperblock=32,
            num_chunks=1
        ) * xp.asarray(weights)

    def get_all_fdk_recon(self):
        self.get_filtered_projection()
        for theta_angle in tqdm(self.projection.gantry_angles, desc="reconstructing..."):
            self.get_each_fdk_recon(theta_angle)


if __name__ == "__main__":
    projection = TFDKProjection(
        source_to_iso_center=221.927,
        source_to_detector=439.144,
        detector_offset_x=1.11777,
        detector_offset_y=0.588107,
        source_offset_x=0,
        source_offset_y=0,
        in_plane_angle=0.192887,
        out_of_plane_angle=0,
        proj_dim=np.array([1536, 384, 360]),
        proj_spacing=np.array([0.139] * 3)
    )

    fdk_recon = TFDKReconstruction(
        projection=projection,
        projection_img=np.fromfile(r"D:\linyuejie\temp_files\python_fdk\cutted_proj_384_360.raw",
                                   dtype=np.uint16).reshape([360, 384, 1536]),
        recon_dim=np.array([512, 512, 512]),
        recon_spacing=np.array([0.2, 0.2, 0.2])
    )

    angle = 90
    pyfdk = np.fromfile(r"D:\linyuejie\temp_files\python_fdk\parallel_parallelproj_recon.raw", dtype=np.float64).reshape(
        [512, 512, 512]).transpose([2, 1, 0])
    extend_weights = parallelproj.to_numpy_array(fdk_recon.get_extend_z_weights(angle))
    fdk_recon.get_each_fdk_recon(angle)
    img = parallelproj.backend.to_numpy_array(fdk_recon.recon_img)

    plt.figure(figsize=(8, 8))
    slice = 190
    plt.subplot(331)
    plt.imshow(extend_weights[slice, :, :])
    plt.xticks([], '')
    plt.yticks([], '')
    plt.subplot(332)
    plt.imshow(extend_weights[:, slice, :])
    plt.xticks([], '')
    plt.yticks([], '')
    plt.subplot(333)
    plt.imshow(extend_weights[:, :, slice])
    plt.xticks([], '')
    plt.yticks([], '')
    plt.subplot(334)
    plt.imshow(img[slice, :, :])
    plt.xticks([], '')
    plt.yticks([], '')
    plt.subplot(335)
    plt.imshow(img[:, slice, :])
    plt.xticks([], '')
    plt.yticks([], '')
    plt.subplot(336)
    plt.imshow(img[:, :, slice])
    plt.xticks([], '')
    plt.yticks([], '')
    plt.subplot(337)
    plt.imshow(pyfdk[slice, :, :])
    plt.xticks([], '')
    plt.yticks([], '')
    plt.subplot(338)
    plt.imshow(pyfdk[:, slice, :])
    plt.xticks([], '')
    plt.yticks([], '')
    plt.subplot(339)
    plt.imshow(pyfdk[:, :, slice])
    plt.xticks([], '')
    plt.yticks([], '')
    plt.show(block=True)

    # plt.subplot(131)
    # plt.imshow(img[256, :, :])
    # plt.subplot(132)
    # plt.imshow(img[:, 256, :])
    # plt.subplot(133)
    # plt.imshow(img[:, :, 256])
    # plt.show(block=True)
