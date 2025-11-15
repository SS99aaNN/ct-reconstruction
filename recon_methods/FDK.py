import array_api_compat
# import array_api_compat.numpy as xp
import array_api_compat.cupy as xp
# import array_api_compat.torch as xp

import matplotlib.pyplot as plt
import numpy as np
import parallelproj
from tqdm import tqdm

from generals.ProjectionGenerals import ProjectionGenerals
from generals.ReconGenerals import ReconGenerals
from correction_methods.ExtendedZWeight import ExtendedZWeight


class FDKProjection(ProjectionGenerals):
    def __init__(self, source_to_iso_center, source_to_detector, detector_offset_x, detector_offset_y, source_offset_x,
                 source_offset_y, in_plane_angle, out_of_plane_angle, proj_dim, proj_spacing):
        # scanner settings
        ProjectionGenerals.__init__(self, source_to_iso_center, source_to_detector, detector_offset_x, detector_offset_y, source_offset_x,
                 source_offset_y, in_plane_angle, out_of_plane_angle, proj_dim, proj_spacing)


class FDKReconstruction(ReconGenerals):
    def __init__(self, projection: FDKProjection, projection_img, recon_dim, recon_spacing):
        ReconGenerals.__init__(self, projection, projection_img, recon_dim, recon_spacing)
        self.recon_img = xp.zeros(self.recon_dim)
        self.get_fdk_projection_weights()
        self.xfdk = None

    def get_extended_z_weights(self):
        self.xfdk = ExtendedZWeight(projection, self)

    def get_fdk_reconstruction_weights(self, gantry_angle):
        x = xp.arange(self.recon_dim[0]) * self.recon_spacing[0] + self.recon_origin[0]
        y = -(xp.arange(self.recon_dim[1]) * self.recon_spacing[1] + self.recon_origin[1])

        grid_x, grid_y = xp.meshgrid(x, y)
        s = grid_y * xp.cos(xp.deg2rad(gantry_angle)) - grid_x * xp.sin(xp.deg2rad(gantry_angle))
        u = self.projection.source_to_detector / (self.projection.source_to_iso_center + s)
        weights = u ** 2 * (2 * xp.pi / self.projection.num_of_projections)
        weights = xp.tile(weights, (self.recon_dim[2], 1, 1))
        return weights.transpose([2, 0, 1])

    def get_fdk_projection_weights(self):
        x = np.arange(self.projection.proj_dim[0]) * self.projection.proj_spacing[0] + self.projection.proj_origin[0]
        y = np.arange(self.projection.proj_dim[1]) * self.projection.proj_spacing[1] + self.projection.proj_origin[1]
        grid_x, grid_y = np.meshgrid(x, y)
        weights = self.projection.source_to_iso_center / np.sqrt(self.projection.source_to_iso_center**2 + grid_x**2 + grid_y**2)
        weights = np.tile(weights, (self.projection.proj_dim[2], 1, 1))
        self.projection_img *= weights

    def get_each_detector_counts(self, gantry_angle):
        current_index = int(gantry_angle / (360 / self.projection.num_of_projections))
        return self.projection_img[current_index, :, :].reshape(-1, 1)

    def get_each_fdk_recon(self, gantry_angle):
        current_index = int(gantry_angle / (360 / self.projection.num_of_projections))
        current_proj_info = self.projection.get_each_info(gantry_angle)
        current_proj_counts = self.projection_img[current_index, :, :].reshape(-1)
        weights = self.get_fdk_reconstruction_weights(gantry_angle)  # fdk weights
        if self.xfdk is not None:
            weights *= self.xfdk.get_extend_z_weights(gantry_angle)  # extend_z weights

        self.recon_img += parallelproj.backend.joseph3d_back(
            xstart=xp.asarray(current_proj_info[:, :3]),
            xend=xp.asarray(current_proj_info[:, 3:6]),
            img_shape=self.recon_dim,
            img_origin=xp.asarray(self.recon_origin),
            voxsize=xp.asarray(self.recon_spacing),
            img_fwd=xp.asarray(current_proj_counts),
            threadsperblock=32,
            num_chunks=1
        ) * xp.asarray(weights)

    def get_all_fdk_recon(self):
        self.get_filtered_projection()
        for angle in tqdm(self.projection.gantry_angles, desc="reconstructing..."):
            self.get_each_fdk_recon(angle)


if __name__ == "__main__":
    projection = FDKProjection(
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

    fdk_recon = FDKReconstruction(
        projection=projection,
        projection_img=np.fromfile(r"D:\linyuejie\temp_files\python_fdk\cutted_proj_384_360.raw",
                                   dtype=np.uint16).reshape(projection.proj_dim[::-1]),
        recon_dim=np.array([512, 512, 512]),
        recon_spacing=np.array([0.2, 0.2, 0.2])
    )

    # test direction
    angle = 135
    pyfdk = np.fromfile(r"D:\linyuejie\temp_files\python_fdk\recon_384.raw", dtype=np.float32).reshape(
        [512, 512, 512]).transpose([2, 1, 0])
    fdk_recon.get_each_fdk_recon(angle)
    recon_img = array_api_compat.to_device(fdk_recon.recon_img, 'cpu')

    # fdk_weights = array_api_compat.to_device(fdk_recon.get_fdk_weights_in_reconstruction_fields(angle), 'cpu')
    extend_weights_trans = extend_weights#.transpose([2, 0, 1])
    # fdk_weights_trans = fdk_weights.transpose([0, 1, 2])

    plt.figure(figsize=(8, 8))
    plt.subplot(331)
    plt.imshow(extend_weights_trans[190, :, :])
    plt.xticks([], '')
    plt.yticks([], '')
    plt.subplot(332)
    plt.imshow(extend_weights_trans[:, 190, :])
    plt.xticks([], '')
    plt.yticks([], '')
    plt.subplot(333)
    plt.imshow(extend_weights_trans[:, :, 190])
    plt.xticks([], '')
    plt.yticks([], '')
    plt.subplot(334)
    plt.imshow(recon_img[190, :, :])
    plt.xticks([], '')
    plt.yticks([], '')
    plt.subplot(335)
    plt.imshow(recon_img[:, 190, :])
    plt.xticks([], '')
    plt.yticks([], '')
    plt.subplot(336)
    plt.imshow(recon_img[:, :, 190])
    plt.xticks([], '')
    plt.yticks([], '')
    plt.subplot(337)
    plt.imshow(pyfdk[190, :, :])
    plt.xticks([], '')
    plt.yticks([], '')
    plt.subplot(338)
    plt.imshow(pyfdk[:, 190, :])
    plt.xticks([], '')
    plt.yticks([], '')
    plt.subplot(339)
    plt.imshow(pyfdk[:, :, 190])
    plt.xticks([], '')
    plt.yticks([], '')
    plt.show(block=True)

    # fdk_recon.get_all_fdk_recon()
    # fdk_recon.recon_img.tofile(r"D:\linyuejie\temp_files\python_fdk\python_fdk.raw")
    # img = array_api_compat.to_device(fdk_recon.recon_img, "cpu")
    # plt.imshow(img[:, 257, :])
    # plt.show(block=True)
