import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
from generals.ProjectionGenerals import ProjectionGenerals
import parallelproj
# import array_api_compat.numpy as xp
import array_api_compat.cupy as xp
# import array_api_compat.torch as xp


class ReconGenerals:
    def __init__(self, projection: ProjectionGenerals, projection_img, recon_dim, recon_spacing):
        self.recon_dim = recon_dim
        self.recon_spacing = recon_spacing
        self.recon_origin = - (recon_dim-1) / 2 * recon_spacing
        self.projection = projection
        self.projection_img = projection_img.astype(np.float32)
        self.get_linear_projections()
        self.filter = self.get_ramp_filter()
        self.recon_img = np.zeros(self.recon_dim)

    def get_linear_projections(self):
        # 这里的I0采用了临时值
        self.projection_img = (-np.log(np.clip(self.projection_img / 53003, 0, None))).astype(np.float32)

    def get_water_pre_correction(self):
        coefs = self.projection.water_pre_correction_coefs
        if coefs is not None:
            corrected_img = np.zeros_like(self.projection_img, dtype=np.float32)
            for i in range(coefs.shape[0]):
                corrected_img += self.projection_img**i * coefs[i]
            self.projection_img = corrected_img

    def get_ramp_filter(self):
        N = self.projection.proj_dim[0]
        d = 1  # sample frequency

        filterRL = np.zeros((N,), dtype=np.float32)
        for i in range(N):
            filterRL[i] = - 1.0 / np.power((i - N / 2) * np.pi * d, 2.0)
            if np.mod(i - N / 2, 2) == 0:
                filterRL[i] = 0
        filterRL[int(N / 2)] = 1 / (4 * np.power(d, 2.0))
        return filterRL

    def get_shepp_logan_filter(self):
        N = self.projection.proj_dim[0]
        d = 1  # sample frequency

        shepp_logan_filter = np.zeros((N,), dtype=np.float32)
        for i in range(N):
            # filterSL[i] = - 2 / (np.power(np.pi, 2.0) * np.power(d, 2.0) * (np.power((4 * (i - N / 2)), 2.0) - 1))
            shepp_logan_filter[i] = - 2 / (np.pi ** 2.0 * d ** 2.0 * (4 * (i - N / 2) ** 2.0 - 1))
        return shepp_logan_filter

    def get_filtered_projection(self):
        use_filter = self.filter
        for i in tqdm(range(self.projection.proj_dim[1]), desc="filtering..."):
            current_img = self.projection_img[:, i, :].copy()
            for j in range(self.projection.proj_dim[2]):
                current_img[j, :] = np.convolve(current_img[j, :], use_filter, "same")
            self.projection_img[:, i, :] = current_img.astype(np.float32)

    def back_projection(self, projection_info, projection_counts):
        bp_img = parallelproj.backend.joseph3d_back(
            xstart=xp.asarray(projection_info[:, :3]),
            xend=xp.asarray(projection_info[:, 3:6]),
            img_shape=self.recon_dim,
            img_origin=xp.asarray(self.recon_origin),
            voxsize=xp.asarray(self.recon_spacing),
            img_fwd=xp.asarray(projection_counts),
            threadsperblock=32,
            num_chunks=1
        )
        return bp_img

    def forward_projection(self, img, projection_info):
        fwd_img = parallelproj.backend.joseph3d_fwd(
            xstart=xp.asarray(projection_info[:, :3]),
            xend=xp.asarray(projection_info[:, 3:6]),
            img=xp.asarray(img),
            img_origin=xp.asarray(self.recon_origin),
            voxsize=xp.asarray(self.recon_spacing),
            threadsperblock=32,
            num_chunks=1
        ).reshape([1536, 1536])
        return parallelproj.backend.to_numpy_array(fwd_img)

    def get_extended_z_weights(self):
        pass

    def get_fdk_reconstruction_weights(self, gantry_angle):
        pass

    def get_fdk_projection_weights(self):
        pass

    def get_each_detector_counts(self, gantry_angle):
        pass

    def get_each_fdk_recon(self, gantry_angle):
        pass

    def get_all_fdk_recon(self):
        pass