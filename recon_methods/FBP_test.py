# import array_api_compat.numpy as xp
import array_api_compat.cupy as xp
# import array_api_compat.torch as xp

import numpy as np
import matplotlib.pyplot as plt
import parallelproj
import SimpleITK as sitk
from joblib import Parallel, delayed

from generals.ProjectionGenerals import ProjectionGenerals


def RLFilter(N, d):
    filterRL = np.zeros((N,))
    for i in range(N):
        filterRL[i] = - 1.0 / np.power((i - N / 2) * np.pi * d, 2.0)
        if np.mod(i - N / 2, 2) == 0:
            filterRL[i] = 0
    filterRL[int(N / 2)] = 1 / (4 * np.power(d, 2.0))
    return filterRL


def SLFilter(N, d):
    filterSL = np.zeros((N,))
    for i in range(N):
        # filterSL[i] = - 2 / (np.power(np.pi, 2.0) * np.power(d, 2.0) * (np.power((4 * (i - N / 2)), 2.0) - 1))
        filterSL[i] = - 2 / (np.pi ** 2.0 * d ** 2.0 * (4 * (i - N / 2) ** 2.0 - 1))
    return filterSL


def get_each_weighted_bp(projection, p_data):
    x = np.arange(512) * 0.2 - 51.2
    recon_grid_x, recon_grid_y = np.meshgrid(x, x)
    weights = projection.source_to_detector ** 2 / (
            projection.source_to_iso_center + recon_grid_x * np.sin(
        np.deg2rad(i * (360 / 720))) + recon_grid_y * np.cos(np.deg2rad(i * (360 / 720)))
    ) ** 2
    img = parallelproj.backend.joseph3d_back(
        xstart=projection.proj_info[i * 1536:((i + 1) * 1536), :3],
        xend=projection.proj_info[i * 1536:((i + 1) * 1536), 3:6],
        img_shape=(512, 512, 512),
        img_origin=np.array([-51.2, -51.2, -51.2]),
        voxsize=np.array([0.2, 0.2, 0.2]),
        img_fwd=p_data,
        threadsperblock=32,
        num_chunks=1
    ) * weights * (np.pi * 2 / 720)
    return img


def get_all_weighted_bq(projection, p_data):
    recons = Parallel(n_jobs=10)(
        delayed(get_each_weighted_bp)(projection, p_data[i, :]) for i in range(720))
    recon_img = recons[0]
    for i in range(1, len(recons)):
        recon_img += recons[i]

    return recon_img


if __name__ == "__main__":
    projection = ProjectionGenerals(
        source_to_iso_center=211.927,
        source_to_detector=439.144,
        detector_offset_x=1.11777,
        detector_offset_y=0.588107,
        source_offset_x=0,
        source_offset_y=0,
        in_plane_angle=0.192887,
        out_of_plane_angle=0,
        proj_dim=np.array([1536, 1, 720]),
        proj_spacing=np.array([0.139]*3)
    )

    fan_beam = sitk.GetArrayFromImage(sitk.ReadImage(r"D:\linyuejie\temp_files\python_fdk\7cm_water_phantom_60kV_700uA_pga3_10fps.mhd"))[:, 768, :]
    fan_beam = -np.log(np.clip(fan_beam / 53003, 0, None))
    filtered_beam = np.zeros_like(fan_beam)
    use_filter = SLFilter(fan_beam.shape[1], 1)

    ratio_1 = projection.source_to_iso_center / np.sqrt(
        projection.source_to_detector**2 + (np.arange(projection.proj_dim[0]) * projection.proj_spacing[0] + projection.proj_origin[0])**2
    )

    for i in range(fan_beam.shape[0]):
        current_data = fan_beam[i, :]
        filtered_beam[i, :] = np.convolve(current_data, use_filter, "same")

    # plt.subplot(211)
    # plt.imshow(fan_beam)
    # plt.subplot(212)
    # plt.imshow(filtered_beam)
    # plt.show(block=True)

    not_filtered = xp.zeros([512, 512, 512])
    filtered = xp.zeros([512, 512, 512])

    weights = xp.zeros([720, 512, 512])
    for i in range(720):
        x = np.arange(512) * 0.2 - 51.2
        recon_grid_x, recon_grid_y = np.meshgrid(x, x)
        # weights[i, :, :] = xp.asarray(projection.source_to_detector ** 2 / (
        #         projection.source_to_iso_center + recon_grid_x * np.sin(
        #     np.deg2rad(i * (360 / 720))) + recon_grid_y * np.cos(np.deg2rad(i * (360 / 720)))
        # ) ** 2 * (np.pi * 2 / 720))
        # weights[i, :, :] = (np.pi * 2 / 720)

        not_filtered += parallelproj.backend.joseph3d_back(
            xstart=xp.asarray(projection.proj_info[i, :, :3]),
            xend=xp.asarray(projection.proj_info[i, :, 3:6]),
            img_shape=(512, 512, 512),
            img_origin=xp.array([-51.2, -51.2, -51.2]),
            voxsize=xp.array([0.2, 0.2, 0.2]),
            img_fwd=xp.asarray(fan_beam[i, :]),
            threadsperblock=32,
            num_chunks=1
        )  # * weights[i, :, :]

        filtered += parallelproj.backend.joseph3d_back(
            xstart=xp.array(projection.proj_info[i, :, :3]),
            xend=xp.array(projection.proj_info[i, :, 3:6]),
            img_shape=(512, 512, 512),
            img_origin=xp.array([-51.2, -51.2, -51.2]),
            voxsize=xp.array([0.2, 0.2, 0.2]),
            img_fwd=xp.array(filtered_beam[i, :]),
            threadsperblock=32,
            num_chunks=1
        )  # * weights[i, :, :]

    not_filtered = parallelproj.backend.to_numpy_array(not_filtered)
    filtered = parallelproj.backend.to_numpy_array(filtered)

    plt.subplot(121)
    plt.imshow(not_filtered[:, 257, :])
    plt.subplot(122)
    plt.imshow(filtered[:, 257, :])
    plt.show(block=True)

