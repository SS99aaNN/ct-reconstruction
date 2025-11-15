import array_api_compat
# import array_api_compat.numpy as xp
import array_api_compat.cupy as xp
# import array_api_compat.torch as xp

import matplotlib.pyplot as plt
import numpy as np
import parallelproj
from tqdm import tqdm
import cv2
import SimpleITK as sitk
from generals.CorrectionGenerals import CorrectionGenerals
from generals.ProjectionGenerals import ProjectionGenerals
from generals.ReconGenerals import ReconGenerals


class WaterPreCorrection(CorrectionGenerals):
    def __init__(self, projection: ProjectionGenerals, reconstruction: ReconGenerals, expect_radius, power_num):
        CorrectionGenerals.__init__(self, projection, reconstruction)
        self.expect_radius = int(expect_radius / self.reconstruction.recon_spacing[0])  # radius in mm
        self.power_num = int(power_num)
        self.upper_boundary = 500
        self.lower_boundary = 300
        self.projection_img = self.reconstruction.projection_img.reshape(-1)
        self.wpc_coefficients = self.get_wpc_coefficients()
        final_img = self.get_wpc_corrected_img()
        final_img.astype(np.float32).tofile(r"D:\linyuejie\temp_files\xjj_wpc\1216\n_power_80kV\py_wpc_test.raw")

    def get_wpc_corrected_img(self):
        powers = np.ones([(self.power_num + 1), np.prod(self.reconstruction.recon_dim)], dtype=np.float32)
        for i in range((self.power_num + 1)):
            # if i == 1:
            #     powers[i] = parallelproj.backend.to_numpy_array(self.reconstruction.recon_img).reshape(-1)
            # else:
            #     self.reconstruction.recon_img = xp.zeros(self.reconstruction.recon_dim)
            #     self.reconstruction.projection_img = (self.projection_img ** i).reshape(self.projection.proj_dim[::-1])
            #     self.reconstruction.get_all_fdk_recon()
            #     powers[i] = parallelproj.backend.to_numpy_array(self.reconstruction.recon_img).reshape(-1)
            powers[i] = np.fromfile(
                r"D:\linyuejie\temp_files\xjj_wpc\1216\n_power_80kV\recon_80kV_power_%d.raw" % i,
                dtype=np.float32)
        corrected_img = np.zeros(np.prod(self.reconstruction.recon_dim), dtype=np.float32)
        for i in range(self.power_num + 1):
            corrected_img += self.wpc_coefficients[i] * powers[i]
        return corrected_img

    def get_wpc_coefficients(self):
        weights_t, weights_omega, average_weights_t, average_weights_omega = self.get_weights_by_manual_mask()
        img_len = self.reconstruction.recon_dim[0] * self.reconstruction.recon_dim[2]
        powers = np.ones([(self.power_num + 1), img_len], dtype=np.float32)
        for i in range((self.power_num + 1)):
            # if i == 1:
            #     powers[i] = parallelproj.backend.to_numpy_array(self.reconstruction.recon_img).reshape(-1)
            # else:
            #     self.reconstruction.recon_img = np.zeros(self.reconstruction.recon_dim)
            #     self.reconstruction.projection_img = (self.projection_img ** i).reshape(self.projection.proj_dim[::-1])
            #     self.reconstruction.get_all_fdk_recon()
            #     powers[i] = parallelproj.backend.to_numpy_array(self.reconstruction.recon_img).reshape(-1)
            img = np.fromfile(
                r"D:\linyuejie\temp_files\xjj_wpc\1216\n_power_80kV\recon_80kV_power_%d.raw" % i,
                dtype=np.float32).reshape(self.reconstruction.recon_dim)[:, self.lower_boundary:self.upper_boundary, :]

            # average by slice
            average_img = np.mean(img, axis=1)
            img[weights_t.reshape(img.shape) <= 0] = np.nan
            img = np.nanmean(img, axis=1)
            average_img[average_weights_t.reshape([img.shape[0], img.shape[1]]) > 0] = img[~np.isnan(img)]

            # filter with padding 0
            # img_with_padding = average_img.copy()
            # img_with_padding[average_weights_t.reshape([img.shape[0], img.shape[1]]) == 0] = np.unique(weights_t[weights_t > 0])
            # img_with_padding = cv2.GaussianBlur(img_with_padding, (3, 3), 1)
            # average_img[average_weights_t.reshape([img.shape[0], img.shape[1]]) > 0] = img_with_padding[average_weights_t.reshape([img.shape[0], img.shape[1]]) > 0]

            # # filter with padding
            # img_with_padding = img.copy()
            # img_with_padding[weights_t == 0] = 0#np.unique(weights_t[weights_t > 0])
            # img_with_padding = img_with_padding.reshape([self.reconstruction.recon_dim[0], (self.upper_boundary - self.lower_boundary), self.reconstruction.recon_dim[2]])
            # for j in range(int(self.upper_boundary - self.lower_boundary)):
            #     img_with_padding[:, j, :] = cv2.GaussianBlur(img_with_padding[:, j, :], (5, 5), 25)
            # img_with_padding = img_with_padding.reshape(-1)
            # img[weights_t > 0] = img_with_padding[weights_t > 0]

            powers[i] = average_img.reshape(-1)

        # mat B(ixj) * vector c(jx1) = mat a(ix1)
        # fit by sum (multi-slice)
        mat_b = np.zeros([(self.power_num + 1), (self.power_num + 1)], dtype=np.float32)
        for i in range(self.power_num + 1):
            for j in range(i, self.power_num + 1):
                mat_b[i, j] = np.sum(average_weights_omega * powers[i] * powers[j])
                if i != j:
                    mat_b[j, i] = np.sum(average_weights_omega * powers[i] * powers[j])

        mat_a = np.zeros([self.power_num + 1], dtype=np.float32)
        for i in range(self.power_num + 1):
            mat_a[i] = np.sum(powers[i] * average_weights_t * average_weights_omega)

        wpc_coefficients = np.linalg.solve(mat_b, mat_a)
        print(wpc_coefficients)
        return wpc_coefficients

    def get_weights(self):
        #power_1 = parallelproj.backend.to_numpy_array(self.reconstruction.recon_img)
        power_1 = np.fromfile(r"D:\linyuejie\temp_files\xjj_wpc\n_degree_linear_integral_projection\recon_power_1.raw", dtype=np.float32).reshape([820, 820, 820])
        # 确定拟合范围
        water_mask = np.zeros([(self.upper_boundary-self.lower_boundary), 3], dtype=int)
        for i in range(self.upper_boundary-self.lower_boundary):
            water_mask[i] = self.get_cylinder_in_recon_img(power_1[:, i, :], self.expect_radius)

        center_fov_radius = int(
            np.sin(
                np.arctan(
                    np.abs(self.projection.proj_origin[0]) / self.projection.source_to_detector
                )) * self.projection.source_to_iso_center / self.reconstruction.recon_spacing[0])
        fov_mask = self.get_cylinder_in_recon_img(power_1[:, int(self.reconstruction.recon_dim[0] / 2), :], center_fov_radius)

        # weights_t --> water = μ_water, other = 0 (μ_water 从 inside_water 的边上取)
        weights_t = np.zeros([(self.upper_boundary-self.lower_boundary), self.reconstruction.recon_dim[1], self.reconstruction.recon_dim[2]], dtype=np.float32)
        boundary_for_mu_water = np.zeros([(self.upper_boundary-self.lower_boundary), self.reconstruction.recon_dim[1], self.reconstruction.recon_dim[2]], dtype=np.uint8)
        for i in range(self.upper_boundary-self.lower_boundary):
            weights_t[i], boundary_for_mu_water[i] = self.get_water_region(water_mask[i])

        mu_water = np.mean(power_1.transpose([1, 0, 2])[self.lower_boundary:self.upper_boundary, :, :][boundary_for_mu_water > 0])
        weights_t[weights_t > 0] = mu_water
        rm_slice_record = []
        if np.any((power_1.transpose([1, 0, 2])[self.lower_boundary:self.upper_boundary, :, :][weights_t > 0] < (mu_water*0.1))):
            print("May have segmentation fault or bubble inside the water region. ")
            print("Remove these slices. ")
            cutted_img = power_1.transpose([1, 0, 2])[self.lower_boundary:self.upper_boundary, :, :]
            cutted_img[weights_t == 0] = 0
            for i in range(self.upper_boundary - self.lower_boundary):
                error_point = cutted_img[i][(0 < cutted_img[i]) & (cutted_img[i] < mu_water * 0.1)]
                if len(error_point) > 50:
                    weights_t[i] = np.zeros_like(weights_t[0])
                    rm_slice_record.append(i)
            print("%d slices have been removed, %d slices left for calculation. " % (len(rm_slice_record), self.upper_boundary - self.lower_boundary - len(rm_slice_record)))

        # weights_omega --> water/air = 1, other=0
        weights_omega = self.get_fov_region(fov_mask)
        weights_omega = np.tile(weights_omega, ((self.upper_boundary-self.lower_boundary), 1, 1))
        boundary_for_shell = np.zeros([(self.upper_boundary-self.lower_boundary), self.reconstruction.recon_dim[1], self.reconstruction.recon_dim[2]])
        for i in range(self.upper_boundary - self.lower_boundary):
            _ = cv2.circle(boundary_for_shell[i], (water_mask[i][0], water_mask[i][1]), water_mask[i][2], (255, 255, 255), 20)
        weights_omega[boundary_for_shell > 0] = 0
        weights_omega[weights_omega > 0] = 1
        if len(rm_slice_record) != 0:
            for i in rm_slice_record:
                weights_omega[i] = np.zeros_like(weights_omega[0])

        # reshape([1, 0, 2]) 将横截面与重建图像对齐
        return weights_t.transpose([1, 0, 2]).reshape(-1), weights_omega.transpose([1, 0, 2]).reshape(-1)

    def get_weights_by_manual_mask(self):
        recon_img = np.fromfile(r"D:\linyuejie\temp_files\xjj_wpc\1216\n_power_80kV\recon_80kV_power_1.raw", dtype=np.float32).reshape([820, 820, 820])
        # for i in range(int(300), int(500)):
        #     recon_img[:, i, :] = cv2.GaussianBlur(recon_img[:, i, :], (5, 5), 5)
        water_inside_mask = sitk.GetArrayFromImage(sitk.ReadImage(r"D:\linyuejie\temp_files\xjj_wpc\1216\n_power_80kV\water_inside.nii"))
        water_outside_mask = sitk.GetArrayFromImage(sitk.ReadImage(r"D:\linyuejie\temp_files\xjj_wpc\1216\n_power_80kV\water_outside.nii"))
        fov_inside_mask = sitk.GetArrayFromImage(sitk.ReadImage(r"D:\linyuejie\temp_files\xjj_wpc\1216\n_power_80kV\fov_inside.nii"))

        weights_t = np.zeros_like(water_inside_mask, dtype=np.float32)
        recon_img[water_inside_mask == 0] = np.nan
        recon_img = np.nanmean(recon_img, axis=1)

        mu_water = np.nanpercentile(recon_img, 95)
        weights_t[water_inside_mask > 0] = mu_water

        weights_omega = np.zeros_like(water_inside_mask, dtype=np.uint8)
        weights_omega[fov_inside_mask > 0] = 1
        weights_omega[(water_inside_mask == 0) & (water_outside_mask > 0)] = 0

        average_weights_t = np.mean(weights_t, axis=1)
        average_weights_t[average_weights_t > 0] = mu_water

        average_weights_omega = np.mean(weights_omega, axis=1)
        average_weights_omega[average_weights_omega > 0] = 1
        return weights_t[:, self.lower_boundary:self.upper_boundary, :].reshape(-1), weights_omega[:, self.lower_boundary:self.upper_boundary, :].reshape(-1), average_weights_t.reshape(-1), average_weights_omega.reshape(-1)

    def get_fov_region(self, fov_mask):
        fov_boundary = np.zeros([self.reconstruction.recon_dim[1], self.reconstruction.recon_dim[2]])
        _ = cv2.circle(fov_boundary, (fov_mask[0], fov_mask[1]), fov_mask[2], (255, 255, 255), 60)

        fov_inside = np.zeros([self.reconstruction.recon_dim[1], self.reconstruction.recon_dim[2]])
        _ = cv2.circle(fov_inside, (fov_mask[0], fov_mask[1]), fov_mask[2], (255, 255, 255), -1)

        fov_inside[fov_boundary > 0] = 0
        return fov_inside

    def get_water_region(self, water_mask_2d):
        inside_water = np.zeros([self.reconstruction.recon_dim[1], self.reconstruction.recon_dim[2]], dtype=np.uint8)
        cv2.circle(inside_water, (water_mask_2d[0], water_mask_2d[1]), water_mask_2d[2], (255, 255, 255), -1)
        thick_boundary = np.zeros([self.reconstruction.recon_dim[1], self.reconstruction.recon_dim[2]], dtype=np.uint8)
        cv2.circle(thick_boundary, (water_mask_2d[0], water_mask_2d[1]), water_mask_2d[2], (255, 255, 255), 15)
        inside_water[thick_boundary > 0] = 0

        water_boundary = np.zeros([self.reconstruction.recon_dim[1], self.reconstruction.recon_dim[2]], dtype=np.uint8)
        cv2.circle(water_boundary, (water_mask_2d[0], water_mask_2d[1]), water_mask_2d[2], (255, 255, 255), 25)
        water_boundary[inside_water == 0] = 0
        return inside_water, water_boundary

    def get_cylinder_in_recon_img(self, recon_img, expect_radius):
        recon_img = (recon_img / np.max(recon_img) * 255).astype(np.uint8)
        blurred_image = cv2.GaussianBlur(recon_img, (15, 15), 0)

        circles = cv2.HoughCircles(
            blurred_image,
            cv2.HOUGH_GRADIENT,
            dp=1.2,  # 累加器分辨率与原图分辨率的比值
            minDist=500,  # 检测到的圆心之间的最小距离
            param1=30,  # Canny 边缘检测的高阈值
            param2=5,  # 累加器阈值，用于圆心检测
            minRadius=expect_radius - 5,  # 最小圆半径
            maxRadius=expect_radius + 5  # 最大圆半径
        )

        if circles.shape[0] == 1:
            circles = np.round(circles[0, 0, :]).astype("int")  # 圆的位置和半径
            cv2.circle(blurred_image, (circles[0], circles[1]), circles[2], (255, 255, 255), -1)
            return circles
        else:
            circles = np.round(circles[0, :]).astype("int")  # 圆的位置和半径
            for (x, y, r) in circles:
                cv2.circle(blurred_image, (x, y), r, (255, 255, 255), 4)  # 绘制圆边
            plt.imshow(blurred_image)
            plt.show(block=True)

    def get_forward_projection_img(self, recon_img):
        fwd_img = np.zeros(self.projection.proj_dim[::-1])
        for i in tqdm(range(fwd_img.shape[0]), desc="Calculating forward projection..."):
            current_proj_info = self.projection.get_each_info(i * (360 / fwd_img.shape[0]))
            fwd_img[i] = self.reconstruction.forward_projection(recon_img, current_proj_info)
        return fwd_img


if __name__ == "__main__":
    from recon_methods.FDK import FDKProjection, FDKReconstruction
    # fdk_projection = FDKProjection(
    #     source_to_iso_center=221.927,
    #     source_to_detector=439.144,
    #     detector_offset_x=1.11777,
    #     detector_offset_y=0.588107,
    #     source_offset_x=0,
    #     source_offset_y=0,
    #     in_plane_angle=0.192887,
    #     out_of_plane_angle=0,
    #     proj_dim=np.array([1536, 1536, 360]),
    #     proj_spacing=np.array([0.139] * 3)
    # )

    fdk_projection = FDKProjection(
            source_to_iso_center=175.659,
            source_to_detector=361.321,
            detector_offset_x=-1.36,
            detector_offset_y=0.05,
            source_offset_x=0,
            source_offset_y=0,
            in_plane_angle=0.154,
            out_of_plane_angle=0,
            proj_dim=np.array([896, 896, 720]),
            proj_spacing=np.array([0.2, 0.2, 0.1])
        )

    fdk_recon = FDKReconstruction(
        projection=fdk_projection,
        projection_img=np.fromfile(r"D:\linyuejie\temp_files\xjj_wpc\1216\water2_40kV_360s_pga5.raw", dtype=np.uint16).reshape(fdk_projection.proj_dim[::-1]),
        recon_dim=np.array([820, 820, 820]),
        recon_spacing=np.array([0.1, 0.1, 0.1])
    )

    fdk_wpc = WaterPreCorrection(
        fdk_projection,
        fdk_recon,
        28.7,
        4
    )


