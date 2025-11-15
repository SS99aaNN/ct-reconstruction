import matplotlib.pyplot as plt
import numpy as np
import array_api_compat.cupy as xp
import parallelproj
from tqdm import tqdm

from generals.CorrectionGenerals import CorrectionGenerals
from generals.ProjectionGenerals import ProjectionGenerals
from generals.ReconGenerals import ReconGenerals


class ExtendedZWeight(CorrectionGenerals):
    def __init__(self, projection: ProjectionGenerals, reconstruction: ReconGenerals):
        CorrectionGenerals.__init__(self, projection, reconstruction)

        # cpu coefs in extended z (xFDK)
        self.cz = np.zeros(self.reconstruction.recon_dim[2])
        self.r = np.zeros(self.reconstruction.recon_dim[:2])
        self.weight_t = np.zeros(self.reconstruction.recon_dim)
        self.phi = np.zeros(self.reconstruction.recon_dim)
        # gpu coefs in extend z (xFDK)
        self.theta_1 = xp.zeros(self.reconstruction.recon_dim)
        self.theta_2 = xp.zeros(self.reconstruction.recon_dim)
        self.delta_theta = xp.zeros(self.reconstruction.recon_dim)

        self.init_extend_z()

    def init_extend_z(self):
        self.get_coefs_in_extend_z()
        self.get_weight_transition_in_extend_z()

    def get_weight_transition_in_extend_z(self):
        r = np.tile(self.r, (self.reconstruction.recon_dim[2], 1, 1))
        cz = np.array([np.full([self.reconstruction.recon_dim[0], self.reconstruction.recon_dim[1]], i) for i in self.cz])

        rf = self.projection.source_to_iso_center
        beta_max, gamma_max, _ = np.arctan(((self.projection.proj_dim - 1) / 2 * self.projection.proj_spacing) / self.projection.source_to_detector)
        rm = self.projection.source_to_iso_center * np.sin(beta_max)
        delta_r = rm ** 2 / (2 * rf)
        r0 = rf - cz - delta_r

        weight_t = np.zeros(self.reconstruction.recon_dim)
        option_1 = (r >= r0 - delta_r) & (r < r0 + delta_r)
        weight_t[option_1] = 1 + np.sin(np.pi / 2 * ((r - r0) / delta_r))[option_1]
        option_2 = (r >= r0 + delta_r)
        weight_t[option_2] = 2
        weight_t /= 2
        self.weight_t = weight_t

    def get_coefs_in_extend_z(self):
        x = ((np.arange(self.reconstruction.recon_dim[0]) + 0.5) - self.reconstruction.recon_dim[0] / 2) * self.reconstruction.recon_spacing[0]
        y = -((np.arange(self.reconstruction.recon_dim[1]) + 0.5) - self.reconstruction.recon_dim[1] / 2) * self.reconstruction.recon_spacing[1]
        z = np.abs(((np.arange(self.reconstruction.recon_dim[2]) + 0.5) - self.reconstruction.recon_dim[2] / 2) * self.reconstruction.recon_spacing[2])

        grid_x, grid_y = np.meshgrid(x, y)
        self.r = np.sqrt(grid_x ** 2 + grid_y ** 2)
        self.phi = xp.asarray(np.arctan2(-grid_x, grid_y))

        beta_max, gamma_max, _ = np.arctan(((
                                                        self.projection.proj_dim - 1) / 2 * self.projection.proj_spacing) / self.projection.source_to_detector)
        c = 1 / np.tan(gamma_max)
        self.cz = c * z

        xp_cz = xp.array([np.full([self.reconstruction.recon_dim[0], self.reconstruction.recon_dim[1]], i) for i in self.cz])
        rf = self.projection.source_to_iso_center
        xp_r = xp.tile(xp.asarray(self.r), (self.reconstruction.recon_dim[2], 1, 1))

        self.theta_1 = xp.tile(self.phi - xp.pi / 2, (self.reconstruction.recon_dim[0], 1, 1))
        self.theta_2 = xp.tile(self.phi + xp.pi / 2, (self.reconstruction.recon_dim[0], 1, 1))
        self.delta_theta = xp.arcsin((rf - xp_cz) / xp_r) - xp.arctan(
            xp.sqrt(xp_r ** 2 - (rf - xp_cz) ** 2) / xp_cz)
        self.delta_theta[(xp_r <= xp.abs(rf - xp_cz))] = (xp.pi / 2 * xp.sign(rf - xp_cz))[
            (xp_r <= xp.abs(rf - xp_cz))]

    def get_beta(self, gantry_angle):
        alpha = np.deg2rad(gantry_angle)
        alpha_minus_phi = xp.tile(alpha - self.phi, (self.reconstruction.recon_dim[2], 1, 1))

        xp_cz = xp.array([np.full([self.reconstruction.recon_dim[0], self.reconstruction.recon_dim[1]], i) for i in self.cz])
        xp_r = xp.tile(xp.asarray(self.r), (self.reconstruction.recon_dim[2], 1, 1))

        option_range = (xp.pi / 2 + xp.arcsin((self.projection.source_to_iso_center - xp_cz) / xp_r))
        option_1 = xp.abs(alpha_minus_phi) > option_range
        option_2 = (xp.abs(alpha_minus_phi - xp.pi * 2) > option_range) & ~(xp.abs(alpha_minus_phi) <= option_range)
        judgement = option_1 & option_2
        alpha_minus_phi[judgement] = np.nan

        beta = -xp.arctan(
            (xp_r * xp.sin(alpha_minus_phi)) / (self.projection.source_to_iso_center + xp_r*xp.cos(alpha_minus_phi))
        )
        return beta

    def get_extend_z_weights(self, gantry_angle):  # angle = gantry_angle
        alpha = np.deg2rad(gantry_angle)
        beta = self.get_beta(gantry_angle)
        theta = alpha + beta
        theta_minus_2pi = theta-xp.pi*2

        weight_ps = xp.zeros(self.reconstruction.recon_dim)
        theta_1_minus_delta_theta = self.theta_1 - self.delta_theta
        theta_1_add_delta_theta = self.theta_1 + self.delta_theta
        theta_2_minus_delta_theta = self.theta_2 - self.delta_theta
        theta_2_add_delta_theta = self.theta_2 + self.delta_theta

        option_1 = (theta >= theta_1_minus_delta_theta)
        option_2 = (theta < theta_1_add_delta_theta) | ((theta_minus_2pi < theta_1_add_delta_theta) & (theta_minus_2pi >= theta_1_minus_delta_theta))
        judgement = option_1 & option_2
        modified_theta = theta
        option = ((theta_minus_2pi < theta_1_add_delta_theta) & (theta_minus_2pi >= theta_1_minus_delta_theta))
        modified_theta[option] = theta_minus_2pi[option]
        weight_ps[judgement] = 1 + xp.sin(np.pi / 2 * ((modified_theta - self.theta_1) / self.delta_theta))[judgement]

        option_1 = (theta >= theta_1_add_delta_theta)
        option_2 = (theta < theta_2_minus_delta_theta) | ((theta_minus_2pi < theta_2_minus_delta_theta) & (theta_minus_2pi >= theta_1_add_delta_theta))
        judgement = option_1 & option_2
        weight_ps[judgement] = 2

        option_1 = (theta >= theta_2_minus_delta_theta)
        option_2 = (theta < theta_2_add_delta_theta) | ((theta_minus_2pi < theta_2_add_delta_theta) & (theta_minus_2pi >= theta_2_minus_delta_theta))
        judgement = option_1 & option_2
        modified_theta = theta
        option = ((theta_minus_2pi < theta_2_add_delta_theta) & (theta_minus_2pi >= theta_2_minus_delta_theta))
        modified_theta[option] = theta_minus_2pi[option]
        weight_ps[judgement] = 1 - xp.sin(np.pi / 2 * ((modified_theta - self.theta_2) / self.delta_theta))[judgement]

        # xp_cz = xp.array([np.full([self.recon_dim[0], self.recon_dim[1]], i) for i in self.cz])
        # xp_r = xp.tile(xp.asarray(self.r), (self.recon_dim[2], 1, 1))
        # xp_phi = xp.tile(self.phi, (self.recon_dim[2], 1, 1))
        #
        # option = self.weight_t < 1
        # weight_ps[option] = (1 + ((2*self.projection.source_to_iso_center - xp_cz) / (xp_r + self.projection.source_to_iso_center)) * xp.cos(theta - xp_phi))[option]
        weight_fs = 1
        weight_c = (1 - xp.asarray(self.weight_t)) * weight_fs + xp.asarray(self.weight_t) * weight_ps
        return weight_c.transpose([2, 0, 1])