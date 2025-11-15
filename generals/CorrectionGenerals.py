from generals.ProjectionGenerals import ProjectionGenerals
from generals.ReconGenerals import ReconGenerals


class CorrectionGenerals:
    def __init__(self, projection: ProjectionGenerals, reconstruction: ReconGenerals):
        self.projection = projection
        self.reconstruction = reconstruction

