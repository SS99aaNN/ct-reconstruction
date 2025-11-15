import numpy as np


class ScannerGenerals:
    def __init__(self, source_to_iso_center, source_to_detector, detector_offset_x, detector_offset_y, source_offset_x,
                 source_offset_y, in_plane_angle, out_of_plane_angle, num_of_projections):
        self.source_to_iso_center = source_to_iso_center
        self.source_to_detector = source_to_detector
        self.detector_offset_x = detector_offset_x
        self.detector_offset_y = detector_offset_y
        self.source_offset_x = source_offset_x
        self.source_offset_y = source_offset_y
        self.in_plane_angle = in_plane_angle
        self.out_of_plane_angle = out_of_plane_angle
        self.num_of_projections = num_of_projections
        self.half_cone_beam_angle = 20

        self.print_scanner_info()

    def print_scanner_info(self):
        print("Current Scanner info:")
        print("\tSource to ISO Center (mm): \t%.4f" % self.source_to_iso_center)
        print("\tSource to Detector (mm): \t%.4f" % self.source_to_detector)
        print("\tDetector Offset in X (mm): \t%.6f" % self.detector_offset_x)
        print("\tDetector Offset in Y (mm): \t%.6f" % self.detector_offset_y)
        print("\tSource Offset in X (mm): \t%.6f" % self.source_offset_x)
        print("\tSource Offset in Y (mm): \t%.6f" % self.source_offset_y)
        print("\tIn Plane Angle (degree): \t%.6f" % self.in_plane_angle)
        print("\tOut of Plane Angle (degree): \t%.6f" % self.out_of_plane_angle)
        print("\tNumber of Projections: \t%.6f" % self.num_of_projections)
