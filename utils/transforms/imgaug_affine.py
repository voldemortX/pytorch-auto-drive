# Adapted from imgaug,
# except we drop the 0.5 offset thing & 1e-4, to align with torchvision
# TorchVision: M = (T * C) * (R * S * Shr) * C^-1
# imgaug: M = (T * C) * (Shr * S * R) * C^-1
# S is just a scaling scalar here, so RS = SR, C & T are of course switchable.
# We align this with torchvision, which should be the same as original imgaug when not using shear,
# note that other than order inconsistency, the formula for shear is also different,
# changed to torchvision's shear.
import math
import numpy as np


class _AffineMatrixGenerator(object):
    def __init__(self, matrix=None):
        if matrix is None:
            matrix = np.eye(3, dtype=np.float32)
        self.matrix = matrix

    def translate(self, x_px, y_px):
        matrix = np.array([
            [1, 0, x_px],
            [0, 1, y_px],
            [0, 0, 1]
        ], dtype=np.float32)
        self._mul(matrix)
        return self

    def scale(self, x_frac, y_frac):
        matrix = np.array([
            [x_frac, 0, 0],
            [0, y_frac, 0],
            [0, 0, 1]
        ], dtype=np.float32)
        self._mul(matrix)
        return self

    def rotate(self, rad):
        rad = -rad
        matrix = np.array([
            [np.cos(rad), np.sin(rad), 0],
            [-np.sin(rad), np.cos(rad), 0],
            [0, 0, 1]
        ], dtype=np.float32)
        self._mul(matrix)
        return self

    def shear(self, x_rad, y_rad):
        # ShearX before ShearY
        matrix = np.array([
            [1, -np.tan(x_rad), 0],
            [-np.tan(y_rad), np.tan(x_rad) * np.tan(y_rad) + 1, 0],
            [0, 0, 1]
        ], dtype=np.float32)
        self._mul(matrix)
        return self

    def _mul(self, matrix):
        self.matrix = np.matmul(matrix, self.matrix)


def get_affine_matrix(center, angle, translate, scale, shear):
    rot = math.radians(angle)
    sx, sy = [math.radians(s) for s in shear]
    matrix_gen = _AffineMatrixGenerator()
    matrix_gen.translate(x_px=-center[0], y_px=-center[1])
    matrix_gen.shear(x_rad=sx, y_rad=sy)
    matrix_gen.scale(x_frac=scale, y_frac=scale)
    matrix_gen.rotate(rot)
    matrix_gen.translate(x_px=center[0], y_px=center[1])
    matrix_gen.translate(x_px=translate[0], y_px=translate[1])

    return matrix_gen.matrix
