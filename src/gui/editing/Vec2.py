import numpy as np
from PyQt5.QtCore import QPointF


class Vector2D(QPointF):
    def __init__(self, x, y):
        super().__init__(x, y)

    def __neg__(self):
        return Vector2D(-self.x(), -self.y())

    def dot_product(self, other):
        return np.dot(self.to_array(), other.to_array())

    def cross_product(self, other):
        return np.cross(self.to_array(), other.to_array())

    def length(self):
        return np.linalg.norm(self.to_array())

    def length_square(self):
        return np.sum(self.to_array() ** 2)

    def normalized(self):
        norm = np.linalg.norm(self.to_array())
        return Vector2D(self.x() / norm, self.y() / norm)

    def distance_to(self, other):
        return np.linalg.norm(self.to_array() - other.to_array())

    def angle_to(self, other):
        unit_self = self.normalized()
        unit_other = other.normalized()
        dot_product = unit_self.dot_product(unit_other)
        return np.arccos(np.clip(dot_product, -1.0, 1.0))

    def rotated(self, angle):
        radian_angle = np.radians(angle)
        rotation_matrix = np.array([[np.cos(radian_angle), -np.sin(radian_angle)],
                                    [np.sin(radian_angle), np.cos(radian_angle)]])
        rotated_vector = np.dot(rotation_matrix, self.to_array())
        return Vector2D(rotated_vector[0], rotated_vector[1])

    def to_numpy(self):
        return np.array([self.x(), self.y()])