import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation
import random


class MatrixMunipulate():
    def __init__(self):
        self.last_matrix = np.eye(4)
        self.new_matrix_received = False

    def init_class(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')

    def convert_to_matrix(self, vector):
        matrix = np.eye(4)
        matrix[:3, 3] = vector[:3]
        matrix[:3, :3] = Rotation.from_euler('xyz', vector[3:], degrees=True).as_matrix()
        return matrix

    def generate_random_matrix(self):
        # position = np.random.rand(3)
        position = [random.uniform(-2, 2) for _ in range(3)]
        rotation_angles = np.random.rand(3) * 2 * np.pi
        r = Rotation.from_euler('xyz', rotation_angles)
        rotation_matrix = r.as_matrix()
        translation_matrix = np.eye(4)
        translation_matrix[:3, 3] = position
        translation_matrix[:3, :3] = rotation_matrix
        return translation_matrix

    def plot_matrix(self, matrix, scale=1000):
        self.ax.cla()
        self.ax.set_xlim(-1, 1)
        self.ax.set_ylim(-1, 1)
        self.ax.set_zlim(-1, 1)
        self.ax.set_xlabel("X", fontsize=16)
        self.ax.set_ylabel("Y", fontsize=16)
        self.ax.set_zlabel("Z", fontsize=16)

        colors = ['red', 'lime', 'blue']
        matrix[:3, 3] = matrix[:3, 3] / scale
        x, y, z = matrix[:3, 3]
        for j in range(3):
            vector = matrix[:3, j]
            self.ax.quiver(x, y, z, vector[0], vector[1], vector[2],
                           color=colors[j], arrow_length_ratio=0.1, length=0.1, normalize=True)

        self.ax.text(x, y, z, 'vector', color='black', fontsize=10)
        plt.draw()
        plt.pause(0.01)

    def plot_matrices(self, matrices, labels=None, scale=1000):
        # fig = plt.figure()
        # self.ax = fig.add_subplot(111, projection='3d')
        self.ax.cla()
        self.ax.set_xlim(-2, 2)
        self.ax.set_ylim(-2, 2)
        self.ax.set_zlim(-2, 2)
        self.ax.set_xlabel("X", fontsize=16)
        self.ax.set_ylabel("Y", fontsize=16)
        self.ax.set_zlabel("Z", fontsize=16)

        colors = ['red', 'lime', 'blue']
        for i, m in enumerate(matrices):
            m[:3, 3] = m[:3, 3] / scale
            x, y, z = m[:3, 3]
            for j in range(3):
                vector = m[:3, j]
                self.ax.quiver(x, y, z, vector[0], vector[1], vector[2],
                               color=colors[j], arrow_length_ratio=0.1, length=0.1, normalize=True)

            label = labels[i] if labels is not None else f"matrix {i + 1}"
            self.ax.text(x, y, z, label, color='black', fontsize=12)

        if labels is not None:
            self.ax.legend(fontsize=12, loc='best')

        plt.draw()
        plt.pause(0.01)

    def update_matrix(self, new_matrix):
        self.last_matrix = new_matrix
        self.new_matrix_received = True

    def show_single_frame_with_matrices(self, matrixes: list, labels: list = None, scale=1000):
        self.plot_matrices(matrixes, labels, scale)
        plt.show()

    def rotate_matrix(self, matrix: np.ndarray, axis: str, angle: float) -> np.ndarray:
        """
        Rotate a matrix around a given axis
        :param matrix: transformation matrix
        :param axis: axis to rotate around (x, y, z)
        :param angle: angle to rotate in degrees
        :return: rotated matrix
        """
        r = np.eye(4)
        r[:3, :3] = Rotation.from_euler(axis, angle, degrees=True).as_matrix()
        return np.dot(matrix, r)

    def show_matrices_animation(self, matrixes: list, labels: list = None, scale=1000):
        plt.ion()
        self.plot_matrices(matrixes, labels, scale)
        plt.show()

    def extract_angles_from_matrix(self, matrix: np.ndarray) -> list:
        """
        Extract the angles from a transformation matrix
        :param matrix: transformation matrix
        :return: [roll, pitch, yaw]
        """
        r = Rotation.from_matrix(matrix[:3, :3])
        return r.as_euler('xyz', degrees=True).tolist()

    def extract_pose_from_mat(self, matrix: np.ndarray) -> list:
        """
        Extract the pose from a transformation matrix
        :param matrix: transformation matrix
        :return: [x, y, z, roll, pitch, yaw]
        """
        x, y, z = matrix[:3, 3]
        rxryrz = Rotation.from_matrix(matrix[:3, :3]).as_euler('xyz', degrees=True).tolist()

        return [x, y, z] + rxryrz

    def find_angle_between_two_vectors(self, point1, point2):
        x1, y1 = point1
        x2, y2 = point2
        angle_rad = np.arctan2(y2 - y1, x2 - x1)
        angle_deg = np.degrees(angle_rad)
        return angle_deg - 90

    # find distanse between two points
    def distance_bt_2_poses(self, pose1, pose2):
        x1, y1, z1 = pose1[:3]
        x2, y2, z2 = pose2[:3]
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)

    def move_matrix_to_target(self, matrix: np.ndarray, offset: list) -> np.ndarray:
        """
        Move a matrix towards a target matrix
        :param matrix: transformation matrix
        :param offset: offset to move
        :return: moved matrix
        """
        Tm_offset = np.eye(4)
        Tm_offset[:3, 3] = offset
        new_matrix = matrix @ Tm_offset
        return new_matrix
