import numpy as np
import cv2 as cv
import open3d as o3d
from imageio import imread
from scipy.spatial.transform import Rotation


def lerp(x, x1, x2, y1, y2):
    # Figure out how 'wide' each range is
    x_span = x2 - x1
    y_span = y2 - y1
    # Convert the left range into a 0-1 range (float)
    scaled = (x - x1) / x_span
    # Convert the 0-1 range into a value in the right range.
    return y1 + (scaled * y_span)


def reproject(depth, fov, condition):
    def _normalize(x, dim, f):
        return (2 * x / dim - 1) * np.tan(f / 2)

    point_cloud = list()

    h, w = np.shape(depth)
    fov_x = fov
    fov_y = fov * h / w
    delta = 0.5

    for i in range(h):
        for j in range(w):
            if condition(depth[i, j]):
                v = -_normalize(i + delta, h, fov_x)
                u = -_normalize(j + delta, w, fov_y)
                pt = np.array([u, v, 1])
                pt = pt * depth[i, j]
                point_cloud.append(pt)

    return point_cloud


def rotation_matrix_from_quaternion(quaternion):
    r = Rotation.from_quat(quaternion)
    return r.as_matrix()


def transform_matrix_vector_quaternion(vector, quaternion):
    rotation_matrix = rotation_matrix_from_quaternion(quaternion)
    translation_vector = np.array([vector[0], vector[1], vector[2]])
    bottom_row = np.array([0.0, 0.0, 0.0, 1.0])
    out = np.hstack((rotation_matrix, translation_vector.reshape(3, 1)))
    out = np.vstack((out, bottom_row))
    return out


def to_original_depth(x, near, far):
    return (near / x - far) / (near - far)


# def transform_matrix_from_point(xs, ys, zs, towards_origin=True):
#     target = np.array([xs, ys, zs])
#     target = target / np.linalg.norm(target)
#     right = np.array([-ys, xs, 0])
#     right = right / np.linalg.norm(right)
#
#     if towards_origin:
#         target = -target
#         right = -right
#
#     up = np.cross(target, right).tolist()
#     x = np.array([right[0], right[1], right[2], 0.0])
#     y = np.array([up[0], up[1], up[2], 0.0])
#     z = np.array([target[0], target[1], target[2], 0.0])
#     d = np.array([xs, ys, zs, 1.0])
#     return np.transpose(np.vstack((x, y, z, d)))

def inverse_transform_matrix(transform_matrix):
    return np.linalg.inv(transform_matrix)


def load_matrix(row0, row1, row2):
    r0 = np.array(row0)
    r1 = np.array(row1)
    r2 = np.array(row2)
    r3 = np.array([0.0, 0.0, 0.0, 1.0])
    matrix = np.vstack((r0, r1, r2, r3))
    print(matrix)
    return matrix


fov = np.deg2rad(53.97213)
near = 0.0
far = 3.0
depth_path1 = 'Depth/1335.png'
depth_path2 = 'Depth/1336.png'
image1 = np.array(imread(depth_path1))
print(image1)
image1 = image1[:, :, 0] / 255.0
depth1 = lerp(image1, 0, 1.0, near, far)
points1 = reproject(depth1, fov, lambda x: x <= 0.99 * far)
vector1 = [0.0, -1.0, 0.0]
vector2 = [0.0, 1.0, 0.0]
quaternion1 = [0.0, 0.7, 0.7, 0.0]
quaternion2 = [0.7, 0.0, 0.0, 0.7]
# 1, 2 - right hand
matrix1 = np.array([[-0.67831, -0.41740, -0.60471, 0.60471],
                    [-0.73477, 0.38533, 0.55824, -0.55824],
                    [0.00000, 0.82298, -0.56806, 0.56806],
                    [0.00000, 0.00000, 0.00000, 1.00000]])

matrix2 = np.array([[0.75257, -0.37408, -0.54194, 0.54194],
                    [-0.65851, -0.42751, -0.61935, 0.61935],
                    [0.00000, 0.82298, -0.56806, 0.56806],
                    [0.00000, 0.00000, 0.00000, 1.00000]])
# matrix1 = np.array([[-1.0, 0.0, 0.0, 0],
#                     [0.0, 0.0, 1.0, -1.0],
#                     [0.00000, 1.0, 0.0, 0.0],
#                     [0.00000, 0.00000, 0.00000, 1.00000]])
# matrix2 = np.array([[1.0, 0.0, 0.0, 0.0],
#                     [0.0, 0.0, -1.0, 1.0],
#                     [0.00000, 1.0, 0.0, 0.0],
#                     [0.00000, 0.00000, 0.00000, 1.00000]])
# 3, 4 - Unity
matrix3 = np.array([[0.99965, 0.00107, 0.02660, -0.02660],
                    [0.00000, 0.99919, -0.04027, 0.04027],
                    [0.02662, 0.04025, 0.99883, -0.99883],
                    [0.00000, 0.00000, 0.00000, 1.00000]])
matrix4 = np.array([[0.99965, 0.00107, 0.02660, -0.02660],
                    [0.00000, 0.99919, -0.04027, 0.04027],
                    [0.02662, -0.04025, -0.99883, 0.99883],
                    [0.00000, 0.00000, 0.00000, 1.00000]])
pcl1 = o3d.geometry.PointCloud()
pcl1.points = o3d.utility.Vector3dVector(points1)
transform_matrix1 = transform_matrix_vector_quaternion(vector1, quaternion1)
transform_matrix2 = transform_matrix_vector_quaternion(vector2, quaternion2)
inverse_matrix1 = inverse_transform_matrix(transform_matrix2)
pcl1.transform(matrix1)
o3d.write_point_cloud("pcl1.ply", pcl1)
image2 = np.array(imread(depth_path2))
image2 = image2[:, :, 0] / 255.0
depth2 = lerp(image2, 0.0, 1.0, near, far)
points2 = reproject(depth2, fov, lambda x: x <= 0.99 * far)
pcl2 = o3d.geometry.PointCloud()
pcl2.points = o3d.utility.Vector3dVector(points2)
inverse_matrix2 = inverse_transform_matrix(transform_matrix2)
pcl2.transform(matrix2)
# print(transform_matrix1)
# print("-----------")
# print(matrix1)
# print("-----------")
# print(transform_matrix2)
# print("-----------")
# print(matrix4)
o3d.write_point_cloud("pcl2.ply", pcl2)
o3d.visualization.draw_geometries([pcl1, pcl2])
