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

    for i in range(h):
        for j in range(w):
            if condition(depth[i, j]):
                v = -_normalize(i + 0.5, h, fov_x)
                u = -_normalize(j + 0.5, w, fov_y)
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

fov = np.deg2rad(53.97213)
near = 0.5
far = 3.0
depth_path1 = 'Depth/1562.png'
depth_path2 = 'Depth/1563.png'
image1 = np.array(imread(depth_path1))
image1 = image1[:, :, 0] / 255.0
depth1 = lerp(image1, 0.0, 1.0, near, far)
points1 = reproject(depth1, fov, lambda x: x <= 0.99 * far)
vector1 = [0.9, 0.1, 0.4]
vector2 = [0.9, -0.1, 0.4]
quaternion1 = [0.6, -0.6, -0.4, 0.4]
quaternion2 = [-0.6, 0.6, 0.4, -0.4]
pcl1 = o3d.geometry.PointCloud()
pcl1.points = o3d.utility.Vector3dVector(points1)
transform_matrix1 = transform_matrix_vector_quaternion(vector2, quaternion2)
inverse_matrix1 = inverse_transform_matrix(transform_matrix1)
pcl1.transform(transform_matrix1)
o3d.write_point_cloud("pcl1.ply", pcl1)
image2 = np.array(imread(depth_path2))
image2 = image2[:, :, 0] / 255.0
depth2 = lerp(image2, 0.0, 1.0, 0.0, far)
points2 = reproject(depth2, fov, lambda x: x <= 0.99 * far)
pcl2 = o3d.geometry.PointCloud()
pcl2.points = o3d.utility.Vector3dVector(points2)
transform_matrix2 = transform_matrix_vector_quaternion(vector2, quaternion2)
inverse_matrix2 = inverse_transform_matrix(transform_matrix2)
pcl2.transform(transform_matrix2)
print(transform_matrix1)
print("-----------")
print(transform_matrix2)
o3d.write_point_cloud("pcl2.ply", pcl2)
o3d.visualization.draw_geometries([pcl1, pcl2])
