import numpy as np
import open3d as o3d
import cv2 as cv
from lxml import etree
import re
from scipy.spatial.transform import Rotation


def lerp(x, x1, x2, y1, y2):
    # Figure out how 'wide' each range is
    x_span = x2 - x1
    y_span = y2 - y1
    # Convert the left range into a 0-1 range (float)
    scaled = (x - x1) / x_span
    # Convert the 0-1 range into a value in the right range.
    return y1 + (scaled * y_span)


def load_depth_image(depth_image_index: int):
    base_path = '/home/witsemp/saved_big/chair/1a74a83fa6d24b3cacd67ce2c72c02e/depth/'
    file_name = str(depth_image_index) + ".png"
    depth_image_path = base_path + file_name
    depth_image = cv.imread(depth_image_path, flags=cv.IMREAD_ANYDEPTH)
    print(depth_image.shape)
    return depth_image


def get_transform_matrix(xml_file_path: str, index: int):
    rows = []
    root = etree.parse(xml_file_path)
    file = root.findall('File[@Index=' + '"' + str(index) + '"' + ']')
    for row_index in range(4):
        row = str(etree.tostring(file[0].findall('MatrixRow' + str(row_index))[0]))
        try:
            row = re.search(rf'<MatrixRow{row_index}>(.+?)</MatrixRow{row_index}>', row).group(1)
        except AttributeError:
            row = ''
        row = row.replace("(", '').replace(")", '').split(', ')
        row = [float(element) for element in row]
        rows.append(row)
    transform_matrix = np.array([rows[0], rows[1], rows[2], rows[3]])
    return transform_matrix

def get_vector_quaternion(xml_file_path: str, index: int):
    rows = []
    root = etree.parse(xml_file_path)
    file = root.findall('File[@Index=' + '"' + str(index) + '"' + ']')
    position = str(etree.tostring(file[0].findall('Position')[0]))
    position = re.search(rf'<Position>(.+?)</Position>', position).group(1)
    position = position.replace("(", '').replace(")", '').split(', ')
    rotation = str(etree.tostring(file[0].findall('Rotation')[0]))
    rotation = re.search(rf'<Rotation>(.+?)</Rotation>', rotation).group(1)
    rotation = rotation.replace("(", '').replace(")", '').split(', ')
    position = [float(element) for element in position]
    rotation = [float(element) for element in rotation]
    return position, rotation

def inverse_transform_matrix(transform_matrix):
    return np.linalg.inv(transform_matrix)

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


def reproject(depth, fov, condition):
    def _normalize(x, dim, f):
        return (2 * x / dim - 1) * np.tan(f / 2)

    point_cloud = list()
    h, w = np.shape(depth)
    fov_x = fov * h / w
    fov_y = fov
    delta = 0.5
    for i in range(h):
        for j in range(w):
            if condition(depth[i, j]):
                v = -_normalize(i + delta, h, fov_y)
                u = -_normalize(j + delta, w, fov_x)
                pt = np.array([u, v, 1])
                pt = pt * depth[i, j]
                point_cloud.append(pt)
    return point_cloud


def pcl_from_images(depth_image, camera_matrix):
    pcl = []
    rows, cols = np.shape(depth_image)
    for row in range(rows):
        for col in range(cols):
            world_coords = depth_image[row, col] * np.dot(camera_matrix, np.array([col, row, 1]).reshape(3, 1))
            pcl.append(world_coords)
    pcl = np.array(pcl).reshape(len(pcl), 3)
    return pcl


def reproject_with_camera_matrix(index1, index2, xml_file_path, camera_matrix, visualise_images=False):
    div = 10000.0
    depth_image1 = load_depth_image(index1)
    depth_image2 = load_depth_image(index2)
    print("Type:\n", depth_image1.dtype)
    print("---------")
    print("Shape:\n", depth_image1.shape)
    print("---------")
    print("Image 1 values:\n", depth_image1)
    print("---------")
    print("Image 2 values:\n", depth_image2)
    print("---------")
    depth_image1_metric = depth_image1 / div
    depth_image2_metric = depth_image2 / div
    print("Scaled to metric:\n", depth_image1_metric)
    print("---------")
    print("Scaled to metric:\n", depth_image2_metric)
    transform_matrix1 = get_transform_matrix(xml_file_path, index1)
    transform_matrix2 = get_transform_matrix(xml_file_path, index2)
    inverse_matrix1 = inverse_transform_matrix(transform_matrix1)
    inverse_matrix2 = inverse_transform_matrix(transform_matrix2)
    pos1, rot1 = get_vector_quaternion(xml_file_path, index1)
    pos2, rot2 = get_vector_quaternion(xml_file_path, index2)
    transform_matrix1_vq = transform_matrix_vector_quaternion(pos1, rot1)
    transform_matrix2_vq = transform_matrix_vector_quaternion(pos2, rot2)
    inverse_matrix1 = inverse_transform_matrix(transform_matrix1)
    inverse_matrix2 = inverse_transform_matrix(transform_matrix2)
    inverse_matrix1_vq = inverse_transform_matrix(transform_matrix1_vq)
    inverse_matrix2_vq = inverse_transform_matrix(transform_matrix2_vq)
    pcl1 = pcl_from_images(depth_image1_metric, camera_matrix)
    pcl2 = pcl_from_images(depth_image2_metric, camera_matrix)
    point_cloud1 = o3d.PointCloud()
    point_cloud2 = o3d.PointCloud()
    point_cloud1.points = o3d.Vector3dVector(pcl1)
    point_cloud2.points = o3d.Vector3dVector(pcl2)
    point_cloud1.transform(transform_matrix1)
    point_cloud2.transform(transform_matrix2)
    # point_cloud2.transform(np.dot(inverse_matrix2, transform_matrix1))
    o3d.visualization.draw_geometries([point_cloud1, point_cloud2])
    if visualise_images:
        cv.imshow('depth1', depth_image1)
        cv.imshow('depth2', depth_image2)
        cv.waitKey(0)


def reproject_with_fov(index1, index2, xml_file_path, fov, near, far, visualise_images=False):
    div = 10000.0
    depth_image1 = load_depth_image(index1)
    depth_image2 = load_depth_image(index2)
    print("Type:\n", depth_image1.dtype)
    print("---------")
    print("Image values:\n", depth_image1)
    print("---------")
    print("Image 2 values:\n", depth_image2)
    print("---------")
    depth_image1_metric = depth_image1 / div
    depth_image2_metric = depth_image2 / div
    print("Scaled to metric 1:\n", depth_image1_metric)
    print("---------")
    print("Scaled to metric 2:\n", depth_image2_metric)
    transform_matrix1 = get_transform_matrix(xml_file_path, index1)
    transform_matrix2 = get_transform_matrix(xml_file_path, index2)
    pcl1 = reproject(depth_image1_metric, fov, lambda x: near <= x <= 0.99 * far)
    pcl2 = reproject(depth_image2_metric, fov, lambda x: near <= x <= 0.99 * far)
    point_cloud1 = o3d.PointCloud()
    point_cloud2 = o3d.PointCloud()
    point_cloud1.points = o3d.Vector3dVector(pcl1)
    point_cloud2.points = o3d.Vector3dVector(pcl2)
    point_cloud1.transform(transform_matrix1)
    point_cloud2.transform(transform_matrix2)
    # point_cloud2.transform(np.dot(inverse_matrix2, transform_matrix1))
    o3d.visualization.draw_geometries([point_cloud1, point_cloud2])
    if visualise_images:
        cv.imshow('depth1', depth_image1)
        cv.imshow('depth2', depth_image2)
        cv.waitKey(0)


if __name__ == '__main__':
    xml_file_path = '/home/witsemp/work/UnityObjectRenderer/output.xml'
    # f_depth_x = 1.0580353759137747e+03
    # f_depth_y = 1.0604471766882732e+03
    # c_depth_x = 9.5177505245974123e+02
    # c_depth_y = 5.4839653660364445e+02
    f_depth_x = 1062.3
    f_depth_y = 1062.3
    c_depth_x = 639.5
    c_depth_y = 479.5
    fov = np.deg2rad(48.63164)
    near = 0.5
    far = 3.0
    camera_matrix = np.array([[1 / f_depth_x, 0., -c_depth_x / f_depth_x],
                              [0., 1 / f_depth_y, -c_depth_y / f_depth_y],
                              [0., 0., 1.]])

    camera_matrix_opencv = np.array(
        [[f_depth_x, 0., c_depth_x],
         [0., f_depth_y, c_depth_y],
         [0., 0., 1.]])
    # 2, 1500 - ok, 1990 - nie ok
    index1 = 2
    index2 = index1 + 1
    # reproject_with_fov(index1, index2, xml_file_path, fov, near, far, visualise_images=True)
    reproject_with_camera_matrix(index1, index2, xml_file_path, camera_matrix, visualise_images=False)
