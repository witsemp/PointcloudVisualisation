import numpy as np
import cv2 as cv
from imageio import imread
import open3d as o3d


def lerp(x, x1, x2, y1, y2):
    # Figure out how 'wide' each range is
    x_span = x2 - x1
    y_span = y2 - y1

    # Convert the left range into a 0-1 range (float)
    scaled = (x - x1) / x_span

    # Convert the 0-1 range into a value in the right range.
    return y1 + (scaled * y_span)


def pcl_from_images(depth_image, camera_matrix):
    fov = np.deg2rad(33.39849)
    near = 1.0
    far = 3.0
    pcl = []
    depth_image = depth_image[:, :, 0] / 255.0
    depth_image = lerp(depth_image, 0.0, 1.0, 0.0, far)
    rows, cols = np.shape(depth_image)
    for row in range(rows):
        for col in range(cols):
            world_coords = depth_image[row, col] * np.dot(camera_matrix, np.array([row, col, 1]).reshape(3, 1))
            pcl.append(world_coords)
    pcl = np.array(pcl).reshape(len(pcl), 3)
    print(pcl)
    return pcl


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


def main():
    fov = np.deg2rad(53.97213)
    near = 0.0
    far = 3.0
    f_depth_x = 1.0580353759137747e+03
    f_depth_y = 1.0604471766882732e+03
    c_depth_x = 9.5177505245974123e+02
    c_depth_y = 5.4839653660364445e+02
    depth_path1 = 'Depth/2248.png'
    depth_path2 = 'Depth/2249.png'
    img1_depth = np.array(imread(depth_path1))
    img2_depth = np.array(imread(depth_path2))
    camera_matrix = np.array(
        [1 / f_depth_x, 0., -c_depth_x / f_depth_x,
         0., 1 / f_depth_y, -c_depth_y / f_depth_y,
         0., 0., 1.]).reshape(3, 3)
    camera_matrix_opencv = np.array(
        [f_depth_x, 0., c_depth_x,
         0., f_depth_y, c_depth_y,
         0., 0., 1.]).reshape(3, 3)
    pcl1 = reproject(img1_depth, fov, lambda x: x <= 0.99 * far)
    point_cloud1 = o3d.PointCloud()
    point_cloud1.points = o3d.Vector3dVector(pcl1)
    pcl2 = reproject(img2_depth, fov, lambda x: x <= 0.99 * far)
    point_cloud2 = o3d.PointCloud()
    point_cloud2.points = o3d.Vector3dVector(pcl1)
    o3d.visualization.draw_geometries([point_cloud1, point_cloud2])


if __name__ == '__main__':
    main()
