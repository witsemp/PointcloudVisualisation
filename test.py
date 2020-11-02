import numpy as np
import cv2 as cv
import open3d as o3d
from imageio import imread


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
                print(depth[i, j])
                point_cloud.append(pt)

    return point_cloud


fov = np.deg2rad(53.97213)
near = 1.0
far = 3.0
depth_path1 = 'Depth/870.png'
depth_path2 = 'Depth/871.png'
image1 = np.array(imread(depth_path1))
image1 = image1[:, :, 0] / 255.0
depth1 = lerp(image1, 0.0, 1.0, 0.0, far)
points1 = reproject(depth1, fov, lambda x: x <= 0.99 * far)
pcl1 = o3d.geometry.PointCloud()
pcl1.points = o3d.utility.Vector3dVector(points1)
o3d.write_point_cloud("pcl1.ply", pcl1)
image2 = np.array(imread(depth_path2))
image2 = image2[:, :, 0] / 255.0
depth2 = lerp(image2, 0.0, 1.0, 0.0, far)
points2 = reproject(depth2, fov, lambda x: x <= 0.99 * far)
pcl2 = o3d.geometry.PointCloud()
pcl2.points = o3d.utility.Vector3dVector(points2)
o3d.write_point_cloud("pcl2.ply", pcl2)
# o3d.visualization.draw_geometries([pcl])
