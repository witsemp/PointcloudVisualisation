import numpy as np
import open3d as o3d
import cv2 as cv
from lxml import etree
import re
import pathlib


def load_depth_image(depth_image_index: int):
    base_path = 'Depth/'
    file_name = str(depth_image_index) + ".png"
    depth_image_path = base_path + file_name
    depth_image = cv.imread(depth_image_path, flags=cv.IMREAD_ANYDEPTH)
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


if __name__ == '__main__':
    xml_file_path = '/home/witsemp/work/UnityObjectRenderer/output.xml'
    index1 = 318
    index2 = 319
    depth_image1 = load_depth_image(index1)
    depth_image2 = load_depth_image(index2)
    transform_matrix1 = get_transform_matrix(xml_file_path, index1)
    transform_matrix2 = get_transform_matrix(xml_file_path, index2)
    print(transform_matrix1)
    print(transform_matrix2)
    cv.imshow('depth1', depth_image1)
    cv.imshow('depth2', depth_image2)
    cv.waitKey(0)
