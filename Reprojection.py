import numpy as np
import open3d as o3d
import cv2 as cv
from lxml import etree
import re


def load_depth_image(depth_image_path: str):
    depth_image = cv.imread(depth_image_path, flags=cv.IMREAD_ANYDEPTH)
    return depth_image


def get_transform_matrix(xml_file_path: str, index: int):
    rows = []
    root = etree.parse(xml_file_path)
    file = root.findall('File[@Index='+'"'+str(index)+'"'+']')
    for row_index in range(4):
        row = str(etree.tostring(file[0].findall('MatrixRow'+str(row_index))[0]))
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
    depth_image_path1 = 'Depth/2248.png'
    depth_image = load_depth_image(depth_image_path1)
    get_transform_matrix(xml_file_path)
