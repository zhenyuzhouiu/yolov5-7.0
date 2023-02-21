import numpy as np


def poly_filter(polys, h, w):
    """
    filter oriented bounding box when their center is out of image
    :param polys: [num_box 8] xyxyxyxy
    :param h: image height
    :param w: image width
    :return:
    """
    # start:end:step
    x = polys[:, 0::2]
    y = polys[:, 1::2]
    x_min = x.min(1)
    x_max = x.max(1)
    y_min = y.min(1)
    y_max = y.min(1)
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2

    return (x_center > 0) & (x_center < w) & (y_center > 0) & (y_center < h)
