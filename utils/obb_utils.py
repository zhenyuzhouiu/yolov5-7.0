# circular gaussian label
# oriented bounding box to ploy
import cv2
import torch
import numpy as np
from math import pi


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


def regular_theta(theta, mode='180', start=-pi / 2):
    """
    limit theta ∈ [-pi/2, pi/2)
    """
    assert mode in ['360', '180']
    cycle = 2 * pi if mode == '360' else pi

    theta = theta - start
    theta = theta % cycle
    return theta + start


def gaussian_label_cpu(label, num_class, u=0, sig=4.0):
    """
    Convert to CSL Label with periodicity gaussian window function
    Args:
        label (float32):[1], theta class
        num_theta_class (int): [1], theta class num
        u (float32):[1], μ in gaussian function
        sig (float32):[1], σ in gaussian function, which is window radius for Circular Smooth Label
    Returns:
        csl_label (array): [num_theta_class], gaussian function smooth label
    """
    x = np.arange(-num_class / 2, num_class / 2)
    y_sig = np.exp(-(x - u) ** 2 / (2 * sig ** 2))
    index = int(num_class / 2 - label)
    return np.concatenate([y_sig[index:],
                           y_sig[:index]], axis=0)


def rbox2poly(obboxes):
    """
    Trans rbox format to poly format.
    Args:
        rboxes (array/tensor): (num_gts, [cx cy l s θ]) θ∈[0, 180)

    Returns:
        polys (array/tensor): (num_gts, [x1 y1 x2 y2 x3 y3 x4 y4])
    """
    if isinstance(obboxes, torch.Tensor):
        # angle [0, 180)
        center, w, h, angle = obboxes[:, :2], obboxes[:, 2:3], obboxes[:, 3:4], obboxes[:, 4:5]
        # radian = (angle-90)/180*pi  # [-pi/2, pi/2)
        radian = angle/180*pi  # [0, pi)
        Cos, Sin = torch.cos(radian), torch.sin(radian)

        vector1 = torch.cat(
            (w / 2 * Cos, -w / 2 * Sin), dim=-1)
        vector2 = torch.cat(
            (-h / 2 * Sin, -h / 2 * Cos), dim=-1)
        point1 = center + vector1 + vector2
        point2 = center + vector1 - vector2
        point3 = center - vector1 - vector2
        point4 = center - vector1 + vector2
        order = obboxes.shape[:-1]
        return torch.cat(
            (point1, point2, point3, point4), dim=-1).reshape(*order, 8)
    else:
        center, w, h, angle = np.split(obboxes, (2, 3, 4), axis=-1)
        # radian = (angle-90)/180*pi  # [-pi/2, pi/2)
        radian = angle/180*pi  # [0, pi)
        Cos, Sin = np.cos(radian), np.sin(radian)

        vector1 = np.concatenate(
            [w / 2 * Cos, -w / 2 * Sin], axis=-1)
        vector2 = np.concatenate(
            [-h / 2 * Sin, -h / 2 * Cos], axis=-1)

        point1 = center + vector1 + vector2
        point2 = center + vector1 - vector2
        point3 = center - vector1 - vector2
        point4 = center - vector1 + vector2
        order = obboxes.shape[:-1]
        return np.concatenate(
            [point1, point2, point3, point4], axis=-1).reshape(*order, 8)


def poly2hbb(polys):
    """
    Trans poly format to hbb format
    Args:
        rboxes (array/tensor): (num_gts, poly)

    Returns:
        hbboxes (array/tensor): (num_gts, [xc yc w h])
    """
    assert polys.shape[-1] == 8
    if isinstance(polys, torch.Tensor):
        x = polys[:, 0::2]  # (num, 4)
        y = polys[:, 1::2]
        x_max = torch.amax(x, dim=1)  # (num)
        x_min = torch.amin(x, dim=1)
        y_max = torch.amax(y, dim=1)
        y_min = torch.amin(y, dim=1)
        x_ctr, y_ctr = (x_max + x_min) / 2.0, (y_max + y_min) / 2.0  # (num)
        h = y_max - y_min  # (num)
        w = x_max - x_min
        x_ctr, y_ctr, w, h = x_ctr.reshape(-1, 1), y_ctr.reshape(-1, 1), w.reshape(-1, 1), h.reshape(-1, 1)  # (num, 1)
        hbboxes = torch.cat((x_ctr, y_ctr, w, h), dim=1)
    else:
        x = polys[:, 0::2]  # (num, 4)
        y = polys[:, 1::2]
        x_max = np.amax(x, axis=1)  # (num)
        x_min = np.amin(x, axis=1)
        y_max = np.amax(y, axis=1)
        y_min = np.amin(y, axis=1)
        x_ctr, y_ctr = (x_max + x_min) / 2.0, (y_max + y_min) / 2.0  # (num)
        h = y_max - y_min  # (num)
        w = x_max - x_min
        x_ctr, y_ctr, w, h = x_ctr.reshape(-1, 1), y_ctr.reshape(-1, 1), w.reshape(-1, 1), h.reshape(-1, 1)  # (num, 1)
        hbboxes = np.concatenate((x_ctr, y_ctr, w, h), axis=1)
    return hbboxes


def poly2obb(polys, num_cls_angle=180, radius=6.0, use_gaussian=True):
    """
    Trans poly format to oriented bounding box format.
    [x1 y1 x2 y2 x3 y3 x4 y4]:-> [x, y, l, s, angle]
    angle will be [0, 180) from the x-axis to the long side with clockwise
    Args:
        polys (array): (num_gts, [x1 y1 x2 y2 x3 y3 x4 y4])
        num_cls_angle (int): [1], theta class num
        radius (float32): [1], window radius for Circular Smooth Label
        use_gaussian: True using gaussian window function

    Returns:
        use_gaussian True:
            rboxes (array):
            csl_labels (array): (num_gts, num_cls_thata)
        elif
            rboxes (array): (num_gts, [cx cy l s angle])
    """
    assert polys.shape[-1] == 8
    if use_gaussian:
        csl_labels = []
    oboxes = []
    for poly in polys:
        poly = np.float32(poly.reshape(4, 2))
        (x, y), (w, h), angle = cv2.minAreaRect(poly)  # opencv angle ∈ [-90, 0)

        # trans opencv format to long-side format with angle [0, 180)
        if w != max(w, h):
            w, h = h, w  # change the long side of bounding box
            angle += 90.0
        else:
            angle = angle + 180.0

        oboxes.append([x, y, w, h, angle])  # w>h with angle [0, 180)
        if use_gaussian:
            csl_label = gaussian_label_cpu(label=angle, num_class=num_cls_angle, u=0, sig=radius)
            csl_labels.append(csl_label)
    if use_gaussian:
        return np.array(oboxes), np.array(csl_labels)
    return np.array(oboxes)


if __name__ == "__main__":
    print("For testing rbox2poly function")