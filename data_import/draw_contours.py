import cv2
import numpy as np

def get_points(points):
    pt_list = np.zeros([len(points), 1, 2])
    for ii, pt in enumerate(points):
        pt_list[ii, 0] = np.array([pt["x"], pt["y"]])
    return pt_list


def set_points(points):
    pt_list = [None] * len(points)
    for ii, pt in enumerate(points):
        pt_list[ii] = dict(x=pt[0, 0], y=pt[0, 1])
    return pt_list


def make_contour(points, anno_type, use_circle=False, point_size=1, **kwargs):
    def circle_points(center, r):
        alphas = np.linspace(0, 2 * np.pi, int(np.ceil(2 * np.pi * r)))
        return np.array([center[0] + r * np.cos(alphas), center[1] + r * np.sin(alphas)]).T[:, None, :]

    if anno_type == "polygon":
        pt_list = get_points(points)

    elif anno_type == "rectangle":
        pt_list = get_points(points)
        pt_list = np.array([pt_list[[0, 0, 1, 1], 0, 0], pt_list[[0, 1, 1, 0], 0, 1]]).T
    elif anno_type == "point":
        pt_list = get_points(points).astype(np.int)
        if point_size > 1:
            r = int(np.ceil(point_size / 2.0))
            pt_list = circle_points(pt_list[0, 0], r)
    elif anno_type == "line":
        pt_list = get_points(points)
        pt_list = np.floor(pt_list).astype(np.int)
        return [pt_list[ii:ii+2] for ii in range(len(pt_list)-1)]
    else:
        return None

    # drawContours works with pixel centers, but annotation tool use upper left corner as reference
    return [np.floor(pt_list).astype(np.int)]


def draw_contours2(segmentation, label_space, draw_buffer=None, drawContoursArgs=None, **kwargs):
    """
    If more than four channels are in the label space only values 1 will be drawn to the segmentation
    :param segmentation:
    :param label_space:
    :param draw_buffer: input draw bufffer, use to draw on top of existing images
    :param drawContoursArgs: Args for drawContours.. eg thickness to draw non filled contours
    :param kwargs: args for make_contours
    :return:
    """
    if not kwargs:
        kwargs = dict(use_circle=False)

    if draw_buffer is None:
        shape = (segmentation["image"]["height"], segmentation["image"]["width"])
        first_label = next(iter(label_space.values()))
        shape = (*shape, len(first_label))
        cont = np.zeros(shape, dtype=np.float32)
    else:
        cont = draw_buffer

    for lbl, color in label_space.items():
        color = color if isinstance(color, (tuple, list)) else color.tolist()
        contours = []
        for anno in segmentation["annotations"]:
            if lbl in anno["label"] or (isinstance(lbl, tuple) and np.any([lbl_ii in anno["label"] for lbl_ii in lbl])):
                contour = make_contour(anno["points"], anno["type"], **kwargs)
                if contour is not None:
                    contours.extend(contour)
        if drawContoursArgs is None:
            drawContoursArgs = dict(thickness=cv2.FILLED)
        if len(color) <= 4:
            cv2.drawContours(cont, contours, -1, tuple(color), **drawContoursArgs)
        else:
            cont[..., np.array(color) == 1] = cv2.drawContours(cont[..., np.array(color) == 1].copy(), contours, -1, 1,
                                                               **drawContoursArgs)

    return cont


# def extract_bounding_box(segmentation, label_space, draw_buffer=None, drawContoursArgs=None, **kwargs):
#     """
#     If more than four channels are in the label space only values 1 will be drawn to the segmentation
#     :param segmentation:
#     :param label_space:
#     :param draw_buffer: input draw bufffer, use to draw on top of existing images
#     :param drawContoursArgs: Args for drawContours.. eg thickness to draw non filled contours
#     :param kwargs: args for make_contours
#     :return:
#     """
#     if not kwargs:
#         kwargs = dict(use_circle=False)
#
#     if draw_buffer is None:
#         shape = (segmentation["image"]["height"], segmentation["image"]["width"])
#         first_label = next(iter(label_space.values()))
#         shape = (*shape, len(first_label))
#         cont = np.zeros(shape, dtype=np.float32)
#     else:
#         cont = draw_buffer
#
#     for lbl, color in label_space.items():
#         color = color if isinstance(color, (tuple, list)) else color.tolist()
#         contours = []
#         for anno in segmentation["annotations"]:
#             if lbl in anno["label"] or (isinstance(lbl, tuple) and np.any([lbl_ii in anno["label"] for lbl_ii in lbl])):
#                 contour = make_contour(anno["points"], anno["type"], **kwargs)
#                 if contour is not None:
#                     contours.extend(contour)
#         if drawContoursArgs is None:
#             drawContoursArgs = dict(thickness=cv2.FILLED)
#         if len(color) <= 4:
#             cv2.drawContours(cont, contours, -1, tuple(color), **drawContoursArgs)
#         else:
#             cont[..., np.array(color) == 1] = cv2.drawContours(cont[..., np.array(color) == 1].copy(), contours, -1, 1,
#                                                                **drawContoursArgs)
#
#     return cont