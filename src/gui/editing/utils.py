import numpy as np


def cm_to_inch(cm):
    return cm * 0.393701


def mm_to_inch(mm):
    return mm * 0.0393701


# def pixel_to_inch(pixel):
#    # I know this is not correct, but it is a good guess
#    return pixel * 0.00333

def perpendicular(vector):
    return np.array([-vector[1], vector[0]])


def normalize(vector):
    return vector / np.linalg.norm(vector)


def construct_arrow(start, end, width):
    vector = end - start
    per = normalize(perpendicular(vector))
    length = np.linalg.norm(vector) * width

    mid1 = start + per * length
    mid2 = start - per * length

    return [start, mid1, end, mid2, start]


def delete_content(info_dock):
    info_dock.currentRow = 0
    info_dock.widgets.clear()

    layout = info_dock.layout
    item = layout.itemAtPosition(0, 0)
    if item is not None:
        layout.removeItem(item)
        item.widget().setParent(None)


def compute_bbox(objects):
    points = []
    for obj in objects:
        compute_bbox_method = getattr(obj, "compute_bbox", None)
        if callable(compute_bbox_method):
            bbox = obj.compute_bbox()
            points.append([bbox[0][0], bbox[0][1]])
            points.append([bbox[1][0], bbox[1][1]])
    if not points:
        return None
    else:
        return compute_bbox_of_points(points)


def compute_bbox_of_points(points):
    x, y = points[0]
    bbox = [[x, y], [x, y]]
    for point in points[1:]:
        if point[0] < bbox[0][0]:
            bbox[0][0] = point[0]
        if point[1] < bbox[0][1]:
            bbox[0][1] = point[1]
        if bbox[1][0] < point[0]:
            bbox[1][0] = point[0]
        if bbox[1][1] < point[1]:
            bbox[1][1] = point[1]
    return bbox


def _compute_parameter(a, b, point):
    """
    :param a: [float, float]
    :param b: [float, float]
    :param point: [float, float]
    :return: t, such that point = a * (1 - t) + b * t
    """
    parameter0 = (point[0] - a[0]) / (b[0] - a[0])
    parameter1 = (point[1] - a[1]) / (b[1] - a[1])
    return [parameter0, parameter1]


def _compute_point(a, b, weight):
    new_point = [None, None]
    new_point[0] = a[0] * (1 - weight[0]) + b[0] * weight[0]
    new_point[1] = a[1] * (1 - weight[1]) + b[1] * weight[1]
    return new_point


def compute_weights(bbox, points):
    weights = []
    a = bbox[0]
    b = bbox[1]
    for point in points:
        parameter = _compute_parameter(a, b, point)
        weights.append(parameter)
    return weights


def compute_points(bbox, weights):
    points = []
    a = bbox[0]
    b = bbox[1]
    for weight in weights:
        point = _compute_point(a, b, weight)
        points.append(point)
    return points


def get_bigger_bbox(bbox1, bbox2):
    """
    :param bbox1:
    :param bbox2:
    :return:
    Assumption: One bounding box is contained inside another
    """

    min_x_1 = min([x for x, y in bbox1])
    min_y_1 = min([y for x, y in bbox1])
    max_x_1 = max([x for x, y in bbox1])
    max_y_1 = max([y for x, y in bbox1])

    min_x_2 = min([x for x, y in bbox2])
    min_y_2 = min([y for x, y in bbox2])
    max_x_2 = max([x for x, y in bbox2])
    max_y_2 = max([y for x, y in bbox2])

    if min_x_1 < min_x_2 or min_y_1 < min_y_2:
        return bbox1

    if max_x_2 < max_x_1 or max_y_2 < max_y_1:
        return bbox1

    return bbox2
