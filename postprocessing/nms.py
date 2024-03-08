import numpy as np

def non_maximum_suppression(boxes, scores, threshold):
    """
    Apply Non-Maximum Suppression (NMS) to suppress overlapping bounding boxes based on their confidence scores.

    Parameters:
    - boxes: List of bounding boxes, each defined by [x, y, width, height].
    - scores: Confidence scores for each bounding box.
    - threshold: IoU (Intersection over Union) threshold to determine when a box should be suppressed.

    Returns:
    - keep_boxes: Indices of boxes to keep after NMS.
    """
    # Convert [x, y, width, height] to [x1, y1, x2, y2]
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = x1 + boxes[:, 2]
    y2 = y1 + boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        intersection = w * h
        iou = intersection / (areas[i] + areas[order[1:]] - intersection)

        inds = np.where(iou <= threshold)[0]
        order = order[inds + 1]

    return keep
