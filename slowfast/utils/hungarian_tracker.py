import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance


def metrics(bb1, bb2):
    def to_tuple(bb):
        w = bb[2] - bb[0]
        h = bb[3] - bb[1]
        return (bb[1], bb[2], h, w)
    i1, j1, h1, w1 = to_tuple(bb1)
    i2, j2, h2, w2 = to_tuple(bb2)

    top = max(i1, i2)
    bottom = min(i1 + h1, i2 + h2)
    left = max(j1, j2)
    right = min(j1 + w1, j2 + w2)

    overlap_height = bottom - top
    overlap_width = right - left

    if overlap_height <= 0 or overlap_width <= 0:
        overlap_area = 0
    else:
        overlap_area = overlap_height * overlap_width

    area1 = h1 * w1
    area2 = h2 * w2

    union_area = area1 + area2 - overlap_area
    iou = overlap_area / union_area

    #height difference
    hdiff = abs(h1-h2)
    #width difference
    wdiff = abs(w1-w2)

    #center coordinates for box1
    cx1, cy1 = j1 + w1/2, i1 + h1/2
    #center coordinates for box2
    cx2, cy2 = j2 + w2/2, i2 + h2/2

    #euclidean distance between the 2 box centers
    cd = distance.euclidean((cx1, cy1), (cx2, cy2))

    return iou, hdiff, wdiff, cd


class HungarianTracker():
    def __init__(self):
        self.current_task_id = 0
        self.tracking = []
        self.ids = []
        self.id_counter = 0

    def new_id(self):
        new_id = self.id_counter
        self.id_counter += 1
        return new_id

    def advance(self, new_bounding_boxes):
        """
        Takes the bounding boxes and returns an id for each one
        If the number of boxes doesn't match the previous task, then new ids
        are assigned for non matched bounding boxes or ids are dropped
        if the number is lower.

        Parameters
        ----------
            new_bounding_boxes: [Tuple(int, 4)]
        Returns
        -------
            ids: [int]
        """
        if self.current_task_id == 0:  # first task
            self.tracking = new_bounding_boxes
            self.ids = list(range(len(new_bounding_boxes)))
            self.id_counter = len(new_bounding_boxes)
            self.current_task_id += 1
            return self.ids

        new_matrix = np.zeros(
            (len(self.tracking), len(new_bounding_boxes)),
            dtype=np.float32)
        for i, bounding_box in enumerate(new_bounding_boxes):
            for j, tracked_bounding_box in enumerate(self.tracking):
                iou, hdiff, wdiff, cd = metrics(bounding_box,
                                                tracked_bounding_box)
                new_matrix[j, i] = iou if iou !=0 else -cd/1000

        row_idx, col_idx = linear_sum_assignment(new_matrix, maximize=True)
        # print(f">> task {self.current_task_id} row_idx = ", row_idx)
        self.current_task_id += 1
        new_ids = [
            self.new_id() if i not in col_idx else 0
            for i in range(len(new_bounding_boxes))]  # init new ids
        for row, col in zip(row_idx, col_idx):
            new_ids[col] = self.ids[row]  # reuse id if match
        # print(f">> task {self.current_task_id} ids = {new_ids}")
        self.tracking = new_bounding_boxes
        self.ids = new_ids

        return self.ids
