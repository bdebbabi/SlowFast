"""
Tracking from json extract
"""
import json

import numpy as np
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance

from slowfast.utils.parser import load_config, parse_args


def reduce_tasks(tasks):
    reduced_tasks = {}
    for pred in tasks:
        if pred['task_id'] not in reduced_tasks:
            reduced_tasks[pred['task_id']] = [pred['box']]
        else:
            reduced_tasks[pred['task_id']].append(pred['box'])
    return reduced_tasks


def metrics(bb1, bb2):
    def to_tuple(bb):
        w = bb['bottom_right_x'] - bb['top_left_x']
        h = bb['bottom_right_y'] - bb['top_left_y']
        return (bb['top_left_x'], bb['top_left_y'], h, w)
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

    def advance(self, new_bounding_boxes):
        if self.current_task_id == 0:  # first task
            self.tracking = new_bounding_boxes
            self.current_task_id += 1
            return

        new_matrix = np.zeros(
            (len(self.tracking), len(new_bounding_boxes)),
            dtype=np.float32)
        for i, bounding_box in enumerate(new_bounding_boxes):
            for j, tracked_bounding_box in enumerate(self.tracking):
                iou, hdiff, wdiff, cd = metrics(bounding_box,
                                       tracked_bounding_box)
                new_matrix[j, i] = iou

        matched_idx = linear_sum_assignment(new_matrix, maximize=True)
        print(">> matched_idx = ", matched_idx)
        self.current_task_id += 1
        self.tracking = new_bounding_boxes


def main(predictions_path):
    """
    Performs a hungarian traking on each tasks
    """
    with open(predictions_path, 'r') as json_file:
        data = json.load(json_file)
    data = reduce_tasks(data)

    tracker = HungarianTracker()
    for task_id in tqdm(data):
        tracker.advance(data[task_id])


if __name__ == "__main__":
    ARGS = parse_args()
    CFG = load_config(ARGS)
    PREDICTIONS_PATH = CFG.DEMO.PREDICTIONS_FILE_PATH
    main(PREDICTIONS_PATH)
