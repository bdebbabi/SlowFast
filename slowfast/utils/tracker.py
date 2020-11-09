import os.path
import urllib.request

import numpy as np

from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import multivariate_normal

import torch
import torchvision

from slowfast.utils import feature_extractor

from deep_sort.tracker import Tracker
from deep_sort.detection import Detection
from deep_sort import nn_matching


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


def download_file(url, file_path):
    urllib.request.urlretrieve(url, file_path)


def load_encoder(model_path="mars-small128.pb"):
    if not os.path.isfile(model_path):
        download_file(
            'https://github.com/theAIGuysCode/yolov4-deepsort/blob/9e745bfb3ea5e7c7505cb11a8e8654f5b1319ad9/model_data/mars-small128.pb?raw=true',
            model_path)
    return feature_extractor.create_box_encoder(model_path, batch_size=1)


class DeepSortTracker():
    def __init__(self, max_age=120, max_cosine_distance=0.4):
        nn_budget = None
        metric = nn_matching.NearestNeighborDistanceMetric(
            "cosine", max_cosine_distance, nn_budget)
        self.tracker = Tracker(metric, max_age=max_age)
        self.encoder = load_encoder()

    def advance(self, new_bounding_boxes, new_frame):
        features = self.encoder(new_frame, new_bounding_boxes)
        detections = [
            Detection(bbox, 1.0, feature)  # TODO: extract confidence from detectron2
            for bbox, feature in zip(new_bounding_boxes, features)
        ]

        self.tracker.predict()
        self.tracker.update(detections)

        box_ids = [
            track.track_id
            for track in self.tracker.tracks
            if track.is_confirmed() and track.time_since_update < 1
        ]
        bboxes = [
            track.to_tlwh()
            for track in self.tracker.tracks
            if track.is_confirmed() and track.time_since_update < 1
        ]
        return box_ids, bboxes


class HungarianTracker():
    def __init__(self):
        self.current_task_id = 0
        self.tracking = []
        self.embeddings = [] # crop embeddings TODO: store for more than 1 prediction
        self.ids = []
        self.id_counter = 0
        self.encoder = load_encoder()

    def new_id(self):
        new_id = self.id_counter
        self.id_counter += 1
        return new_id

    def metrics(self, bb1, bb2):
        return metrics(bb1, bb2)

    def compare_embeddings(self, ft1, ft2):
        """Returns the cosine distance between ft1 and ft2"""
        dot_product = ft1.dot(ft2)
        norm1 = np.linalg.norm(ft1)
        norm2 = np.linalg.norm(ft2)
        return dot_product / (norm1 * norm2)

    def advance(self, new_bounding_boxes, new_frame):
        """
        Takes the bounding boxes and returns an id for each one
        If the number of boxes doesn't match the previous task, then new ids
        are assigned for non matched bounding boxes or ids are dropped
        if the number is lower.

        Parameters
        ----------
            new_bounding_boxes: [Tuple(int, 4)]
            new_frame: [int] The new image frame
        Returns
        -------
            ids: [int]
        """
        
        # Batch all new boxes
        new_embeddings = self.encoder(new_frame, new_bounding_boxes)

        if self.current_task_id == 0:  # first task
            self.tracking = new_bounding_boxes
            self.ids = list(range(len(new_bounding_boxes)))
            self.id_counter = len(new_bounding_boxes)
            self.current_task_id += 1
            self.embeddings = new_embeddings
            return self.ids

        new_matrix = np.zeros(
            (len(self.tracking), len(new_bounding_boxes)),
            dtype=np.float32)

        for i, bounding_box in enumerate(new_bounding_boxes):
            for j, tracked_bounding_box in enumerate(self.tracking):
                iou, hdiff, wdiff, cd = metrics(bounding_box, tracked_bounding_box)
                score = self.compare_embeddings(self.embeddings[j], new_embeddings[i,])
                new_matrix[j, i] = iou+score
                # new_matrix[j, i] = iou if iou !=0 else -cd/1000

        row_idx, col_idx = linear_sum_assignment(new_matrix, maximize=True)
        self.current_task_id += 1
        new_ids = [
            self.new_id() if i not in col_idx else 0
            for i in range(len(new_bounding_boxes))]  # init new ids
        for row, col in zip(row_idx, col_idx):
            new_ids[col] = self.ids[row]  # reuse id if match
        self.tracking = new_bounding_boxes
        self.ids = new_ids
        self.embeddings = new_embeddings

        return self.ids, new_bounding_boxes
