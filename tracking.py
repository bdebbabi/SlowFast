"""
Tracking from json extract
"""
import json

from tqdm import tqdm

from slowfast.utils.hungarian_tracker import HungarianTracker
from slowfast.utils.parser import load_config, parse_args


def reduce_tasks(tasks):
    def transform_bbox(bbox):
        """
        Transform the box back to a tensor
        """
        return (bbox['top_left_x'], bbox['top_left_y'],
                bbox['bottom_right_x'], bbox['bottom_right_y'])
    reduced_tasks = {}
    for pred in tasks:
        bbox = transform_bbox(pred['box'])
        if pred['task_id'] not in reduced_tasks:
            reduced_tasks[pred['task_id']] = [bbox]
        else:
            reduced_tasks[pred['task_id']].append(bbox)
    return reduced_tasks


def main(predictions_path):
    """
    Performs a hungarian traking on each tasks
    """
    with open(predictions_path, 'r') as json_file:
        data = json.load(json_file)
    data = reduce_tasks(data)

    tracker = HungarianTracker()
    for task_id in tqdm(data):
        ids = tracker.advance(data[task_id])
        print(f">> task {task_id} ids = {ids}")


if __name__ == "__main__":
    ARGS = parse_args()
    CFG = load_config(ARGS)
    PREDICTIONS_PATH = CFG.DEMO.PREDICTIONS_FILE_PATH
    main(PREDICTIONS_PATH)
