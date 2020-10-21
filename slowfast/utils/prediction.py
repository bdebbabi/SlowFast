from slowfast.utils.misc import get_class_names
from slowfast.utils.parser import load_config, parse_args
import numpy as np
import json 
args = parse_args()

cfg = load_config(args)
class_names_path = cfg.DEMO.LABEL_FILE_PATH #ava_classnames.json
mode = cfg.DEMO.VIS_MODE #top-k or thres
common_class_thres = cfg.DEMO.COMMON_CLASS_THRES #0.7
uncommon_class_thres = cfg.DEMO.UNCOMMON_CLASS_THRES #0.3
common_class_names = cfg.DEMO.COMMON_CLASS_NAMES
top_k = cfg.TENSORBOARD.MODEL_VIS.TOPK_PREDS #1
predictions_path = cfg.DEMO.PREDICTIONS_FILE_PATH
scores_path = cfg.DEMO.SCORES_FILE_PATH

def get_best_predictions(scores):
    """
        Choose which predictions to keep based on the config file
        Args:
            scores (dict): dict contains the score for each class
    """
    if mode == 'top-k':
        if top_k>len(scores):
            return scores
        else:
            return {class_name:score for (class_name, score) in list(scores.items())[0:top_k]}
    elif mode == 'thres':
        temp = scores.copy()
        for class_name, score in temp.items():
            if score < common_class_thres and class_name in common_class_names:
                scores.pop(class_name)
            elif score < uncommon_class_thres and class_name not in common_class_names:
                scores.pop(class_name)
        return scores

class_names, _, _ = get_class_names(class_names_path)
predictions = []
file = np.genfromtxt(scores_path,delimiter=',')

for line in file:
    prediction_scores = {class_name:score for class_name, score in zip(class_names, line[6:])}
    prediction_scores = dict( sorted(prediction_scores.items(),
                           key=lambda item: item[1],
                           reverse=True))
    prediction_scores = get_best_predictions(prediction_scores)                           
    
    box_coords = {'top_left_x':line[2],
                  'top_left_y':line[3],
                  'bottom_right_x':line[4],
                  'bottom_right_y':line[5]}
    predictions.append({'task_id':int(line[0]),
                        'box_id': int(line[1]), 
                        'box':box_coords,
                        'predictions':prediction_scores})

with open(predictions_path, 'w') as f:
    json.dump(predictions, f, indent=4)

