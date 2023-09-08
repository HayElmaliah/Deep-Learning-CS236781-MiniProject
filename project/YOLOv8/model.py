import os
import csv
import json
import shutil
import ultralytics
from ultralytics import YOLO

def initialize_model(model_path='yolov8n.pt'):
    return YOLO(model_path)

def train_custom_model(model, save_dir, yaml_path, extra_params={}):
    if os.path.exists(os.path.join(save_dir, 'weights', 'best.pt')):
        print(f"Model already trained and saved in {save_dir}.")
        return None
    try:
        loaded_model = YOLO(os.path.join(save_dir, 'weights', 'best.pt'))
        print("Using cached model.")
        return loaded_model
    except FileNotFoundError:
        print("Training new model...")
        training_results = model.train(data=yaml_path, imgsz=640, cache=True, name=save_dir, **extra_params)
        os.makedirs(os.path.join(save_dir, 'weights'), exist_ok=True)
        src_dir = os.path.join('runs', 'detect', save_dir)
        dest_dir = save_dir
        for filename in os.listdir(src_dir):
            src_path = os.path.join(src_dir, filename)
            dest_path = os.path.join(dest_dir, filename)
            if os.path.exists(src_path) and src_path != dest_path:
                shutil.copy(src_path, dest_path)
        return model

def freeze_layer(trainer):
    model = trainer.model
    num_freeze = 10
    print(f"Freezing {num_freeze} layers")
    freeze = [f'model.{x}.' for x in range(num_freeze)]  # layers to freeze 
    for k, v in model.named_parameters(): 
        v.requires_grad = True  # train all layers 
        if any(x in k for x in freeze): 
            print(f'freezing {k}') 
            v.requires_grad = False 
    print(f"{num_freeze} layers are freezed.")

def evaluate_model_performance(
    model,
    save_dir,
    iou_threshold=0.6
):
    result_file = os.path.join(save_dir, f"IoU_threshold_{iou_threshold}_results.json")
    try:
        with open(result_file, 'r') as f:
            results = json.load(f)
        print(f"Using cached results from {result_file}.")
    except FileNotFoundError:
        print("Evaluating model...")
        eval_results = model.val(split='test', iou=iou_threshold)
        src_dir = os.path.join('runs', 'detect', 'val')
        dest_dir = os.path.join(save_dir, f"IoU_threshold_{iou_threshold}")
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        for filename in os.listdir(src_dir):
            src_path = os.path.join(src_dir, filename)
            dest_path = os.path.join(dest_dir, filename)
            if os.path.exists(src_path):
                shutil.move(src_path, dest_path)
        shutil.rmtree('runs')
        csv_file_path = os.path.join(dest_dir, f"IoU_threshold_{iou_threshold}_results.csv")
        with open(csv_file_path, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['Metric', 'Value'])
            for key, value in eval_results.results_dict.items():
                csvwriter.writerow([key, value])
    return eval_results
