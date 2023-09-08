import os
import cv2
import csv
import json
import tqdm
import yaml
import random
import shutil
import requests
import numpy as np
import splitfolders
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
from PIL import Image
from os.path import join
from matplotlib import gridspec
from pycocotools.coco import COCO
from IPython.core.display import display, HTML


def read_taco_data_from_server() -> COCO:
    path = '/datasets/TACO-master/data/annotations.json'
    with open(path, 'r') as f:
        annotations = json.load(f)
    coco = COCO(path)
    return coco

def analyze_coco_data(coco_data: COCO) -> None:
    """
    Analyzes and visualizes key statistics and category distribution from a COCO dataset.

    Args:
    - coco_data (COCO): The COCO dataset to analyze.
    """

    print("="*50)
    print("Dataset Overview:")
    print("="*50)

    # Basic Information Extraction
    all_categories = coco_data.getCatIds()
    all_images = coco_data.getImgIds()
    all_annotations = coco_data.getAnnIds()

    print(f"-> Total Categories: {len(all_categories)}\n"
          f"-> Total Images: {len(all_images)}\n"
          f"-> Total Annotations: {len(all_annotations)}")

    print("="*50)
    print("Category Distribution:")
    print("="*50)

    # Category Distribution Analysis
    categories = [category['name'] for category in coco_data.loadCats(all_categories)]
    category_counts = np.zeros(len(all_categories), dtype=int)
    
    for annotation_id in all_annotations:
        annotation_info = coco_data.loadAnns(annotation_id)
        category_counts[annotation_info[0]['category_id']] += 1

    # Visualization
    sns.set_style("darkgrid")

    data_frame = pd.DataFrame({'Categories': categories, 'Annotation Counts': category_counts})
    data_frame = data_frame.sort_values('Annotation Counts', ascending=False)

    plt.figure(figsize=(10, 10))
    plt.title('Histogram: Distribution of Categories')
    sns.barplot(x="Annotation Counts", y="Categories", data=data_frame, palette="coolwarm")
    plt.tight_layout()
    plt.show()

def show_directory_structure():
    html_code = '''
    <style>
      .directory-structure {
        font-family: monospace;
        white-space: pre;
        border: 2px solid #007BFF;
        padding: 20px;
        background-color: #f1f1f1;
        border-radius: 12px;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
      }
      .line {
        border-bottom: 2px dashed #339CFF;
        margin: 8px 0;
      }
      .directory-structure div {
        color: #333;
      }
      .directory-structure .comment {
        color: #228B22;
      }
    </style>
    <div class="directory-structure">
      mini_project/ <span class="comment"># Main folder</span>
      <div class="line"></div>
      ├── Part1_MiniProject.ipynb <span class="comment"># Main notebook for analyzing the wet part</span>
      <div class="line"></div>
      ├── Part2_TheoreticalQuestions.ipynb <span class="comment"># Main notebook for theoretical questions</span>
      <div class="line"></div>
      ├── imgs/ <span class="comment"># Images data folder</span>
      │   └── taco_data_split/ <span class="comment"># Split dataset</span>
      │       ├── train/ <span class="comment"># Training data</span>
      │       │   ├── images/
      │       │   └── labels/
      │       ├── val/ <span class="comment"># Validation data</span>
      │       │   ├── images/
      │       │   └── labels/
      │       └── test/ <span class="comment"># Test data</span>
      │           ├── images/
      │           └── labels/
      <div class="line"></div>
      ├── project/ <span class="comment"># Code folder</span>
      │   ├── utils.py <span class="comment"># Utility functions mainly for displays</span>
      │   └── YOLOv8/ <span class="comment"># YOLOv8 specific code</span>
      │       ├── config.yaml <span class="comment"># Configuration for YOLO training and validation</span>
      │       ├── main.py <span class="comment"># Main script for running on the server</span>
      │       ├── YOLO_Fine_Tuning.ipynb <span class="comment"># Main Notebook for fine-tuning YOLO</span>
      │       ├── model.py <span class="comment"># YOLOv8 model scripts for training and testing</span>
      │       └── yolo.sh <span class="comment"># Shell script for running the model</span>
      <div class="line"></div>
      ├── results/ <span class="comment"># Results folder for each experiment's results</span>
      <div class="line"></div>
      └── tests/ <span class="comment"># Tests folder with testing results</span>
    </div>
    '''
    display(HTML(html_code))

def display_optimizers_experiment_metrics():
    sns.set_style("darkgrid")
    csv_paths = {
        'SGD': os.path.join(os.getcwd(), 'results/Optimizers_exp/SGD2/results.csv'),
        'Adam': os.path.join(os.getcwd(), 'results/Optimizers_exp/Adam/results.csv'),
        'AdamW': os.path.join(os.getcwd(), 'results/Optimizers_exp/AdamW/results.csv'),
        'RMSProp': os.path.join(os.getcwd(), 'results/Optimizers_exp/RMSProp2/results.csv')
    }
    fig, ax = plt.subplots(4, 1, figsize=(15, 20), tight_layout=True)
    ax = ax.ravel()
    metrics = ['metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)']
    titles = ['Precision', 'Recall', 'mAP50', 'mAP50-95']
    for i, metric in enumerate(metrics):
        for label, path in csv_paths.items():
            df = pd.read_csv(path)
            df.columns = df.columns.str.strip()
            x = df['epoch']
            y = df[metric]
            ax[i].plot(x, y, marker='.', label=label, linewidth=2, markersize=8)
            ax[i].set_xticks(x[::10])
            ax[i].set_title(titles[i], fontsize=18)
            ax[i].legend()
    plt.show()

def display_confusion_matrices(folder_path, title):
    sns.set(style="white")
    img_path = os.path.join(folder_path, 'confusion_matrix.png')
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.axis("off")
    plt.title(title, fontsize=18, fontweight='bold')
    sns.despine(left=True, bottom=True)
    plt.show()

def display_val_images(folder_path, title):
    sns.set(style="whitegrid")
    img_label_path = os.path.join(folder_path, 'val_batch2_labels.jpg')
    img_pred_path = os.path.join(folder_path, 'val_batch2_pred.jpg')
    img_label = cv2.imread(img_label_path)
    img_pred = cv2.imread(img_pred_path)
    img_label = cv2.cvtColor(img_label, cv2.COLOR_BGR2RGB)
    img_pred = cv2.cvtColor(img_pred, cv2.COLOR_BGR2RGB)
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    axes[0].imshow(img_label)
    axes[0].set_title(f"{title} Ground Truth", fontsize=16, fontweight='bold')
    axes[0].axis("off")
    axes[1].imshow(img_pred)
    axes[1].set_title(f"{title} Prediction", fontsize=16, fontweight='bold')
    axes[1].axis("off")
    sns.despine(left=True, bottom=True)
    plt.show()

def display_tuning_experiment_metrics():
    path = os.path.join(os.getcwd(), 'results/HyperParams_tuning_exp/yolo_hyperparams_tuning.csv')
    df = pd.read_csv(path)
    
    def style_hyperparams(val):
        """Apply styling to hyperparameter columns"""
        color = sns.diverging_palette(220, 20, as_cmap=True)(val)
        return f'background-color: {color};'

    def style_metrics(val):
            """Apply styling to metric columns"""
            color = sns.light_palette("green", as_cmap=True)(val)
            return f'background-color: {color};'

    sns.set(style="whitegrid")
    table_styles = [{
        'selector': 'th',
        'props': [('font-size', '7pt')]
    }, {
        'selector': 'td',
        'props': [('font-size', '7pt')]
    }]
    styled_df = df.style.background_gradient(cmap=sns.diverging_palette(220, 20, as_cmap=True), subset=['lr0', 'momentum', 'batch']) \
                       .background_gradient(cmap=sns.light_palette("#79C", as_cmap=True), subset=['train/box_loss', 'train/cls_loss', 'train/dfl_loss', 'precision', 'recall', 'mAP50', 'mAP50-95', 'val/box_loss', 'val/cls_loss', 'val/dfl_loss']) \
                       .set_table_styles(table_styles)
    return styled_df

def display_tuning_experiment_heatmap():
    path = os.path.join(os.getcwd(), 'results/HyperParams_tuning_exp/yolo_hyperparams_tuning.csv')
    df = pd.read_csv(path)
    df_filtered = df[df['momentum'] == 0.97]
    pivot_table = df_filtered.pivot_table(values='mAP50', index='lr0', columns='batch')
    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1)
    sns.heatmap(pivot_table, annot=True, cmap="Blues", fmt=".2f", linewidths=.5)
    plt.title('Impact of learning Rate and Batch Size on mAP50 (Momentum=0.97)')
    plt.xlabel('Batch Size')
    plt.ylabel('learning Rate')
    plt.show()

def plot_image(image_path, figsize=(10, 10)):
    plt.figure(figsize=figsize)
    img = mpimg.imread(image_path)
    plt.imshow(img)
    plt.axis('off')
    plt.show()

def display_tuning_experiment_chosen_metrics():
    plot_image(os.path.join(os.getcwd(), 'results/HyperParams_tuning_exp/lr_0.001_momentum_0.97_batch_size_82/results.png'), figsize=(15, 15))

def display_tuning_experiment_val_images():
    display_val_images(os.path.join(os.getcwd(), 'results/HyperParams_tuning_exp/lr_0.001_momentum_0.97_batch_size_82'), 'HyperParameter tuning result')

def display_freezing_experiment_chosen_metrics():
    plot_image(os.path.join(os.getcwd(), 'results/Freezing_layers_exp/frozen_yolo_SGD_lr0_0.001_momentum_0.97_batch_83/results.png'), figsize=(15, 15))

def display_freezing_experiment_val_images():
    display_val_images(os.path.join(os.getcwd(), 'results/Freezing_layers_exp/frozen_yolo_SGD_lr0_0.001_momentum_0.97_batch_83'), 'Freezing result')

def display_IoU_analysis_metrics():
    final_df = pd.DataFrame()
    folders = [f"IoU_threshold_{number}" for number in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]]
    for folder in folders:
        folder = os.path.join(os.getcwd(), 'tests' + '/' + folder)
        csv_path = os.path.join(folder, f"IoU_threshold_{folder.split('_')[-1]}_results.csv")
        df = pd.read_csv(csv_path)
        iou_thresh = 'IoU - ' + folder.split('_')[-1]
        df.rename(columns={'Value': iou_thresh}, inplace=True)
        df['Metric'] = df['Metric'].str.replace('metrics/', '').str.replace('(B)', '')
        if final_df.empty:
            final_df = df
        else:
            final_df = pd.merge(final_df, df[['Metric', iou_thresh]], on='Metric', how='left')
        final_df = final_df[final_df['Metric'] != 'fitness']
    df = final_df
    table_styles = [{'selector': 'th', 'props': [('font-size', '10pt')]}, {'selector': 'td', 'props': [('font-size', '10pt')]}]
    return df.style.background_gradient(cmap=sns.light_palette("#79C", as_cmap=True), subset=df.columns[1:]).set_table_styles(table_styles)

def display_IoU_analyis_bars():
    final_df = pd.DataFrame()
    folders = [f"IoU_threshold_{number}" for number in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]]
    for folder in folders:
        folder = os.path.join(os.getcwd(), 'tests' + '/' + folder)
        csv_path = os.path.join(folder, f"IoU_threshold_{folder.split('_')[-1]}_results.csv")
        df = pd.read_csv(csv_path)
        iou_thresh = folder.split('_')[-1]
        df.rename(columns={'Value': iou_thresh}, inplace=True)
        if final_df.empty:
            final_df = df
        else:
            final_df = pd.merge(final_df, df[['Metric', iou_thresh]], on='Metric', how='left')
        final_df = final_df[final_df['Metric'] != 'fitness']
    df = final_df
    df['Metric'] = df['Metric'].str.replace('metrics/', '').str.replace('(B)', '')
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    for i, metric in enumerate(df['Metric']):
        ax = axes[i-1]
        df_filtered = df[df['Metric'] == metric]
        df_filtered = df_filtered.drop(columns=['Metric']).T
        df_filtered.columns = ['Value']
        df_filtered['IoU Threshold'] = df_filtered.index
        sns.barplot(x='IoU Threshold', y='Value', data=df_filtered, ax=ax, palette='Blues')
        ax.set_title(f'{metric} Across Different IoU Thresholds', fontsize=16)
        ax.set_xlabel('IoU Threshold', fontsize=14)
        ax.set_ylabel(f'{metric} Value', fontsize=14)
    plt.tight_layout()
    plt.show()

def display_testing_images():
    display_val_images(os.path.join(os.getcwd(), 'tests/IoU_threshold_0.6'), 'Testing result')

def read_metrics_from_csv(file_path):
    return pd.read_csv(file_path)

def style_metrics(df):
    def highlight_max(s):
        is_max = s == s.max()
        return ['background-color: yellow' if v else '' for v in is_max]
    df['Metric'] = df['Metric'].str.replace('metrics/', '').str.replace('(B)', '')
    styled_df = df.style.apply(highlight_max, subset=['Value']) \
                        .background_gradient(cmap='Blues', subset=['Value']) \
                        .set_table_styles({
                            'Metric': [{'selector': 'td:hover',
                                        'props': [('font-size', '25px')]}],
                            'Value': [{'selector': 'td:hover',
                                       'props': [('font-size', '25px')]}]
                        })
    return styled_df

def display_testing_metrics():
    file_path = os.path.join(os.getcwd(), 'tests/IoU_threshold_0.6/IoU_threshold_0.6_results.csv')
    df = read_metrics_from_csv(file_path)
    return style_metrics(df)
