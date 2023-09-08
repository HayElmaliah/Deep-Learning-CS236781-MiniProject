# Object Detection with YOLOv8 on TACO Dataset

## Overview

This project aims to explore the capabilities of the YOLOv8 (You Only Look Once version 8) model for object detection tasks, specifically focusing on the TACO (Trash Annotations in Context) dataset. The project includes various experiments to fine-tune the model, evaluate its performance, and understand its robustness and limitations.

## Dataset

Please note that the TACO dataset is not included in this repository due to its size. To download and prepare the dataset for the project, follow the instructions provided in the `./project/YOLOv8/YOLO_Fine_Tuning.ipynb` notebook.

## Notebooks

### Main Notebook: `./Part1_MiniProject.ipynb`

This Jupyter Notebook serves as the central document for the project and is structured as follows:

- **Introduction**: Brief overview of the problem statement and objectives.
- **Experiments and Results**: Comprehensive experiments to fine-tune the model and evaluate its performance.
  - **Choosing the Optimizer**: Initial phase to select the most suitable optimizer.
  - **Mini Grid Search Over Hyperparameters**: Fine-tuning the selected optimizer's parameters.
  - **Frozen vs Non-Frozen Backbone**: Comparing the performance of models with frozen and non-frozen backbone layers.
- **Testing Phase**: Results of the testing phase, including various metrics like precision, recall, and mAP.
  - **IoU Thresholds Analysis**: As part of the testing phase, an investigation into the impact of varying IoU thresholds on model performance is included.
- **Future Work**: Discussion on potential future directions for the project.
- **Conclusion**: Final thoughts and summary of the project.

### Data Preparation and Model Training: `./project/YOLOv8/YOLO_Fine_Tuning.ipynb`

For details on data preparation, model training, and testing, please refer to this notebook. It contains the technical steps for setting up the dataset and training the YOLOv8 model.

### Theoretical Questions Notebook: `./Part2_TheoreticalQuestions.ipynb`

This notebook is dedicated to addressing various theoretical questions related to object detection, machine learning, and data science. The questions are answered in a detailed manner to facilitate a better understanding of the concepts involved.
