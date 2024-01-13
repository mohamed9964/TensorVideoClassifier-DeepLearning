
# Advanced Video Data Analysis and Binary Classification Using Deep Learning Models

## Project Overview
Our project delves into the realm of video data analysis, handling tensor-formatted video data featuring 18 diverse object categories. This rich dataset includes bounding boxes, coordinates, segmentation, and depth masks. We aim to apply this data in binary classification tasks, exploring a range of advanced machine learning models and techniques.

## Key Phases of the Project
- **Data Loading:** Loading tensor-formatted videos using "project.ipynb" methodology.
- **Frame Division:** Breaking down videos into frames for easier processing.
- **Object Category Extraction:** Deriving labels for binary classification from each frame.
- **Data Exploration & Visualization:** Selecting a class for binary classification.
- **Label Modification:** Adjusting labels to '1' for the chosen class and '0' for others.
- **Preprocessing:** Techniques to prepare data for classification.
- **Model Evolution & Importance:** Progression of binary classification models.
- **Advanced Model Utilization:** Implementing models like MobileNet and YOLO.
- **Ablation Studies:** Analyzing the effect of design choices and hyperparameters.
- **Parameter Effects Analysis:** Exploring learning rates, batch sizes, and image sizes.

## Technologies and Libraries Used
- Python
- TensorFlow, TensorFlow Datasets
- Keras
- NumPy, Pandas
- Matplotlib, Seaborn
- Scikit-image, Scikit-learn
- Ultralytics YOLO

## Data Loading and Preparation
Detailed steps and methodology for loading and preparing the dataset using TensorFlow Datasets, and the process for converting and merging datasets into DataFrames.

## Model Training and Evaluation
Description of the process for building, training, and evaluating various models (ResNet50, InceptionV3, InceptionResNetV2, MobileNetV2, and YOLO v8), including an in-depth analysis of model performance and metrics.

## Ablation Study
Analysis of model performance under various hyperparameters, including learning rates, batch sizes, and image sizes. Insights into the impact of these parameters on model efficiency and effectiveness.



## Contact Information
- [m.hanyy996@gmail.com]
