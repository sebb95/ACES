Aces project.

README file starter pack
General description
configs/ → system configuration
data/ → input
outputs/ → generated artifacts
src/ → implementation

Configugation ACES/configs
configs/
The configs directory contains configuration files for different system components such as camera setup, counting logic, model configuration, and user interface settings. These files define system parameters and runtime behavior without modifying source code. The directory enables centralized management of configuration settings and supports easier deployment and reproducibility of the system.

Dataset description ACES/data
The dataset consists of labeled fish images prepared in YOLO segmentation format. Each image has a corresponding label file that describes the polygon outline of each fish instance. The dataset is organized into training and validation subsets, containing image files and annotation files. The annotations include class identifiers corresponding to fish species and normalized coordinates representing segmentation masks.
Annotation format
Each annotation file contains segmentation information for the objects present in the corresponding image. Each line represents one fish instance and begins with a class identifier followed by normalized coordinates describing the polygon outline of the object. These coordinates define the segmentation mask used during training.
Dataset configuration file (dataset.yaml)
The dataset is configured through a YAML file that defines dataset paths and class names used during training. The dataset configuration file defines the location of the dataset and the mapping between numeric class identifiers and fish species names. It specifies the root dataset directory and the relative paths to training and validation image folders. The file is used by the training pipeline to locate data and interpret annotation labels.
Output structure ACES/outputs
The project stores generated artifacts under the outputs directory. Training and inference runs are saved under outputs/runs, including visualizations, metrics, and run-specific files. Stable trained model files are copied to outputs/weights so they can be reused easily in later inference and evaluation steps.
outputs/runs
Contains run-specific artifacts generated during training or inference, such as plots, validation previews, prediction images, and intermediate model files.
outputs/weights
Contains reusable trained model files, such as the best-performing checkpoint selected from a training run.
SRC - implementation ACES/src
Common: ACES/src/common
The common module contains shared utilities and helper functions used across multiple components of the system. This includes generic functionality such as configuration handling, file utilities, logging helpers, and other reusable tools that are not specific to a single subsystem.
Purpose: avoid code duplication, provide shared infrastructure for other modules
Machine Learning: ACES/src/ml
The ml module contains the machine learning components used for fish detection and analysis. It includes model training, dataset preparation, inference logic, and supporting utilities related to the AI pipeline. The module is responsible for developing, training, and evaluating the neural network models used in the system.
Purpose:
model development
training pipelines
inference logic
dataset handling

Model training ACES/src/ml/train:

config.py
The configuration file is currently stored close to the training code because it only defines parameters for the machine learning training and inference pipeline. This keeps the baseline block self-contained and easier to develop, test, and understand. If the project later expands into a larger unified configuration system, the file can be moved into a shared configuration module.
This file defines the configuration used by the training and inference pipeline. It stores model selection, dataset paths, training hyperparameters, output directories, and inference settings in one central place. This avoids hardcoding values directly in the training script and makes the pipeline easier to adjust, reproduce, and document.
dataset.py
Before starting a training run, the dataset structure is validated programmatically. The validation step verifies that the dataset configuration file exists, checks that required keys are present, and confirms that the expected image and label directories are available. This prevents training runs from failing due to incorrect dataset paths or incomplete dataset preparation. This file does not load data. Ultralytics handle this internally. 
train_baseline.py
This script is the entry point for baseline model training. It loads the training configuration, validates the dataset, creates the required output folders, launches YOLO training, and saves the best model checkpoint to a stable output location for later inference and evaluation.
model.py
The model module handles creation and loading of YOLO model instances used in the training and inference pipeline. It provides a simple interface for initializing a pretrained model for training and loading trained model weights for inference. This separation keeps model management independent from training logic and improves modularity of the machine learning pipeline.



src/vision
Description
The vision module implements the computer vision pipeline responsible for detecting, tracking, and counting fish in image or video streams. It processes visual input from the camera system and applies trained machine learning models to identify fish instances and produce structured outputs used by downstream components.
Purpose:
image and video processing
object detection / segmentation
tracking and counting logic

src/telemetry
The telemetry module handles collection and processing of system telemetry and operational data. This may include runtime statistics, processing performance metrics, and other system signals that help monitor the behavior and performance of the system during operation.
Purpose:
runtime monitoring
performance tracking
system diagnostics

