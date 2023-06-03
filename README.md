# Object Detection with YOLO on COCO Dataset

This project demonstrates the use of the YOLO (You Only Look Once) object detection model on the COCO dataset. It processes images from the dataset and provides object detection results. The model used is a YOLOv8 Large model for better accuracy.

## Prerequisites

The project requires the following packages:

- `cv2`
- `json`
- `glob`
- `pycocotools`
- `ultralytics` YOLO

## Installation

To run the project, follow these steps:

1. Clone the repository:

    ```
    git clone https://github.com/thisishamody/yolo_paper_project.git
    ```

2. Install the necessary dependencies:

    ```
    pip install opencv-python pycocotools glob2 ultralytics
    ```

3. Download the model weights and place them in the `Yolo-Weights` directory. You can download them from [here](https://github.com/ultralytics/yolov5/releases). In this project, we use `yolov8l.pt` for better accuracy.

## Usage

The `main.py` script runs the object detection process on images located in `coco/val2017`.

The script evaluates the model's predictions and saves them in a JSON file named `predictions.json`. It also calculates the mean Average Precision (mAP) for the model's predictions.

To run the script:

