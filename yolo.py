import cv2
import json
import os
import glob
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics import YOLO

# Load the model, yolov8l for large, yolov8n for nano, nano runs way faster but Large have better accuracy
model = YOLO("../Yolo-Weights/yolov8l.pt")

# datasset
dataset_path = "/Users/hamodyx0/Documents/school/projectttt/coco/val2017"

# Get the images in the dataset directory
image_files = glob.glob(os.path.join(dataset_path, "*.jpg"))

# For each image in the directory
preds = []
for image_file in image_files:
    # Read the image
    img = cv2.imread(image_file)

    # The detections
    results = model(img,{'conf_thres': 0.15})
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Confidence
            conf = float(box.conf[0])

            # Class Name
            cls = int(box.cls[0])

            # Append the prediction to the list
            preds.append({
                "image_id": int(os.path.basename(image_file).split(".")[0]),
                "category_id": cls,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": conf
            })

# Save predictions to a JSON file
with open('predictions.json', 'w') as f:
    json.dump(preds, f)

# Evaluate mAP
gt_annotations_file = '/Users/hamodyx0/Documents/school/projectttt/coco/instances_val2017.json'  # Replace this with your ground truth annotations file
pred_annotations_file = 'predictions.json'  # The predictions file we just created

cocoGt = COCO(gt_annotations_file)
cocoDt = cocoGt.loadRes(pred_annotations_file)

cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()

# Print mAP
print("mAP@.5: ", cocoEval.stats[0])
print("mAP@.5:.95: ", cocoEval.stats[1])
