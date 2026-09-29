"""
detector.py
-----------
Wraps OpenCV's DNN module around a pre-trained YOLOv4-tiny model.

Concepts explained:

- YOLO ("You Only Look Once") is a neural network that was already
  trained (by its creators, on the COCO dataset of ~330,000 labeled
  images covering 80 everyday object classes) to detect objects in a
  single pass over an image. We are NOT training anything here - we
  load their finished, ready-made model and just ask it questions.

- A "blob" is how OpenCV feeds an image into a neural network: the
  image is resized to a fixed size (416x416 here), pixel values are
  scaled to 0-1, and the color channel order is adjusted. Think of it
  as "packaging" the image the way the network expects to receive it.

- The model doesn't return neat boxes directly - it returns hundreds
  of raw candidate detections, many overlapping or low-confidence.
  We filter out low-confidence ones, then apply Non-Maximum
  Suppression (NMS): when several boxes overlap heavily on the same
  object, NMS keeps only the single best one and discards the rest.
"""

import cv2
import numpy as np

CONFIDENCE_THRESHOLD = 0.5   # ignore detections the model is less than 50% sure about
NMS_THRESHOLD = 0.4          # how much overlap counts as "the same object"
INPUT_SIZE = 416             # YOLOv4-tiny's expected input width/height


def load_model(weights_path: str, config_path: str, names_path: str):
    """Load the YOLO network and the list of class names it can detect."""
    net = cv2.dnn.readNet(weights_path, config_path)

    layer_names = net.getLayerNames()
    out_layer_indices = net.getUnconnectedOutLayers()
    output_layers = [layer_names[i - 1] for i in out_layer_indices]

    with open(names_path, "r") as f:
        classes = [line.strip() for line in f.readlines()]

    return net, output_layers, classes


def detect_objects(net, output_layers, classes, frame):
    """
    Run YOLO on a single video frame.
    Returns a list of dicts: {"box": (x, y, w, h), "label": str, "confidence": float}
    """
    height, width = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(
        frame, 1 / 255.0, (INPUT_SIZE, INPUT_SIZE), swapRB=True, crop=False
    )
    net.setInput(blob)
    outputs = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = int(np.argmax(scores))
            confidence = float(scores[class_id])

            if confidence > CONFIDENCE_THRESHOLD:
                # YOLO gives box center (x, y) and size (w, h) as
                # fractions of the image size - convert to pixels.
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                box_w = int(detection[2] * width)
                box_h = int(detection[3] * height)
                x = int(center_x - box_w / 2)
                y = int(center_y - box_h / 2)

                boxes.append([x, y, box_w, box_h])
                confidences.append(confidence)
                class_ids.append(class_id)

    # Non-Maximum Suppression to remove duplicate overlapping boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

    detections = []
    if len(indices) > 0:
        for i in indices.flatten():
            detections.append(
                {
                    "box": tuple(boxes[i]),
                    "label": classes[class_ids[i]],
                    "confidence": confidences[i],
                }
            )
    return detections
